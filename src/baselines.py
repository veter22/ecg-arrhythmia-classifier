"""
Сравнение классических ML-бейзлайнов и абляция архитектуры на DS1 -> DS2
=========================================================================
Запуск: python -m src.baselines
"""

import json
import sys
from pathlib import Path
from typing import Dict

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader
import yaml

from src.contracts import IDX_TO_LABEL, NUM_CLASSES
from src.losses.focal_loss import FocalLoss, compute_effective_num_weights
from src.models.hybrid_cnn import HybridECGClassifier
from src.train import InMemoryECGDataset, load_npz_safely, resolve_dataset_path


def extract_tabular_features(signals: np.ndarray, rr: np.ndarray) -> np.ndarray:
    """Извлекает 14 табличных признаков: 4 RR-признака + 10 морфологических статистик окна."""
    sig = signals[:, 0, :] if signals.ndim == 3 else signals

    # Окно вокруг R-пика (индекс 96 из 256)
    qrs_zone = sig[:, 76:126]
    st_zone = sig[:, 126:186]

    morph_features = np.column_stack([
        np.max(sig, axis=1),
        np.min(sig, axis=1),
        np.ptp(sig, axis=1),
        np.std(sig, axis=1),
        np.mean(np.abs(sig), axis=1),
        np.max(qrs_zone, axis=1) - np.min(qrs_zone, axis=1),
        np.std(qrs_zone, axis=1),
        np.mean(st_zone, axis=1),
        np.std(st_zone, axis=1),
        np.sum(np.diff(sig, axis=1) ** 2, axis=1),
    ]).astype(np.float32)

    return np.hstack([rr.astype(np.float32), morph_features])


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """Считает Macro-F1 и F1 по каждому классу."""
    all_labels = list(range(NUM_CLASSES))
    class_names = [IDX_TO_LABEL[i] for i in all_labels]
    macro_f1 = float(f1_score(y_true, y_pred, labels=all_labels, average="macro", zero_division=0))
    report = classification_report(
        y_true, y_pred, labels=all_labels, target_names=class_names, output_dict=True, zero_division=0
    )
    return {
        "macro_f1": round(macro_f1, 4),
        "N_f1": round(float(report["N"]["f1-score"]), 4),
        "S_f1": round(float(report["S"]["f1-score"]), 4),
        "V_f1": round(float(report["V"]["f1-score"]), 4),
        "F_f1": round(float(report["F"]["f1-score"]), 4),
    }


def train_cnn_only_ablation(
    X_sig_train: np.ndarray,
    X_rr_train: np.ndarray,
    y_train: np.ndarray,
    pids_train: np.ndarray,
    X_sig_test: np.ndarray,
    X_rr_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int = 25,
    seed: int = 42,
) -> Dict:
    """Обучает HybridECGClassifier с отключённой RR-веткой (нулевой тензор вместо RR)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gkf = GroupKFold(n_splits=5)
    tr_idx, va_idx = next(gkf.split(X_sig_train, y_train, groups=pids_train))

    class_counts = [int(np.sum(y_train[tr_idx] == c)) for c in range(NUM_CLASSES)]
    alpha = compute_effective_num_weights(class_counts, beta=0.999).to(device)
    criterion = FocalLoss(alpha=alpha, gamma=2.0)

    zeros_tr = np.zeros_like(X_rr_train)
    zeros_te = np.zeros_like(X_rr_test)

    train_loader = DataLoader(
        InMemoryECGDataset(X_sig_train[tr_idx], zeros_tr[tr_idx], y_train[tr_idx]),
        batch_size=128,
        shuffle=True,
    )
    val_loader = DataLoader(
        InMemoryECGDataset(X_sig_train[va_idx], zeros_tr[va_idx], y_train[va_idx]),
        batch_size=256,
        shuffle=False,
    )
    test_loader = DataLoader(
        InMemoryECGDataset(X_sig_test, zeros_te, y_test),
        batch_size=256,
        shuffle=False,
    )

    model = HybridECGClassifier(num_classes=NUM_CLASSES).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_f1 = -1.0
    best_weights = None
    all_labels = list(range(NUM_CLASSES))

    for _ in range(epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            logits = model(batch["signal"].to(device), batch["rr"].to(device))
            loss = criterion(logits, batch["label"].to(device))
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                logits = model(batch["signal"].to(device), batch["rr"].to(device))
                val_preds.append(torch.argmax(logits, dim=1).cpu().numpy())
                val_targets.append(batch["label"].numpy())

        val_f1 = float(
            f1_score(np.concatenate(val_targets), np.concatenate(val_preds), labels=all_labels, average="macro", zero_division=0)
        )
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_weights = {k: v.cpu() for k, v in model.state_dict().items()}

    model.load_state_dict(best_weights)
    model.eval()
    test_preds = []
    with torch.no_grad():
        for batch in test_loader:
            logits = model(batch["signal"].to(device), batch["rr"].to(device))
            test_preds.append(torch.argmax(logits, dim=1).cpu().numpy())

    return evaluate_predictions(y_test, np.concatenate(test_preds))


def run_baselines(config_path: str = "configs/config.yaml") -> None:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    data_cfg = cfg.get("data", {})
    train_npz = resolve_dataset_path(data_cfg, "train_file", "ds1_train.npz")
    test_npz = resolve_dataset_path(data_cfg, "test_file", "ds2_test.npz")

    X_sig_tr, X_rr_tr, y_tr, pids_tr = load_npz_safely(train_npz)
    X_sig_te, X_rr_te, y_te, _ = load_npz_safely(test_npz)

    results = {}

    # 1. Random Forest (только RR-интервалы)
    print("[1/3] Обучение Baseline 1: Random Forest (только RR-признаки)...")
    rf_rr = RandomForestClassifier(
        n_estimators=200, max_depth=12, class_weight="balanced_subsample", random_state=42, n_jobs=-1
    )
    rf_rr.fit(X_rr_tr, y_tr)
    results["RF (RR only)"] = evaluate_predictions(y_te, rf_rr.predict(X_rr_te))

    # 2. Random Forest (RR + морфология)
    print("[2/3] Обучение Baseline 2: Random Forest (RR + морфология)...")
    X_tab_tr = extract_tabular_features(X_sig_tr, X_rr_tr)
    X_tab_te = extract_tabular_features(X_sig_te, X_rr_te)
    rf_full = RandomForestClassifier(
        n_estimators=300, max_depth=14, class_weight="balanced_subsample", random_state=42, n_jobs=-1
    )
    rf_full.fit(X_tab_tr, y_tr)
    results["RF (RR + Morphology)"] = evaluate_predictions(y_te, rf_full.predict(X_tab_te))

    # 3. 1D-ResNet Only (без RR-ветки)
    print("[3/3] Обучение Baseline 3 (Абляция): 1D-ResNet без RR-ветки (25 эпох)...")
    results["1D-ResNet Only (no RR)"] = train_cnn_only_ablation(
        X_sig_tr, X_rr_tr, y_tr, pids_tr, X_sig_te, X_rr_te, y_te, epochs=25
    )

    # 4. Подгружаем финальный результат нашей гибридной модели из ds2_metrics.json
    hybrid_metrics_path = Path("reports/ds2_metrics.json")
    if hybrid_metrics_path.exists():
        with open(hybrid_metrics_path, "r", encoding="utf-8") as f:
            hm = json.load(f)
        results["Hybrid 1D-ResNet + RR-MLP (Ours)"] = {
            "macro_f1": hm["macro_f1"],
            "N_f1": hm["per_class"]["N"]["f1"],
            "S_f1": hm["per_class"]["S"]["f1"],
            "V_f1": hm["per_class"]["V"]["f1"],
            "F_f1": hm["per_class"]["F"]["f1"],
        }

    # Вывод итоговой таблицы
    print("\n=== Сравнение моделей на отложенной выборке DS2 (22 новых пациента) ===")
    print(f"{'Модель':<34} | {'Macro-F1':>8} | {'F1 (N)':>7} | {'F1 (S)':>7} | {'F1 (V)':>7} | {'F1 (F)':>7}")
    print("-" * 82)
    for name, m in results.items():
        print(
            f"{name:<34} | {m['macro_f1']:>8.4f} | {m['N_f1']:>7.4f} | "
            f"{m['S_f1']:>7.4f} | {m['V_f1']:>7.4f} | {m['F_f1']:>7.4f}"
        )

    out_path = Path("reports/baselines_comparison.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] Результаты сравнения сохранены в: '{out_path}'")


if __name__ == "__main__":
    run_baselines()