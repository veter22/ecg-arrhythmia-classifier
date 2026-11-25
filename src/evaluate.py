"""
Независимая оценка ансамбля на DS2: raw vs post-hoc метрики, N/S/V F1 и разбор по пациентам — [ML-04 v3]
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader
import yaml

from src.contracts import IDX_TO_LABEL, NUM_CLASSES, validate_batch
from src.models.hybrid_cnn import HybridECGClassifier
from src.train import (
    InMemoryECGDataset,
    generate_synthetic_ds1,
    load_npz_safely,
    resolve_dataset_path,
    resolve_path,
)


def load_ensemble_from_checkpoint(
    ckpt_file: Path,
    model_cfg: Dict,
    device: torch.device,
) -> Tuple[List[HybridECGClassifier], Dict]:
    if not ckpt_file.exists():
        raise FileNotFoundError(f"Чекпоинт '{ckpt_file}' не найден!")

    checkpoint = torch.load(ckpt_file, map_location=device, weights_only=True)
    saved_params = checkpoint.get("model_params", {})
    state_dicts = checkpoint.get("fold_state_dicts") or [checkpoint["model_state_dict"]]

    models: List[HybridECGClassifier] = []
    for sd in state_dicts:
        m = HybridECGClassifier(
            in_channels=int(saved_params.get("in_channels", model_cfg.get("in_channels", 1))),
            signal_length=int(saved_params.get("signal_length", model_cfg.get("signal_length", 256))),
            rr_features_dim=int(saved_params.get("rr_features_dim", model_cfg.get("rr_features_dim", 6))),
            cnn_embed_dim=int(saved_params.get("cnn_embed_dim", model_cfg.get("cnn_embed_dim", 128))),
            rr_embed_dim=int(saved_params.get("rr_embed_dim", model_cfg.get("rr_embed_dim", 32))),
            num_classes=int(saved_params.get("num_classes", NUM_CLASSES)),
            dropout=float(saved_params.get("dropout", model_cfg.get("dropout", 0.3))),
        ).to(device)
        m.load_state_dict(sd)
        m.eval()
        models.append(m)

    return models, checkpoint


def predict_ensemble(
    models: List[HybridECGClassifier],
    loader: DataLoader,
    device: torch.device,
    decision_weights: Optional[List[float]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Возвращает y_true, raw_preds, tuned_preds и усреднённые вероятности ансамбля."""
    all_targets, all_probs = [], []

    with torch.no_grad():
        for batch in loader:
            validate_batch(batch)
            sig = batch["signal"].to(device)
            rr = batch["rr"].to(device)

            batch_probs = torch.zeros((sig.size(0), NUM_CLASSES), dtype=torch.float32, device=device)
            for m in models:
                batch_probs += F.softmax(m(sig, rr), dim=1)
            batch_probs /= float(len(models))

            all_targets.append(batch["label"].numpy())
            all_probs.append(batch_probs.cpu().numpy())

    y_true = np.concatenate(all_targets)
    y_probs = np.concatenate(all_probs, axis=0)
    raw_preds = np.argmax(y_probs, axis=1)

    if decision_weights is not None:
        w = np.asarray(decision_weights, dtype=np.float64).reshape(1, -1)
        tuned_preds = np.argmax(y_probs * w, axis=1)
    else:
        tuned_preds = raw_preds.copy()

    return y_true, raw_preds, tuned_preds, y_probs


def _save_figure_atomic(fig: plt.Figure, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(target_path.parent), prefix=f".{target_path.stem}_", suffix=".png")
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        fig.savefig(tmp_path, format="png", bbox_inches="tight")
        os.replace(tmp_path, target_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def _save_json_atomic(data: Dict, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(target_path.parent), prefix=f".{target_path.stem}_", suffix=".json")
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, target_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, save_path: Path) -> None:
    class_names = [IDX_TO_LABEL[i] for i in range(NUM_CLASSES)]
    cm_abs = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    row_sums = cm_abs.sum(axis=1, keepdims=True)
    cm_norm = np.zeros_like(cm_abs, dtype=np.float64)
    np.divide(cm_abs.astype(np.float64), row_sums, out=cm_norm, where=row_sums != 0)
    cm_norm *= 100.0

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), dpi=150)
    configs = [
        (axes[0], cm_abs, "Матрица ошибок (количество ударов)", "d"),
        (axes[1], cm_norm, "Матрица ошибок (Recall по строкам, %)", ".1f"),
    ]

    for ax, matrix, title, fmt in configs:
        im = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set(
            xticks=np.arange(NUM_CLASSES),
            yticks=np.arange(NUM_CLASSES),
            xticklabels=class_names,
            yticklabels=class_names,
            ylabel="Истинный класс (Кардиолог)",
            xlabel="Предсказание модели",
            title=title,
        )
        thresh = matrix.max() / 2.0
        for i in range(NUM_CLASSES):
            for j in range(NUM_CLASSES):
                val_str = f"{matrix[i, j]:{fmt}}" + ("%" if fmt == ".1f" else "")
                ax.text(
                    j, i, val_str, ha="center", va="center",
                    color="white" if matrix[i, j] > thresh else "black",
                    fontsize=10, fontweight="bold" if i == j else "normal",
                )

    plt.tight_layout()
    _save_figure_atomic(fig, save_path)
    plt.close(fig)


def evaluate(
    config_path: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    smoke_test: bool = False,
    run_id: Optional[str] = None,
) -> Dict:
    cfg_file = resolve_path(config_path) if config_path else (ROOT_DIR / "configs/config.yaml")
    with open(cfg_file, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", cfg.get("train", {}))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = resolve_path(train_cfg.get("checkpoint_dir", "checkpoints"))
    base_ckpt = "smoke_best_model.pt" if smoke_test else "best_model.pt"
    default_ckpt = ckpt_dir / (f"{run_id}_{base_ckpt}" if run_id else base_ckpt)
    ckpt_file = resolve_path(checkpoint_path) if checkpoint_path else default_ckpt

    models, checkpoint = load_ensemble_from_checkpoint(ckpt_file, model_cfg, device)
    decision_weights = checkpoint.get("decision_weights") or checkpoint.get("class_multipliers") or [1.0] * NUM_CLASSES
    print(f"[OK] Загружен ансамбль из {len(models)} фолдов: '{ckpt_file.relative_to(ROOT_DIR).as_posix()}'")
    print(f"[INFO] Веса решающего правила [N, S, V, F]: {decision_weights}")

    signal_length = int(model_cfg.get("signal_length", 256))
    rr_features_dim = int(model_cfg.get("rr_features_dim", 6))

    if smoke_test:
        signals, rr_features, labels, patient_ids = generate_synthetic_ds1(num_patients=6, samples_per_patient=80, seed=99)
    else:
        test_npz_path = resolve_dataset_path(data_cfg, "test_file", "ds2_test.npz")
        signals, rr_features, labels, patient_ids = load_npz_safely(
            test_npz_path, expected_signal_len=signal_length, expected_rr_dim=rr_features_dim
        )

    test_loader = DataLoader(InMemoryECGDataset(signals, rr_features, labels), batch_size=256, shuffle=False)
    y_true, raw_pred, tuned_pred, y_probs = predict_ensemble(models, test_loader, device, decision_weights=decision_weights)

    all_labels = list(range(NUM_CLASSES))
    class_names = [IDX_TO_LABEL[i] for i in all_labels]

    raw_macro_f1 = float(f1_score(y_true, raw_pred, labels=all_labels, average="macro", zero_division=0))
    tuned_macro_f1 = float(f1_score(y_true, tuned_pred, labels=all_labels, average="macro", zero_division=0))
    nsv_macro_f1 = float(f1_score(y_true, tuned_pred, labels=[0, 1, 2], average="macro", zero_division=0))

    report_dict = classification_report(
        y_true, tuned_pred, labels=all_labels, target_names=class_names, output_dict=True, zero_division=0
    )

    y_true_bin = label_binarize(y_true, classes=all_labels)
    pr_auc_per_class = {
        cls_name: float(average_precision_score(y_true_bin[:, idx], y_probs[:, idx]))
        if np.sum(y_true_bin[:, idx]) > 0 else 0.0
        for idx, cls_name in enumerate(class_names)
    }

    # Разбор качества по отдельным пациентам DS2
    per_patient: Dict[str, Dict] = {}
    for pid in sorted(np.unique(patient_ids).tolist()):
        mask = patient_ids == pid
        p_true = y_true[mask]
        p_pred = tuned_pred[mask]
        present_cls = sorted(np.unique(p_true).tolist())
        p_f1 = float(f1_score(p_true, p_pred, labels=present_cls, average="macro", zero_division=0))
        per_patient[str(pid)] = {
            "num_beats": int(np.sum(mask)),
            "present_classes": [IDX_TO_LABEL[c] for c in present_cls],
            "patient_macro_f1": round(p_f1, 4),
        }

    print(f"\n=== Итоговые метрики на {'Smoke-DS2' if smoke_test else 'Hold-out DS2'} ({len(per_patient)} пациентов, {len(y_true):,} ударов) ===")
    print(f"Raw Macro-F1 (без порогов):        {raw_macro_f1:.4f}")
    print(f"Post-hoc Tuned Macro-F1 (N/S/V/F): {tuned_macro_f1:.4f}")
    print(f"Post-hoc Macro-F1 (N/S/V):         {nsv_macro_f1:.4f}\n")
    print(f"{'Класс':<8} | {'Precision':>9} | {'Recall':>9} | {'F1-score':>9} | {'PR-AUC':>9} | {'Support':>9}")
    print("-" * 68)
    for cls_name in class_names:
        m = report_dict[cls_name]
        print(
            f"{cls_name:<8} | {m['precision']:>9.4f} | {m['recall']:>9.4f} | "
            f"{m['f1-score']:>9.4f} | {pr_auc_per_class[cls_name]:>9.4f} | {int(m['support']):>9,}"
        )

    prefix = f"{run_id}_" if run_id else ""
    fig_path = ROOT_DIR / "reports/figures" / f"{prefix}{'smoke_confusion_matrix.png' if smoke_test else 'confusion_matrix.png'}"
    json_path = ROOT_DIR / "reports" / f"{prefix}{'smoke_ds2_metrics.json' if smoke_test else 'ds2_metrics.json'}"

    summary = {
        "run_id": run_id or "default",
        "checkpoint": ckpt_file.relative_to(ROOT_DIR).as_posix(),
        "split": "smoke_ds2" if smoke_test else "ds2_test",
        "num_patients": int(len(per_patient)),
        "num_beats": int(len(y_true)),
        "raw_macro_f1": round(raw_macro_f1, 4),
        "macro_f1": round(tuned_macro_f1, 4),
        "macro_f1_nsv": round(nsv_macro_f1, 4),
        "decision_weights": decision_weights,
        "confusion_matrix_figure": fig_path.relative_to(ROOT_DIR).as_posix(),
        "per_class": {
            cls_name: {
                "precision": round(float(report_dict[cls_name]["precision"]), 4),
                "recall": round(float(report_dict[cls_name]["recall"]), 4),
                "f1": round(float(report_dict[cls_name]["f1-score"]), 4),
                "pr_auc": round(pr_auc_per_class[cls_name], 4),
                "support": int(report_dict[cls_name]["support"]),
            }
            for cls_name in class_names
        },
        "per_patient": per_patient,
    }

    plot_confusion_matrix(y_true, tuned_pred, fig_path)
    _save_json_atomic(summary, json_path)
    print(f"\n[OK] Матрица ошибок сохранена в: '{fig_path.relative_to(ROOT_DIR).as_posix()}'")
    print(f"[OK] JSON-отчёт сохранён в: '{json_path.relative_to(ROOT_DIR).as_posix()}'")
    return summary


if __name__ == "__main__":
    plt.switch_backend("Agg")
    parser = argparse.ArgumentParser(description="Оценка ансамбля на тестовой выборке DS2")
    parser.add_argument("--config", type=str, default=None, help="Путь к конфигурации")
    parser.add_argument("--checkpoint", type=str, default=None, help="Путь к чекпоинту .pt")
    parser.add_argument("--run-id", type=str, default=None, help="Идентификатор запуска")
    parser.add_argument("--smoke-test", action="store_true", help="Тестовый запуск")
    args = parser.parse_args()
    evaluate(args.config, checkpoint_path=args.checkpoint, smoke_test=args.smoke_test, run_id=args.run_id)