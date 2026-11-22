"""
Независимая оценка модели на отложенной выборке DS2 — [ML-04 v2]
=================================================================
Ветка: main
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
)


def load_model_from_checkpoint(
    ckpt_file: Path,
    model_cfg: Dict,
    device: torch.device,
) -> Tuple[HybridECGClassifier, Dict]:
    """Безопасно загружает веса модели (weights_only=True) с учётом размерностей из чекпоинта."""
    if not ckpt_file.exists():
        raise FileNotFoundError(f"Чекпоинт '{ckpt_file}' не найден!")

    checkpoint = torch.load(ckpt_file, map_location=device, weights_only=True)
    saved_params = checkpoint.get("model_params", {})

    model = HybridECGClassifier(
        cnn_embed_dim=int(saved_params.get("cnn_embed_dim", model_cfg.get("cnn_embed_dim", 128))),
        rr_embed_dim=int(saved_params.get("rr_embed_dim", model_cfg.get("rr_embed_dim", 32))),
        num_classes=int(saved_params.get("num_classes", NUM_CLASSES)),
        dropout=float(saved_params.get("dropout", model_cfg.get("dropout", 0.3))),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, checkpoint


def predict_on_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_multipliers: Optional[List[float]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Возвращает истинные метки, предсказанные классы (с учётом калибровки) и вероятности Softmax."""
    model.eval()
    all_targets = []
    all_preds = []
    all_probs = []

    mult_tensor = None
    if class_multipliers is not None:
        mult_tensor = torch.tensor(class_multipliers, dtype=torch.float32, device=device).view(1, -1)

    with torch.no_grad():
        for batch in loader:
            validate_batch(batch)
            signals = batch["signal"].to(device)
            rr = batch["rr"].to(device)
            labels = batch["label"]

            logits = model(signals, rr)
            probs = F.softmax(logits, dim=1)
            scaled_probs = probs * mult_tensor if mult_tensor is not None else probs
            preds = torch.argmax(scaled_probs, dim=1)

            all_targets.append(labels.numpy())
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    return (
        np.concatenate(all_targets),
        np.concatenate(all_preds),
        np.concatenate(all_probs, axis=0),
    )


def _save_figure_atomic(fig: plt.Figure, target_path: Path) -> None:
    """Атомарно сохраняет изображение графика на диск."""
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
    """Атомарно сохраняет JSON-отчёт на диск."""
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


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Path,
) -> None:
    """Строит и атомарно сохраняет двойную Confusion Matrix (абсолютные числа и Recall по строкам)."""
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
                    j,
                    i,
                    val_str,
                    ha="center",
                    va="center",
                    color="white" if matrix[i, j] > thresh else "black",
                    fontsize=10,
                    fontweight="bold" if i == j else "normal",
                )

    plt.tight_layout()
    _save_figure_atomic(fig, save_path)
    plt.close(fig)


def evaluate(
    config_path: str,
    checkpoint_path: Optional[str] = None,
    smoke_test: bool = False,
    run_id: Optional[str] = None,
) -> Dict:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", cfg.get("train", {}))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Устройство инференса: {device}")

    # 1. Загрузка чекпоинта и калибровочных множителей
    ckpt_dir = Path(train_cfg.get("checkpoint_dir", "checkpoints"))
    base_ckpt = "smoke_best_model.pt" if smoke_test else "best_model.pt"
    default_ckpt = ckpt_dir / (f"{run_id}_{base_ckpt}" if run_id else base_ckpt)
    ckpt_file = Path(checkpoint_path) if checkpoint_path else default_ckpt

    model, checkpoint = load_model_from_checkpoint(ckpt_file, model_cfg, device)
    multipliers = checkpoint.get("class_multipliers", [1.0] * NUM_CLASSES)
    print(f"[OK] Загружен чекпоинт '{ckpt_file}' (эпоха {checkpoint.get('epoch', '?')})")
    print(f"[INFO] Калибровочные множители классов [N, S, V, F]: {multipliers}")

    # 2. Загрузка или генерация DS2
    if smoke_test:
        print("[INFO] Режим --smoke-test: генерация синтетического набора DS2...")
        signals, rr_features, labels, patient_ids = generate_synthetic_ds1(num_patients=6, samples_per_patient=80, seed=99)
    else:
        test_npz_path = resolve_dataset_path(data_cfg, "test_file", "ds2_test.npz")
        signals, rr_features, labels, patient_ids = load_npz_safely(test_npz_path)

    test_dataset = InMemoryECGDataset(signals, rr_features, labels)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    # 3. Получение предсказаний с учётом калибровки порогов
    y_true, y_pred, y_probs = predict_on_loader(
        model, test_loader, device, class_multipliers=multipliers
    )

    # 4. Расчёт метрик по всем 4 классам
    all_labels = list(range(NUM_CLASSES))
    class_names = [IDX_TO_LABEL[i] for i in all_labels]
    macro_f1 = float(f1_score(y_true, y_pred, labels=all_labels, average="macro", zero_division=0))

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=all_labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    y_true_bin = label_binarize(y_true, classes=all_labels)
    pr_auc_per_class = {}
    for idx, cls_name in enumerate(class_names):
        if np.sum(y_true_bin[:, idx]) > 0:
            pr_auc_per_class[cls_name] = float(average_precision_score(y_true_bin[:, idx], y_probs[:, idx]))
        else:
            pr_auc_per_class[cls_name] = 0.0

    # Вывод сводной таблицы в терминал
    print(f"\n=== Итоговые метрики на {'Smoke-DS2' if smoke_test else 'Hold-out DS2'} ({len(np.unique(patient_ids))} пациентов, {len(y_true):,} ударов) ===")
    print(f"Главная метрика Macro-F1: {macro_f1:.4f}\n")
    print(f"{'Класс':<8} | {'Precision':>9} | {'Recall':>9} | {'F1-score':>9} | {'PR-AUC':>9} | {'Support':>9}")
    print("-" * 68)
    for cls_name in class_names:
        m = report_dict[cls_name]
        print(
            f"{cls_name:<8} | {m['precision']:>9.4f} | {m['recall']:>9.4f} | "
            f"{m['f1-score']:>9.4f} | {pr_auc_per_class[cls_name]:>9.4f} | {int(m['support']):>9,}"
        )

    # 5. Согласованная атомарная публикация отчётов
    prefix = f"{run_id}_" if run_id else ""
    fig_name = f"{prefix}{'smoke_confusion_matrix.png' if smoke_test else 'confusion_matrix.png'}"
    json_name = f"{prefix}{'smoke_ds2_metrics.json' if smoke_test else 'ds2_metrics.json'}"

    fig_path = Path("reports/figures") / fig_name
    json_path = Path("reports") / json_name

    summary = {
        "run_id": run_id or "default",
        "checkpoint": str(ckpt_file),
        "split": "smoke_ds2" if smoke_test else "ds2_test",
        "num_patients": int(len(np.unique(patient_ids))),
        "num_beats": int(len(y_true)),
        "macro_f1": round(macro_f1, 4),
        "class_multipliers": multipliers,
        "confusion_matrix_figure": str(fig_path),
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
    }

    plot_confusion_matrix(y_true, y_pred, fig_path)
    _save_json_atomic(summary, json_path)

    print(f"\n[OK] Матрица ошибок сохранена в: '{fig_path}'")
    print(f"[OK] JSON-отчёт сохранён в: '{json_path}'")
    return summary


if __name__ == "__main__":
    plt.switch_backend("Agg")

    parser = argparse.ArgumentParser(description="Оценка модели на тестовой выборке DS2")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Путь к конфигурации")
    parser.add_argument("--checkpoint", type=str, default=None, help="Путь к файлу весов .pt")
    parser.add_argument("--run-id", type=str, default=None, help="Идентификатор запуска для изоляции артефактов")
    parser.add_argument("--smoke-test", action="store_true", help="Тестовый запуск на синтетических данных")
    args = parser.parse_args()

    evaluate(args.config, checkpoint_path=args.checkpoint, smoke_test=args.smoke_test, run_id=args.run_id)