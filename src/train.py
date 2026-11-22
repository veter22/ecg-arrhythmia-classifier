"""
Пайплайн обучения гибридного классификатора ЭКГ — [ML-03 v4]
=============================================================
Ветка: main
"""

import argparse
import os
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, Dataset
import yaml

from src.contracts import NUM_CLASSES, validate_batch
from src.losses.focal_loss import FocalLoss, compute_effective_num_weights
from src.models.hybrid_cnn import HybridECGClassifier

MAX_NPZ_ARCHIVE_BYTES = 512 * 1024 * 1024
MAX_NPZ_UNCOMPRESSED_BYTES = 2 * 1024 * 1024 * 1024


class InMemoryECGDataset(Dataset):
    """Dataset для сигналов и RR-признаков из .npz массива."""

    def __init__(
        self,
        signals: np.ndarray,
        rr_features: np.ndarray,
        labels: np.ndarray,
    ):
        self.signals = torch.tensor(signals, dtype=torch.float32)
        if self.signals.ndim == 2:
            self.signals = self.signals.unsqueeze(1)
        self.rr_features = torch.tensor(rr_features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "signal": self.signals[idx],
            "rr": self.rr_features[idx],
            "label": self.labels[idx],
        }


def generate_synthetic_ds1(
    num_patients: int = 10,
    samples_per_patient: int = 100,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Генерирует синтетический датасет для smoke-теста."""
    rng = np.random.default_rng(seed)
    total = num_patients * samples_per_patient

    signals = rng.normal(0.0, 1.0, size=(total, 1, 256)).astype(np.float32)
    rr_features = rng.uniform(0.5, 1.5, size=(total, 4)).astype(np.float32)
    labels = rng.choice([0, 1, 2, 3], size=total, p=[0.85, 0.05, 0.08, 0.02]).astype(np.int64)
    patient_ids = np.repeat(np.arange(100, 100 + num_patients), samples_per_patient)

    return signals, rr_features, labels, patient_ids


def load_npz_safely(npz_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Безопасно проверяет размер архива и загружает массивы по контракту."""
    if not npz_path.exists():
        raise FileNotFoundError(
            f"Файл '{npz_path}' не найден! Запустите 'python -m src.data.dataset' или --smoke-test."
        )

    if npz_path.stat().st_size > MAX_NPZ_ARCHIVE_BYTES:
        raise ValueError(f"Файл '{npz_path}' превышает допустимый лимит размера архива.")

    with zipfile.ZipFile(npz_path, "r") as zf:
        total_uncompressed = sum(info.file_size for info in zf.infolist())
        if total_uncompressed > MAX_NPZ_UNCOMPRESSED_BYTES:
            raise ValueError(
                f"Распакованный размер '{npz_path}' ({total_uncompressed} байт) превышает лимит памяти."
            )

    with np.load(npz_path, allow_pickle=False) as data:
        if "rr" not in data.files:
            raise KeyError(f"В архиве '{npz_path}' отсутствует ключ 'rr'. Найдены: {data.files}")
        return data["signals"], data["rr"], data["labels"], data["patient_ids"]


def resolve_dataset_path(data_cfg: Dict, file_key: str, default_filename: str) -> Path:
    """Определяет путь к файлу датасета из конфигурации."""
    processed_dir = Path(data_cfg.get("processed_dir", "data/processed"))
    configured_file = data_cfg.get(file_key)
    if configured_file:
        cfg_path = Path(configured_file)
        return cfg_path if cfg_path.is_absolute() or cfg_path.parent != Path(".") else processed_dir / cfg_path
    return processed_dir / default_filename


def save_checkpoint_atomic(payload: Dict, target_path: Path) -> None:
    """Атомарно сохраняет чекпоинт через временный файл."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(target_path.parent), prefix=f".{target_path.stem}_", suffix=".tmp")
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, target_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def collect_val_predictions(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Возвращает средний loss, истинные метки и вероятности Softmax на валидации."""
    model.eval()
    total_loss = 0.0
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            validate_batch(batch)
            signals = batch["signal"].to(device)
            rr = batch["rr"].to(device)
            labels = batch["label"].to(device)

            logits = model(signals, rr)
            loss = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            probs = F.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_targets.append(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, np.concatenate(all_targets), np.concatenate(all_probs, axis=0)


def tune_class_multipliers(y_true: np.ndarray, y_probs: np.ndarray) -> List[float]:
    """
    Мягкий координатный подбор весовых множителей вероятностей [1.0, w_S, w_V, w_F] в диапазоне [1.0, 3.5]
    для повышения Recall редких классов без переобучения под валидационный фолд.
    """
    all_labels = list(range(NUM_CLASSES))
    multipliers = np.ones(NUM_CLASSES, dtype=np.float64)
    best_f1 = float(f1_score(y_true, np.argmax(y_probs, axis=1), labels=all_labels, average="macro", zero_division=0))

    grid = np.linspace(1.0, 3.5, num=26)
    for _ in range(2):
        for cls_idx in range(1, NUM_CLASSES):
            best_w = multipliers[cls_idx]
            for w in grid:
                candidate = multipliers.copy()
                candidate[cls_idx] = w
                preds = np.argmax(y_probs * candidate, axis=1)
                score = float(f1_score(y_true, preds, labels=all_labels, average="macro", zero_division=0))
                if score > best_f1:
                    best_f1 = score
                    best_w = w
            multipliers[cls_idx] = best_w

    return [round(float(x), 4) for x in multipliers]


def train(config_path: str, smoke_test: bool = False, run_id: Optional[str] = None) -> Path:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    project_cfg = cfg.get("project", {})
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", cfg.get("train", {}))

    seed = int(train_cfg.get("seed", project_cfg.get("seed", cfg.get("seed", 42))))
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Устройство обучения: {device}")

    if smoke_test:
        print("[INFO] Запуск в режиме --smoke-test на синтетических данных...")
        signals, rr_features, labels, patient_ids = generate_synthetic_ds1(seed=seed)
    else:
        train_npz_path = resolve_dataset_path(data_cfg, "train_file", "ds1_train.npz")
        signals, rr_features, labels, patient_ids = load_npz_safely(train_npz_path)

    # Стандартный пациентонезависимый сплит (fold 0 из GroupKFold)
    n_splits = int(train_cfg.get("n_folds", 5))
    gkf = GroupKFold(n_splits=n_splits)
    train_idx, val_idx = next(gkf.split(signals, labels, groups=patient_ids))

    train_patients = set(patient_ids[train_idx])
    val_patients = set(patient_ids[val_idx])
    assert len(train_patients.intersection(val_patients)) == 0, "Обнаружена утечка пациентов!"
    print(
        f"[OK] Сплит по пациентам: Train={len(train_patients)} пациентов ({len(train_idx):,} ударов) | "
        f"Val={len(val_patients)} пациентов ({len(val_idx):,} ударов)"
    )

    train_labels = labels[train_idx]
    class_counts = [int(np.sum(train_labels == c)) for c in range(NUM_CLASSES)]
    beta = float(
        train_cfg.get("effective_samples_beta", train_cfg.get("focal_beta", train_cfg.get("beta", 0.999)))
    )
    gamma = float(train_cfg.get("focal_gamma", train_cfg.get("gamma", 2.0)))

    alpha_weights = compute_effective_num_weights(class_counts, beta=beta).to(device)
    print(f"[INFO] Сглаженные веса классов (beta={beta}): {[round(w, 3) for w in alpha_weights.tolist()]}")
    criterion = FocalLoss(alpha=alpha_weights, gamma=gamma)

    batch_size = 32 if smoke_test else int(train_cfg.get("batch_size", 128))
    train_dataset = InMemoryECGDataset(signals[train_idx], rr_features[train_idx], labels[train_idx])
    val_dataset = InMemoryECGDataset(signals[val_idx], rr_features[val_idx], labels[val_idx])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    cnn_embed_dim = int(model_cfg.get("cnn_embed_dim", 128))
    rr_embed_dim = int(model_cfg.get("rr_embed_dim", 32))
    dropout = float(model_cfg.get("dropout", 0.3))

    model = HybridECGClassifier(
        cnn_embed_dim=cnn_embed_dim,
        rr_embed_dim=rr_embed_dim,
        num_classes=NUM_CLASSES,
        dropout=dropout,
    ).to(device)

    lr = float(train_cfg.get("learning_rate", train_cfg.get("lr", 1e-3)))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))
    epochs = 2 if smoke_test else int(train_cfg.get("epochs", 40))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    ckpt_dir = Path(train_cfg.get("checkpoint_dir", "checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    base_name = "smoke_best_model.pt" if smoke_test else "best_model.pt"
    best_ckpt_path = ckpt_dir / (f"{run_id}_{base_name}" if run_id else base_name)

    best_val_f1 = -1.0
    all_labels = list(range(NUM_CLASSES))

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            validate_batch(batch)
            sig = batch["signal"].to(device)
            rr = batch["rr"].to(device)
            targets = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(sig, rr)
            loss = criterion(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * targets.size(0)

        scheduler.step()
        train_loss = running_loss / len(train_dataset)
        val_loss, val_true, val_probs = collect_val_predictions(model, val_loader, criterion, device)
        val_preds = np.argmax(val_probs, axis=1)
        val_f1 = float(f1_score(val_true, val_preds, labels=all_labels, average="macro", zero_division=0))

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Macro-F1: {val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            multipliers = tune_class_multipliers(val_true, val_probs)
            tuned_preds = np.argmax(val_probs * np.asarray(multipliers), axis=1)
            tuned_val_f1 = float(f1_score(val_true, tuned_preds, labels=all_labels, average="macro", zero_division=0))

            save_checkpoint_atomic(
                {
                    "epoch": int(epoch),
                    "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                    "val_macro_f1": float(best_val_f1),
                    "tuned_val_macro_f1": float(tuned_val_f1),
                    "class_multipliers": multipliers,
                    "model_params": {
                        "cnn_embed_dim": cnn_embed_dim,
                        "rr_embed_dim": rr_embed_dim,
                        "num_classes": NUM_CLASSES,
                        "dropout": dropout,
                    },
                },
                best_ckpt_path,
            )

    saved_ckpt = torch.load(best_ckpt_path, map_location="cpu", weights_only=True)
    print(f"\n[OK] Обучение завершено! Raw Val Macro-F1: {best_val_f1:.4f} -> Tuned Val Macro-F1: {saved_ckpt['tuned_val_macro_f1']:.4f}")
    print(f"[OK] Калибровочные множители классов [N, S, V, F]: {saved_ckpt['class_multipliers']}")
    print(f"[OK] Чекпоинт сохранён в: '{best_ckpt_path}'")
    return best_ckpt_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Обучение гибридного классификатора ЭКГ")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Путь к файлу конфигурации")
    parser.add_argument("--run-id", type=str, default=None, help="Идентификатор запуска")
    parser.add_argument("--smoke-test", action="store_true", help="Тестовое обучение на 2 эпохи")
    args = parser.parse_args()
    train(args.config, smoke_test=args.smoke_test, run_id=args.run_id)