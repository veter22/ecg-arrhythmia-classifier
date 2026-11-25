"""
Пайплайн обучения ансамбля 5 фолдов StratifiedGroupKFold и OOF-подбора весов — [ML-03 v5]
"""

import argparse
import os
import random
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
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset
import yaml

from src.contracts import NPZ_REQUIRED_KEYS, NUM_CLASSES, validate_batch, validate_npz_arrays
from src.losses.focal_loss import FocalLoss, compute_effective_num_weights
from src.models.hybrid_cnn import HybridECGClassifier

MAX_NPZ_ARCHIVE_BYTES = 512 * 1024 * 1024
MAX_NPZ_UNCOMPRESSED_BYTES = 2 * 1024 * 1024 * 1024


def seed_everything(seed: int) -> None:
    """Полная фиксация генераторов случайных чисел."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class InMemoryECGDataset(Dataset):
    """Dataset для ЭКГ-сигналов и RR-признаков с мягкой аугментацией сигнала на обучении."""

    def __init__(
        self,
        signals: np.ndarray,
        rr_features: np.ndarray,
        labels: np.ndarray,
        augment: bool = False,
    ):
        self.signals = torch.tensor(signals, dtype=torch.float32)
        if self.signals.ndim == 2:
            self.signals = self.signals.unsqueeze(1)
        self.rr_features = torch.tensor(rr_features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sig = self.signals[idx]
        rr = self.rr_features[idx]

        if self.augment:
            # 1. Масштабирование амплитуды ±10%
            scale = 1.0 + torch.empty(1).uniform_(-0.10, 0.10).item()
            # 2. Лёгкий сдвиг изолинии
            offset = torch.empty(1).uniform_(-0.08, 0.08).item()
            # 3. Высокочастотный шум измерительного тракта
            noise = torch.randn_like(sig) * 0.025
            sig = sig * scale + offset + noise

        return {"signal": sig, "rr": rr, "label": self.labels[idx]}


def generate_synthetic_ds1(
    num_patients: int = 10,
    samples_per_patient: int = 100,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    total = num_patients * samples_per_patient
    signals = rng.normal(0.0, 1.0, size=(total, 1, 256)).astype(np.float32)
    rr_features = rng.uniform(0.5, 1.5, size=(total, 6)).astype(np.float32)
    labels = rng.choice([0, 1, 2, 3], size=total, p=[0.85, 0.05, 0.08, 0.02]).astype(np.int64)
    patient_ids = np.repeat(np.arange(100, 100 + num_patients), samples_per_patient)
    return signals, rr_features, labels, patient_ids


def load_npz_safely(
    npz_path: Path,
    expected_signal_len: int = 256,
    expected_rr_dim: int = 6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not npz_path.exists():
        raise FileNotFoundError(f"Файл '{npz_path}' не найден! Запустите 'python -m src.data.dataset'.")
    if npz_path.stat().st_size > MAX_NPZ_ARCHIVE_BYTES:
        raise ValueError(f"Файл '{npz_path}' превышает лимит размера архива.")

    with zipfile.ZipFile(npz_path, "r") as zf:
        if sum(info.file_size for info in zf.infolist()) > MAX_NPZ_UNCOMPRESSED_BYTES:
            raise ValueError(f"Распакованный размер '{npz_path}' превышает лимит памяти.")

    with np.load(npz_path, allow_pickle=False) as data:
        missing = [k for k in NPZ_REQUIRED_KEYS if k not in data.files]
        if missing:
            raise KeyError(f"В архиве '{npz_path}' отсутствуют ключи: {missing}")
        signals = data["signals"]
        rr = data["rr"]
        labels = data["labels"]
        patient_ids = data["patient_ids"]

    validate_npz_arrays(
        signals, rr, labels, patient_ids,
        expected_signal_len=expected_signal_len,
        expected_rr_dim=expected_rr_dim,
    )
    return signals, rr, labels, patient_ids


def resolve_path(rel_or_abs: str) -> Path:
    p = Path(rel_or_abs)
    return p if p.is_absolute() else ROOT_DIR / p


def resolve_dataset_path(data_cfg: Dict, file_key: str, default_filename: str) -> Path:
    processed_dir = resolve_path(data_cfg.get("processed_dir", "data/processed"))
    configured_file = data_cfg.get(file_key)
    if configured_file:
        cfg_path = Path(configured_file)
        return cfg_path if cfg_path.is_absolute() or cfg_path.parent != Path(".") else processed_dir / cfg_path
    return processed_dir / default_filename


def save_checkpoint_atomic(payload: Dict, target_path: Path) -> None:
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


def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    all_probs, all_targets = [], []

    with torch.no_grad():
        for batch in loader:
            validate_batch(batch)
            sig = batch["signal"].to(device)
            rr = batch["rr"].to(device)
            targets = batch["label"].to(device)

            logits = model(sig, rr)
            loss = criterion(logits, targets)
            total_loss += loss.item() * targets.size(0)

            all_probs.append(F.softmax(logits, dim=1).cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    return total_loss / len(loader.dataset), np.concatenate(all_targets), np.concatenate(all_probs, axis=0)


def tune_decision_weights(y_true: np.ndarray, oof_probs: np.ndarray) -> List[float]:
    """
    Post-hoc подбор весов решающего правила [w_N, w_S, w_V, w_F] на out-of-fold вероятностях
    всей выборки DS1 (22 пациента) для оптимизации Macro-F1.
    """
    all_labels = list(range(NUM_CLASSES))
    weights = np.ones(NUM_CLASSES, dtype=np.float64)
    best_f1 = float(f1_score(y_true, np.argmax(oof_probs, axis=1), labels=all_labels, average="macro", zero_division=0))

    grid = np.geomspace(0.4, 5.0, num=47)
    for _ in range(3):
        for cls_idx in range(NUM_CLASSES):
            best_w = weights[cls_idx]
            for w in grid:
                candidate = weights.copy()
                candidate[cls_idx] = w
                preds = np.argmax(oof_probs * candidate, axis=1)
                score = float(f1_score(y_true, preds, labels=all_labels, average="macro", zero_division=0))
                if score > best_f1:
                    best_f1 = score
                    best_w = w
            weights[cls_idx] = best_w

    # Нормируем относительно w_N для удобства интерпретации
    weights = weights / weights[0]
    return [round(float(x), 4) for x in weights]


def train(config_path: Optional[str] = None, smoke_test: bool = False, run_id: Optional[str] = None) -> Path:
    cfg_file = resolve_path(config_path) if config_path else (ROOT_DIR / "configs/config.yaml")
    with open(cfg_file, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    project_cfg = cfg.get("project", {})
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", cfg.get("train", {}))

    seed = int(train_cfg.get("seed", project_cfg.get("seed", 42)))
    seed_everything(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Устройство обучения: {device}")

    in_channels = int(model_cfg.get("in_channels", 1))
    signal_length = int(model_cfg.get("signal_length", 256))
    rr_features_dim = int(model_cfg.get("rr_features_dim", 6))
    cnn_embed_dim = int(model_cfg.get("cnn_embed_dim", 128))
    rr_embed_dim = int(model_cfg.get("rr_embed_dim", 32))
    dropout = float(model_cfg.get("dropout", 0.3))

    if smoke_test:
        print("[INFO] Запуск в режиме --smoke-test на синтетических данных...")
        signals, rr_features, labels, patient_ids = generate_synthetic_ds1(seed=seed)
    else:
        train_npz_path = resolve_dataset_path(data_cfg, "train_file", "ds1_train.npz")
        signals, rr_features, labels, patient_ids = load_npz_safely(
            train_npz_path, expected_signal_len=signal_length, expected_rr_dim=rr_features_dim
        )

    n_splits = 2 if smoke_test else int(train_cfg.get("n_folds", 5))
    try:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = list(splitter.split(signals, labels, groups=patient_ids))
    except ValueError:
        splitter = GroupKFold(n_splits=n_splits)
        splits = list(splitter.split(signals, labels, groups=patient_ids))

    beta = float(train_cfg.get("effective_samples_beta", 0.9999))
    gamma = float(train_cfg.get("focal_gamma", 2.0))
    batch_size = 32 if smoke_test else int(train_cfg.get("batch_size", 128))
    lr = float(train_cfg.get("learning_rate", 1e-3))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))
    epochs = 2 if smoke_test else int(train_cfg.get("epochs", 35))

    oof_probs = np.zeros((len(labels), NUM_CLASSES), dtype=np.float32)
    fold_states: List[Dict[str, torch.Tensor]] = []
    fold_val_scores: List[float] = []
    all_labels = list(range(NUM_CLASSES))

    for fold_idx, (train_idx, val_idx) in enumerate(splits, start=1):
        tr_pats = set(patient_ids[train_idx])
        va_pats = set(patient_ids[val_idx])
        assert len(tr_pats.intersection(va_pats)) == 0, f"Утечка пациентов на фолде {fold_idx}!"
        print(f"\n--- Fold {fold_idx}/{n_splits} | Train: {len(tr_pats)} пат. ({len(train_idx):,}) | Val: {len(va_pats)} пат. ({len(val_idx):,}) ---")

        class_counts = [int(np.sum(labels[train_idx] == c)) for c in range(NUM_CLASSES)]
        alpha_weights = compute_effective_num_weights(class_counts, beta=beta).to(device)
        criterion = FocalLoss(alpha=alpha_weights, gamma=gamma)

        train_loader = DataLoader(
            InMemoryECGDataset(signals[train_idx], rr_features[train_idx], labels[train_idx], augment=True),
            batch_size=batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            InMemoryECGDataset(signals[val_idx], rr_features[val_idx], labels[val_idx], augment=False),
            batch_size=batch_size * 2,
            shuffle=False,
        )

        model = HybridECGClassifier(
            in_channels=in_channels,
            signal_length=signal_length,
            rr_features_dim=rr_features_dim,
            cnn_embed_dim=cnn_embed_dim,
            rr_embed_dim=rr_embed_dim,
            num_classes=NUM_CLASSES,
            dropout=dropout,
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_fold_f1 = -1.0
        best_fold_weights = None
        best_fold_probs = None

        for epoch in range(1, epochs + 1):
            model.train()
            running_loss = 0.0
            for batch in train_loader:
                validate_batch(batch)
                optimizer.zero_grad()
                logits = model(batch["signal"].to(device), batch["rr"].to(device))
                loss = criterion(logits, batch["label"].to(device))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                running_loss += loss.item() * batch["label"].size(0)

            scheduler.step()
            val_loss, val_true, val_probs = collect_predictions(model, val_loader, criterion, device)
            val_f1 = float(f1_score(val_true, np.argmax(val_probs, axis=1), labels=all_labels, average="macro", zero_division=0))

            if epoch % 5 == 0 or epoch == epochs:
                print(
                    f"  [Fold {fold_idx} | Ep {epoch:02d}/{epochs:02d}] "
                    f"Train Loss: {running_loss / len(train_idx):.4f} | Val Loss: {val_loss:.4f} | Val Macro-F1: {val_f1:.4f}"
                )

            if val_f1 > best_fold_f1:
                best_fold_f1 = val_f1
                best_fold_weights = {k: v.cpu() for k, v in model.state_dict().items()}
                best_fold_probs = val_probs

        oof_probs[val_idx] = best_fold_probs
        fold_states.append(best_fold_weights)
        fold_val_scores.append(best_fold_f1)
        print(f"[OK] Fold {fold_idx} завершён | Best Val Macro-F1: {best_fold_f1:.4f}")

    # Честный post-hoc подбор весов решающего правила на OOF-предсказаниях всей выборки DS1
    raw_oof_f1 = float(f1_score(labels, np.argmax(oof_probs, axis=1), labels=all_labels, average="macro", zero_division=0))
    decision_weights = tune_decision_weights(labels, oof_probs)
    tuned_oof_f1 = float(
        f1_score(labels, np.argmax(oof_probs * np.asarray(decision_weights), axis=1), labels=all_labels, average="macro", zero_division=0)
    )

    ckpt_dir = resolve_path(train_cfg.get("checkpoint_dir", "checkpoints"))
    base_name = "smoke_best_model.pt" if smoke_test else "best_model.pt"
    best_ckpt_path = ckpt_dir / (f"{run_id}_{base_name}" if run_id else base_name)

    save_checkpoint_atomic(
        {
            "n_folds": n_splits,
            "fold_state_dicts": fold_states,
            "fold_val_macro_f1": [round(s, 4) for s in fold_val_scores],
            "oof_raw_macro_f1": round(raw_oof_f1, 4),
            "oof_tuned_macro_f1": round(tuned_oof_f1, 4),
            "decision_weights": decision_weights,
            "model_params": {
                "in_channels": in_channels,
                "signal_length": signal_length,
                "rr_features_dim": rr_features_dim,
                "cnn_embed_dim": cnn_embed_dim,
                "rr_embed_dim": rr_embed_dim,
                "num_classes": NUM_CLASSES,
                "dropout": dropout,
            },
        },
        best_ckpt_path,
    )

    print(f"\n[OK] Кросс-валидация завершена! OOF Raw Macro-F1: {raw_oof_f1:.4f} -> Post-hoc Tuned Macro-F1: {tuned_oof_f1:.4f}")
    print(f"[OK] Подобранные веса решающего правила [N, S, V, F]: {decision_weights}")
    print(f"[OK] Чекпоинт ансамбля сохранён в: '{best_ckpt_path}'")
    return best_ckpt_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Обучение ансамбля классификатора ЭКГ")
    parser.add_argument("--config", type=str, default=None, help="Путь к конфигурации")
    parser.add_argument("--run-id", type=str, default=None, help="Идентификатор запуска")
    parser.add_argument("--smoke-test", action="store_true", help="Тестовое обучение")
    args = parser.parse_args()
    train(args.config, smoke_test=args.smoke_test, run_id=args.run_id)