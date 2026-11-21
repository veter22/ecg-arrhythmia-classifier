"""
Пайплайн обучения гибридного классификатора ЭКГ — [ML-03]
=========================================================
Ветка: feature/ml-model

Особенности реализации:
1. Пациентонезависимый сплит (GroupKFold) внутри DS1 — предотвращает утечку данных.
2. Сглаженный Focal Loss с весами эффективного числа сэмплов (Cui et al.).
3. Оптимизатор AdamW + CosineAnnealingLR.
4. Отслеживание Macro-F1 на валидации и сохранение лучшего чекпоинта.
5. Поддержка режима `--smoke-test` для проверки пайплайна на синтетических данных.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, Dataset
import yaml

from src.contracts import NUM_CLASSES, validate_batch
from src.losses.focal_loss import FocalLoss, compute_effective_num_weights
from src.models.hybrid_cnn import HybridECGClassifier


class InMemoryECGDataset(Dataset):
    """Dataset для работы с массивами сигналов, RR-признаков и меток."""

    def __init__(self, signals: np.ndarray, rr_features: np.ndarray, labels: np.ndarray):
        self.signals = torch.tensor(signals, dtype=torch.float32)
        if self.signals.ndim == 2:
            self.signals = self.signals.unsqueeze(1)  # [N, 256] -> [N, 1, 256]
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
    """Генерирует синтетический датасет с реалистичным дисбалансом классов для smoke-теста."""
    rng = np.random.default_rng(seed)
    total = num_patients * samples_per_patient

    signals = rng.normal(0.0, 1.0, size=(total, 1, 256)).astype(np.float32)
    rr_features = rng.uniform(0.5, 1.5, size=(total, 4)).astype(np.float32)
    # Имитация дисбаланса AAMI: ~85% N (0), ~5% S (1), ~8% V (2), ~2% F (3)
    labels = rng.choice([0, 1, 2, 3], size=total, p=[0.85, 0.05, 0.08, 0.02]).astype(np.int64)
    patient_ids = np.repeat(np.arange(100, 100 + num_patients), samples_per_patient)

    return signals, rr_features, labels, patient_ids


def load_ds1_arrays(processed_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Загружает подготовленный напарником файл ds1_train.npz."""
    npz_path = processed_dir / "ds1_train.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"Файл '{npz_path}' не найден! Дождитесь завершения [DATA-03] или запустите с флагом --smoke-test."
        )

    data = np.load(npz_path)
    return data["signals"], data["rr_features"], data["labels"], data["patient_ids"]


def evaluate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Оценка модели на валидационной выборке (loss и Macro-F1)."""
    model.eval()
    total_loss = 0.0
    all_preds = []
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
            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    return avg_loss, macro_f1


def train(config_path: str, smoke_test: bool = False) -> None:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", cfg.get("train", {}))

    seed = int(train_cfg.get("seed", cfg.get("seed", 42)))
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Устройство обучения: {device}")

    # 1. Загрузка или генерация данных
    if smoke_test:
        print("[INFO] Запуск в режиме --smoke-test на синтетических данных...")
        signals, rr_features, labels, patient_ids = generate_synthetic_ds1(seed=seed)
    else:
        processed_dir = Path(data_cfg.get("processed_dir", "data/processed"))
        signals, rr_features, labels, patient_ids = load_ds1_arrays(processed_dir)

    # 2. Пациентонезависимый сплит (GroupKFold, 5 фолдов -> 80% пациентов train / 20% val)
    n_splits = int(train_cfg.get("n_folds", 5))
    gkf = GroupKFold(n_splits=n_splits)
    train_idx, val_idx = next(gkf.split(signals, labels, groups=patient_ids))

    train_patients = set(patient_ids[train_idx])
    val_patients = set(patient_ids[val_idx])
    assert len(train_patients.intersection(val_patients)) == 0, "Обнаружена утечка пациентов между train и val!"
    print(
        f"[OK] Сплит по пациентам: Train={len(train_patients)} пациентов ({len(train_idx):,} ударов) | "
        f"Val={len(val_patients)} пациентов ({len(val_idx):,} ударов)"
    )

    # 3. Расчёт сглаженных весов Focal Loss строго по обучающей части
    train_labels = labels[train_idx]
    class_counts = [int(np.sum(train_labels == c)) for c in range(NUM_CLASSES)]
    beta = float(train_cfg.get("focal_beta", train_cfg.get("beta", 0.999)))
    gamma = float(train_cfg.get("focal_gamma", train_cfg.get("gamma", 2.0)))

    alpha_weights = compute_effective_num_weights(class_counts, beta=beta).to(device)
    criterion = FocalLoss(alpha=alpha_weights, gamma=gamma)

    # 4. DataLoaders
    batch_size = 32 if smoke_test else int(train_cfg.get("batch_size", 128))
    train_dataset = InMemoryECGDataset(signals[train_idx], rr_features[train_idx], labels[train_idx])
    val_dataset = InMemoryECGDataset(signals[val_idx], rr_features[val_idx], labels[val_idx])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 5. Инициализация модели и оптимизатора
    model = HybridECGClassifier(
        cnn_embed_dim=int(model_cfg.get("cnn_embed_dim", 128)),
        rr_embed_dim=int(model_cfg.get("rr_embed_dim", 32)),
        num_classes=NUM_CLASSES,
        dropout=float(model_cfg.get("dropout", 0.3)),
    ).to(device)

    lr = float(train_cfg.get("learning_rate", train_cfg.get("lr", 1e-3)))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))
    epochs = 2 if smoke_test else int(train_cfg.get("epochs", 40))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 6. Цикл обучения и сохранение лучшего чекпоинта
    ckpt_dir = Path(train_cfg.get("checkpoint_dir", "checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_name = "smoke_best_model.pt" if smoke_test else "best_model.pt"
    best_ckpt_path = ckpt_dir / ckpt_name

    best_val_f1 = -1.0

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
        val_loss, val_f1 = evaluate_epoch(model, val_loader, criterion, device)

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Macro-F1: {val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_macro_f1": best_val_f1,
                    "config": cfg,
                },
                best_ckpt_path,
            )

    print(f"\n[OK] Обучение завершено! Лучший Val Macro-F1: {best_val_f1:.4f}")
    print(f"[OK] Чекпоинт сохранён в: '{best_ckpt_path}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Обучение гибридного классификатора ЭКГ")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Путь к файлу конфигурации")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Запуск тестового обучения на 2 эпохи на синтетических данных",
    )
    args = parser.parse_args()
    train(args.config, smoke_test=args.smoke_test)