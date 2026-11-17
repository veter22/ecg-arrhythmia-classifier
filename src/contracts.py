"""
ЕДИНЫЙ КОНТРАКТ ДАННЫХ (Data Contract) — [TASK-00]
==================================================
Связующее звено между модулями Data Lead (feature/data-pipeline)
и ML Lead (feature/ml-model).
"""

from dataclasses import dataclass
from typing import Dict, List
import numpy as np
import torch

# Маппинг сырых аннотаций кардиологов MIT-BIH в 4 целевых класса AAMI EC57.
# Символы, отсутствующие в словаре (класс 'Q': '/', 'f', 'Q', и не-удары), игнорируются.
AAMI_MAPPING: Dict[str, str] = {
    # Normal (N)
    "N": "N",
    "L": "N",
    "R": "N",
    "e": "N",
    "j": "N",
    # Supraventricular ectopic (S)
    "A": "S",
    "a": "S",
    "J": "S",
    "S": "S",
    # Ventricular ectopic (V)
    "V": "V",
    "E": "V",
    # Fusion (F)
    "F": "F",
}

LABEL_TO_IDX: Dict[str, int] = {"N": 0, "S": 1, "V": 2, "F": 3}
IDX_TO_LABEL: Dict[int, str] = {0: "N", 1: "S", 2: "V", 3: "F"}
NUM_CLASSES: int = 4

# Константы размерностей
SIGNAL_WINDOW_LEN: int = 256
RR_FEATURE_NAMES: List[str] = ["pre_RR", "post_RR", "local_RR", "ratio_RR"]
RR_FEATURES_DIM: int = len(RR_FEATURE_NAMES)

# Обязательные ключи в файлах data/processed/*.npz
NPZ_REQUIRED_KEYS: List[str] = ["signals", "rr", "labels", "patient_ids"]


@dataclass
class ECGBatch:
    """Типизированное представление батча, передаваемого из DataLoader в модель."""
    signal: torch.Tensor  # [B, 1, 256], torch.float32
    rr: torch.Tensor      # [B, 4],      torch.float32
    label: torch.Tensor   # [B],         torch.long


def validate_npz_arrays(
    signals: np.ndarray,
    rr: np.ndarray,
    labels: np.ndarray,
    patient_ids: np.ndarray,
) -> None:
    """
    Проверяет подготовленные numpy-массивы перед сохранением в .npz (для Data Lead).
    Выбрасывает AssertionError при нарушении контракта.
    """
    n_samples = len(labels)
    assert n_samples > 0, "Датасет пуст (0 сэмплов)."
    assert signals.shape == (n_samples, SIGNAL_WINDOW_LEN), (
        f"Ожидалась форма signals {(n_samples, SIGNAL_WINDOW_LEN)}, получена {signals.shape}"
    )
    assert rr.shape == (n_samples, RR_FEATURES_DIM), (
        f"Ожидалась форма rr {(n_samples, RR_FEATURES_DIM)}, получена {rr.shape}"
    )
    assert patient_ids.shape == (n_samples,), (
        f"Ожидалась форма patient_ids {(n_samples,)}, получена {patient_ids.shape}"
    )

    assert not np.isnan(signals).any() and not np.isinf(signals).any(), "В signals найдены NaN или Inf."
    assert not np.isnan(rr).any() and not np.isinf(rr).any(), "В rr найдены NaN или Inf."

    unique_labels = set(np.unique(labels).tolist())
    assert unique_labels.issubset({0, 1, 2, 3}), f"Недопустимые метки классов: {unique_labels}"


def validate_batch(batch: Dict[str, torch.Tensor]) -> None:
    """
    Проверяет словарь батча, возвращаемый DataLoader (для интеграционных тестов).
    """
    for key in ("signal", "rr", "label"):
        assert key in batch, f"В батче отсутствует обязательный ключ '{key}'"

    sig, rr, lbl = batch["signal"], batch["rr"], batch["label"]
    batch_size = lbl.shape[0]

    assert sig.shape == (batch_size, 1, SIGNAL_WINDOW_LEN), f"Неверная форма signal: {sig.shape}"
    assert sig.dtype == torch.float32, f"Ожидался dtype float32 для signal, получен {sig.dtype}"

    assert rr.shape == (batch_size, RR_FEATURES_DIM), f"Неверная форма rr: {rr.shape}"
    assert rr.dtype == torch.float32, f"Ожидался dtype float32 для rr, получен {rr.dtype}"

    assert lbl.shape == (batch_size,), f"Неверная форма label: {lbl.shape}"
    assert lbl.dtype == torch.long, f"Ожидался dtype long для label, получен {lbl.dtype}"


if __name__ == "__main__":
    # Проверка контракта на синтетических данных
    fake_batch = {
        "signal": torch.randn(16, 1, SIGNAL_WINDOW_LEN, dtype=torch.float32),
        "rr": torch.randn(16, RR_FEATURES_DIM, dtype=torch.float32),
        "label": torch.randint(0, NUM_CLASSES, (16,), dtype=torch.long),
    }
    validate_batch(fake_batch)
    print("[TASK-00] Contract verification passed successfully!")