"""
Единый контракт данных и батчей для всех модулей пайплайна.
"""

from typing import Dict, Tuple
import numpy as np
import torch

NUM_CLASSES = 4
LABEL_TO_IDX: Dict[str, int] = {"N": 0, "S": 1, "V": 2, "F": 3}
IDX_TO_LABEL: Dict[int, str] = {v: k for k, v in LABEL_TO_IDX.items()}
NPZ_REQUIRED_KEYS: Tuple[str, ...] = ("signals", "rr", "labels", "patient_ids")


def validate_npz_arrays(
    signals: np.ndarray,
    rr: np.ndarray,
    labels: np.ndarray,
    patient_ids: np.ndarray,
    expected_signal_len: int = 256,
    expected_rr_dim: int = 6,
) -> None:
    """Проверяет размерности, типы и отсутствие NaN/Inf перед сохранением и после загрузки .npz."""
    if signals.ndim == 2:
        n_samples, sig_len = signals.shape
    elif signals.ndim == 3 and signals.shape[1] == 1:
        n_samples, _, sig_len = signals.shape
    else:
        raise ValueError(f"Некорректная форма signals: {signals.shape}. Ожидается (N, {expected_signal_len}) или (N, 1, {expected_signal_len}).")

    if sig_len != expected_signal_len:
        raise ValueError(f"Длина окна сигнала {sig_len} != {expected_signal_len}.")

    if rr.shape != (n_samples, expected_rr_dim):
        raise ValueError(f"Некорректная форма rr: {rr.shape}. Ожидается ({n_samples}, {expected_rr_dim}).")

    if labels.shape != (n_samples,) or patient_ids.shape != (n_samples,):
        raise ValueError("Несовпадение числа сэмплов в labels или patient_ids.")

    if not np.isfinite(signals).all() or not np.isfinite(rr).all():
        raise ValueError("Обнаружены значения NaN или Inf в массивах signals / rr.")

    unique_labels = set(np.unique(labels).tolist())
    if not unique_labels.issubset(set(IDX_TO_LABEL.keys())):
        raise ValueError(f"Обнаружены недопустимые метки классов: {unique_labels}")


def validate_batch(batch: Dict[str, torch.Tensor]) -> None:
    """Проверяет структуру и отсутствие NaN/Inf в батче DataLoader."""
    for key in ("signal", "rr", "label"):
        if key not in batch:
            raise KeyError(f"В батче отсутствует обязательный ключ '{key}'.")
    if not torch.isfinite(batch["signal"]).all() or not torch.isfinite(batch["rr"]).all():
        raise ValueError("Обнаружены NaN/Inf в тензорах батча.")