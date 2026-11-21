"""
Сглаженный Focal Loss для несбалансированных классов ЭКГ — [ML-02]
===================================================================
Ветка: feature/ml-model
Реализует:
1. Расчёт весов классов на основе эффективного числа сэмплов (Cui et al., CVPR 2019).
2. Многоклассовый Focal Loss (Lin et al., ICCV 2017) для стабилизации обучения
   и сохранения высокого Precision на редких аритмиях ('S', 'V', 'F').
"""

import sys
from pathlib import Path
from typing import Optional, Sequence

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_effective_num_weights(
    class_counts: Sequence[int],
    beta: float = 0.999,
) -> torch.Tensor:
    """
    Вычисляет сглаженные веса классов по формуле эффективного числа сэмплов:
        w_c = (1 - beta) / (1 - beta^{N_c})

    Веса нормализуются так, чтобы их сумма равнялась количеству классов K
    (сохраняет масштаб learning rate при обучении).

    Args:
        class_counts: количество примеров каждого класса в обучающей выборке [N_0, ..., N_{K-1}]
        beta: коэффициент сглаживания (по ТЗ: 0.999)

    Returns:
        Тензор весов формы [K] типа torch.float32
    """
    counts = np.asarray(class_counts, dtype=np.float64)
    assert np.all(counts > 0), f"Все классы в train должны иметь > 0 примеров, получено: {counts}"
    assert 0.0 <= beta < 1.0, f"Параметр beta должен быть в диапазоне [0, 1), получено: {beta}"

    effective_num = 1.0 - np.power(beta, counts)
    weights = (1.0 - beta) / effective_num

    # Нормализация: сумма весов == K (средний вес == 1.0)
    weights = weights / np.sum(weights) * len(counts)
    return torch.tensor(weights, dtype=torch.float32)


class FocalLoss(nn.Module):
    """
    Многоклассовый Focal Loss с поддержкой сглаженных весов alpha:
        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"Неподдерживаемый режим reduction='{reduction}'. Используйте 'mean', 'sum' или 'none'.")

        self.register_buffer("alpha", alpha)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  предсказания модели до Softmax, форма [B, C]
            targets: истинные индексы классов, форма [B], dtype=torch.long
        """
        # log_pt = log(softmax(logits))[target] = -CE(logits, targets)
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)

        focal_factor = (1.0 - pt) ** self.gamma
        loss = focal_factor * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


if __name__ == "__main__":
    # Проверка по критериям приёмки [ML-02] на синтетическом дисбалансе 1000 : 10 : 5 : 1
    synth_counts = [1000, 10, 5, 1]
    weights = compute_effective_num_weights(synth_counts, beta=0.999)

    # Сравнение с жёсткими линейными обратными весами N / (K * N_c)
    linear_weights = np.sum(synth_counts) / (len(synth_counts) * np.asarray(synth_counts))
    print(f"[ML-02] Синтетический дисбаланс: {synth_counts}")
    print(f"  Линейные обратные веса (запрещённые): {np.round(linear_weights, 2)}")
    print(f"  Сглаженные веса (beta=0.999):         {np.round(weights.numpy(), 4)}")

    # Проверка расчёта лосса и градиентов
    criterion = FocalLoss(alpha=weights, gamma=2.0)
    logits = torch.randn(8, 4, requires_grad=True)
    targets = torch.tensor([0, 0, 1, 2, 3, 0, 1, 2], dtype=torch.long)

    loss_val = criterion(logits, targets)
    loss_val.backward()

    assert not torch.isnan(loss_val) and loss_val.item() > 0, "Некорректное значение лосса!"
    assert logits.grad is not None, "Градиенты не дошли до логитов!"
    print(f"[ML-02] Тест пройден успешно! Значение FocalLoss: {loss_val.item():.4f}")