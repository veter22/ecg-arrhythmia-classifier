"""
Сглаженный Focal Loss для несбалансированных классов ЭКГ — [ML-02]
===================================================================
Ветка: feature/ml-model
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
    Безопасно обрабатывает классы с нулевым числом примеров в фолде.
    """
    counts = np.asarray(class_counts, dtype=np.float64)
    assert np.all(counts >= 0), f"Количество примеров не может быть отрицательным: {counts}"
    assert np.any(counts > 0), "Хотя бы один класс должен содержать > 0 примеров!"
    assert 0.0 <= beta < 1.0, f"Параметр beta должен быть в диапазоне [0, 1), получено: {beta}"

    weights = np.zeros_like(counts, dtype=np.float64)
    positive_mask = counts > 0
    effective_num = 1.0 - np.power(beta, counts[positive_mask])
    weights[positive_mask] = (1.0 - beta) / effective_num

    # Нормализация: сумма весов == K (средний вес == 1.0)
    weights = weights / np.sum(weights) * len(counts)
    return torch.tensor(weights, dtype=torch.float32)


class FocalLoss(nn.Module):
    """Многоклассовый Focal Loss: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)."""

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"Неподдерживаемый режим reduction='{reduction}'.")

        self.register_buffer("alpha", alpha)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
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
    # 1. Проверка сглаженных весов и обработки нулевого класса
    synth_counts = [1000, 10, 5, 1]
    weights = compute_effective_num_weights(synth_counts, beta=0.999)
    zero_safe_weights = compute_effective_num_weights([100, 0, 5, 1], beta=0.999)
    assert zero_safe_weights[1].item() == 0.0, "Нулевой класс должен получать вес 0.0"

    # 2. Независимая аналитическая проверка формулы FocalLoss (gamma=2.0 vs gamma=0.0)
    logits = torch.tensor([[2.0, 0.5, -1.0, 0.0], [0.1, 1.5, -0.5, 0.2]], requires_grad=True)
    targets = torch.tensor([0, 1], dtype=torch.long)

    probs = torch.softmax(logits.detach(), dim=1)
    pt_manual = probs[torch.arange(2), targets]
    ce_manual = -torch.log(pt_manual)
    expected_focal = torch.mean(weights[targets] * ((1.0 - pt_manual) ** 2.0) * ce_manual)

    criterion = FocalLoss(alpha=weights, gamma=2.0)
    criterion_ce = FocalLoss(alpha=weights, gamma=0.0)

    loss_val = criterion(logits, targets)
    loss_ce_val = criterion_ce(logits, targets)
    loss_val.backward()

    assert torch.allclose(loss_val, expected_focal, atol=1e-6), (
        f"Ошибка формулы FocalLoss: получено {loss_val.item()}, ожидалось {expected_focal.item()}"
    )
    assert not torch.allclose(loss_val, loss_ce_val), "Focal scaling (gamma=2.0) не изменил значение лосса!"
    assert logits.grad is not None, "Градиенты не дошли до логитов!"

    print(f"[ML-02] Все тесты (включая аналитическую проверку) пройдены! Loss={loss_val.item():.4f}")