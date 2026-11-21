"""
Гибридная архитектура нейросети: 1D ResNet + RR-MLP — [ML-01]
=============================================================
Ветка: feature/ml-model
Объединяет энкодер морфологии ЭКГ-сигнала ([B, 1, 256])
и энкодер контекста сердечного ритма ([B, 4]).
"""

import sys
from pathlib import Path

# Позволяет запускать файл напрямую (через Run в IDE) без ошибки импорта src
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
import torch.nn as nn


class ResidualBlock1D(nn.Module):
    """Одномерный остаточный блок (1D ResNet) для сохранения мелких деталей формы ЭКГ."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5, stride: int = 1):
        super().__init__()
        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()

        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)

        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.act(out + residual)
        return out


class HybridECGClassifier(nn.Module):
    """
    Гибридный классификатор аритмий (Морфология 1D ResNet + Ритм RR-MLP).

    Входы (по контракту src/contracts.py):
        signal: [B, 1, 256] — нормализованное окно ЭКГ вокруг R-пика
        rr:     [B, 4]      — [pre_RR, post_RR, local_RR, ratio_RR]

    Выход:
        logits: [B, 4]      — ненормализованные логиты для классов ['N', 'S', 'V', 'F']
    """

    def __init__(
        self,
        in_channels: int = 1,
        rr_features_dim: int = 4,
        cnn_embed_dim: int = 128,
        rr_embed_dim: int = 32,
        num_classes: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()

        # 1. Ветка морфологии: [B, 1, 256] -> [B, cnn_embed_dim]
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.GELU(),
        )

        self.res_blocks = nn.Sequential(
            ResidualBlock1D(32, 32, kernel_size=5, stride=1),
            nn.MaxPool1d(kernel_size=2),  # 256 -> 128
            ResidualBlock1D(32, 64, kernel_size=5, stride=1),
            nn.MaxPool1d(kernel_size=2),  # 128 -> 64
            ResidualBlock1D(64, cnn_embed_dim, kernel_size=3, stride=1),
            nn.AdaptiveAvgPool1d(1),      # [B, cnn_embed_dim, 1]
            nn.Flatten(),                  # [B, cnn_embed_dim]
        )

        # 2. Ветка ритма (RR-интервалы): [B, 4] -> [B, rr_embed_dim]
        self.rhythm_encoder = nn.Sequential(
            nn.Linear(rr_features_dim, rr_embed_dim),
            nn.LayerNorm(rr_embed_dim),
            nn.GELU(),
            nn.Linear(rr_embed_dim, rr_embed_dim),
            nn.LayerNorm(rr_embed_dim),
            nn.GELU(),
        )

        # 3. Классифицирующая голова: [B, cnn_embed_dim + rr_embed_dim] -> [B, num_classes]
        fusion_dim = cnn_embed_dim + rr_embed_dim
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(64, num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Инициализация весов для стабильного старта обучения."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, signal: torch.Tensor, rr: torch.Tensor) -> torch.Tensor:
        x_morph = self.stem(signal)
        x_morph = self.res_blocks(x_morph)

        x_rhythm = self.rhythm_encoder(rr)

        fused = torch.cat([x_morph, x_rhythm], dim=1)
        logits = self.classifier(fused)
        return logits


if __name__ == "__main__":
    # Smoke-тест по критериям приёмки [ML-01]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HybridECGClassifier().to(device)

    batch_size = 16
    dummy_signal = torch.randn(batch_size, 1, 256, device=device)
    dummy_rr = torch.randn(batch_size, 4, device=device)
    dummy_targets = torch.randint(0, 4, (batch_size,), device=device)

    # 1. Прямой проход (forward)
    logits = model(dummy_signal, dummy_rr)
    assert logits.shape == (batch_size, 4), f"Ожидалась форма {(batch_size, 4)}, получена {logits.shape}"

    # 2. Обратное распространение ошибки (backward)
    loss = nn.CrossEntropyLoss()(logits, dummy_targets)
    loss.backward()

    has_grads = all(p.grad is not None for p in model.parameters() if p.requires_grad)
    assert has_grads, "Не все обучаемые параметры получили градиенты!"

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[ML-01] Smoke-тест пройден успешно на устройстве: {device}")
    print(f"Форма выхода: {logits.shape} | Обучаемых параметров: {total_params:,}")