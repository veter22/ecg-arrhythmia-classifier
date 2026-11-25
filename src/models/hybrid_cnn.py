"""
Гибридная нейросетевая архитектура 1D-Dilated-ResNet + RR-MLP — [ML-02 v2]
"""

import torch
import torch.nn as nn


class ResBlock1D(nn.Module):
    """Остаточный 1D-блок с дилатацией для увеличения рецептивного поля без потери разрешения."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        padding = dilation
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, dilation=dilation, bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        return self.act(out + self.shortcut(x))


class HybridECGClassifier(nn.Module):
    """
    Двухпоточная архитектура:
      1) Морфологическая ветка (Dilated 1D-ResNet + 3-зонный темпоральный пулинг P / QRS / ST-T)
      2) Ритмологическая ветка (MLP по 6 признакам вариабельности RR-интервалов)
    """

    def __init__(
        self,
        in_channels: int = 1,
        signal_length: int = 256,
        rr_features_dim: int = 6,
        cnn_embed_dim: int = 128,
        rr_embed_dim: int = 32,
        num_classes: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.signal_length = signal_length
        self.rr_features_dim = rr_features_dim

        # Свёрточный ствол с расширенным рецептивным полем (~220 отсчётов / ~610 мс при 360 Гц)
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=9, stride=2, padding=4, bias=False),
            nn.BatchNorm1d(32),
            nn.GELU(),
        )
        self.layer1 = ResBlock1D(32, 64, stride=2, dilation=1, dropout=0.1)
        self.layer2 = ResBlock1D(64, 128, stride=2, dilation=2, dropout=0.15)
        self.layer3 = ResBlock1D(128, cnn_embed_dim, stride=1, dilation=4, dropout=0.15)

        # Сохраняем порядок фаз сердечного цикла: [зона P | зона QRS | зона ST-T]
        self.temporal_pyramid = nn.AdaptiveAvgPool1d(3)
        self.morph_proj = nn.Sequential(
            nn.Linear(cnn_embed_dim * 3, cnn_embed_dim),
            nn.LayerNorm(cnn_embed_dim),
            nn.GELU(),
        )

        # Ветка RR-интервалов
        self.rr_mlp = nn.Sequential(
            nn.Linear(rr_features_dim, 32),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(32, rr_embed_dim),
            nn.BatchNorm1d(rr_embed_dim),
            nn.GELU(),
        )

        # Классификационная голова
        fused_dim = cnn_embed_dim + rr_embed_dim
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fused_dim, 96),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(96, num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, signal: torch.Tensor, rr: torch.Tensor) -> torch.Tensor:
        x = self.stem(signal)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # Вектор размерности (B, cnn_embed_dim * 3) с сохранением порядка сегментов
        x_pyr = self.temporal_pyramid(x).flatten(1)
        morph_emb = self.morph_proj(x_pyr)

        rr_emb = self.rr_mlp(rr)
        fused = torch.cat([morph_emb, rr_emb], dim=1)
        return self.classifier(fused)