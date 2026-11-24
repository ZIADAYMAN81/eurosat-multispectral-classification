"""Model zoo: the original fully connected network and a compact convolutional network."""
from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


class DeepMLP(nn.Module):
    """Fully connected deep network on the flattened 13 x 64 x 64 input (53,248 features).

    This is the architecture explored in the original coursework, where five width and
    learning-rate variations were compared.
    """

    def __init__(self, input_size: int = 13 * 64 * 64, hidden_sizes: Sequence[int] = (2048, 1024),
                 num_classes: int = 10, dropout: float = 0.0):
        super().__init__()
        layers, prev = [], input_size
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU(inplace=True)]
            if dropout:
                layers.append(nn.Dropout(dropout))
            prev = h
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(prev, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(x.flatten(1)))


def _block(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class SpectralCNN(nn.Module):
    """Compact CNN that keeps the spatial structure the MLP throws away.

    A 1 x 1 "spectral" convolution first mixes the 13 bands per pixel, then three convolutional
    blocks learn spatial texture; global average pooling keeps the parameter count small.
    """

    def __init__(self, in_channels: int = 13, num_classes: int = 10, width: int = 32):
        super().__init__()
        self.spectral = nn.Sequential(nn.Conv2d(in_channels, width, 1, bias=False), nn.BatchNorm2d(width),
                                      nn.ReLU(inplace=True))
        self.features = nn.Sequential(_block(width, width), _block(width, width * 2), _block(width * 2, width * 4))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.2),
                                  nn.Linear(width * 4, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(self.spectral(x)))


def build(name: str, num_classes: int = 10, **kwargs) -> nn.Module:
    if name == "mlp":
        return DeepMLP(num_classes=num_classes, **kwargs)
    if name == "cnn":
        return SpectralCNN(num_classes=num_classes, **kwargs)
    raise ValueError(f"unknown model {name!r} (use 'mlp' or 'cnn')")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
