"""Small models for the benchmark tasks."""

from __future__ import annotations

import math

import torch
from torch import nn


class SmallCNN(nn.Module):
    """ResNet-flavoured CNN for 3x16x16 inputs (stand-in for ResNet-18 on CIFAR-10 on CPU)."""

    def __init__(
        self,
        num_classes: int = 10,
        width: int = 32,
        dropout: float = 0.1,
        per_sample_norm_eps: float = 1e-5,
        input_norm: bool = True,
    ):
        super().__init__()
        self.eps = per_sample_norm_eps
        self.input_norm = input_norm
        self.stem = nn.Sequential(nn.Conv2d(3, width, 3, padding=1), nn.BatchNorm2d(width), nn.ReLU())
        self.block1 = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1),
            nn.BatchNorm2d(width),
            nn.ReLU(),
            nn.Conv2d(width, width, 3, padding=1),
            nn.BatchNorm2d(width),
        )
        self.down = nn.Sequential(
            nn.Conv2d(width, 2 * width, 3, stride=2, padding=1), nn.BatchNorm2d(2 * width), nn.ReLU()
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(2 * width, 2 * width, 3, padding=1),
            nn.BatchNorm2d(2 * width),
            nn.ReLU(),
            nn.Conv2d(2 * width, 2 * width, 3, padding=1),
            nn.BatchNorm2d(2 * width),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout)
        self.fc1 = nn.Linear(2 * width, 64)
        self.head = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_norm:  # per-sample contrast normalization
            mu = x.mean(dim=(1, 2, 3), keepdim=True)
            sd = x.std(dim=(1, 2, 3), keepdim=True)
            x = (x - mu) / (sd + self.eps) if self.eps else (x - mu) / sd
        x = self.stem(x)
        x = torch.relu(x + self.block1(x))
        x = self.down(x)
        x = torch.relu(x + self.block2(x))
        x = self.pool(x).flatten(1)
        x = self.drop(torch.relu(self.fc1(x)))
        return self.head(x)


class TextClassifier(nn.Module):
    def __init__(self, vocab: int, num_classes: int, d: int = 64, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.emb = nn.Embedding(vocab, d, padding_idx=0)
        self.pos = nn.Parameter(torch.randn(1, 128, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, 4, 2 * d, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Linear(d, num_classes)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # mask: 1 = real token, 0 = padding
        h = self.emb(x) + self.pos[:, : x.shape[1]]
        h = self.encoder(h, src_key_padding_mask=(mask == 0))
        m = mask.unsqueeze(-1).to(h.dtype)
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)
        return self.head(pooled)


class TinyGPT(nn.Module):
    def __init__(self, vocab: int, d: int = 64, layers: int = 2, seq_len: int = 64, dropout: float = 0.1):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Parameter(torch.randn(1, seq_len, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, 4, 4 * d, dropout=dropout, batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab)
        self.register_buffer("causal", torch.triu(torch.full((seq_len, seq_len), float("-inf")), 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[1]
        h = self.tok(x) + self.pos[:, :T]
        h = self.blocks(h, mask=self.causal[:T, :T], is_causal=True)
        return self.head(self.ln(h))


def bad_init_(model: nn.Module) -> None:
    """The bug: weights drawn from N(0, 1) regardless of fan-in."""
    for m in model.modules():
        if isinstance(m, nn.Linear | nn.Conv2d):
            nn.init.normal_(m.weight, 0.0, 1.0)


def chance_loss(k: int) -> float:
    return math.log(k)
