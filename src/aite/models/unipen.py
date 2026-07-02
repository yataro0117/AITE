"""PyTorch model definitions for Unipen handwriting classification.

Input convention: (B, T, 2) float32, values in [-1, 1], T=50 for Unipen.
All forward() calls return logits (pre-softmax).

Architectures
-------------
CNN3Unipen       : 3 conv blocks (64→128→128) + FC head
CNN4Unipen       : 4 conv blocks (64→128→128→128) + FC head
BLSTM2Unipen     : Bidirectional LSTM (100 units/direction) + Linear head
TransformerUnipen: Encoder-only Transformer (d=64, 4 blocks)

Factory
-------
get_unipen_model(arch, dataset, seq_len=50) -> nn.Module
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Shared building block
# ---------------------------------------------------------------------------

class _ConvBlock(nn.Module):
    """Conv1d(k=3, same) → BN → pointwise-linear(k=1) → ReLU → MaxPool(2)."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm1d(out_ch)
        self.pw = nn.Conv1d(out_ch, out_ch, kernel_size=1)
        self.pool = nn.MaxPool1d(kernel_size=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(self.pw(x))
        return self.pool(x)


def _conv_flat_size(seq_len: int, num_pools: int, last_ch: int) -> int:
    t = seq_len
    for _ in range(num_pools):
        t = t // 2
    return t * last_ch


# ---------------------------------------------------------------------------
# CNN-3
# ---------------------------------------------------------------------------

class CNN3Unipen(nn.Module):
    """1D CNN-3: three ConvBlocks (64→128→128) + FC head.

    Input : (B, T, 2)
    Output: logits (B, nb_class)
    """

    def __init__(self, nb_class: int, seq_len: int = 50) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            _ConvBlock(2, 64),
            _ConvBlock(64, 128),
            _ConvBlock(128, 128),
        )
        flat_size = _conv_flat_size(seq_len, num_pools=3, last_ch=128)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, nb_class),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)  # (B, T, 2) → (B, 2, T)
        x = self.blocks(x)
        return self.fc(x)


# ---------------------------------------------------------------------------
# CNN-4
# ---------------------------------------------------------------------------

class CNN4Unipen(nn.Module):
    """1D CNN-4: four ConvBlocks (64→128→128→128) + FC head.

    Input : (B, T, 2)
    Output: logits (B, nb_class)
    """

    def __init__(self, nb_class: int, seq_len: int = 50) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            _ConvBlock(2, 64),
            _ConvBlock(64, 128),
            _ConvBlock(128, 128),
            _ConvBlock(128, 128),
        )
        flat_size = _conv_flat_size(seq_len, num_pools=4, last_ch=128)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, nb_class),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)  # (B, T, 2) → (B, 2, T)
        x = self.blocks(x)
        return self.fc(x)


# ---------------------------------------------------------------------------
# BLSTM-2
# ---------------------------------------------------------------------------

class BLSTM2Unipen(nn.Module):
    """Bidirectional LSTM (100 units/direction) + Linear logit head.

    Input : (B, T, 2)
    Output: logits (B, nb_class)
    """

    def __init__(self, nb_class: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=2,
            hidden_size=100,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Linear(200, nb_class)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h, _) = self.lstm(x)
        h = torch.cat([h[0], h[1]], dim=1)  # (B, 200)
        return self.fc(h)


# ---------------------------------------------------------------------------
# Transformer
# ---------------------------------------------------------------------------

class _TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ff_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_n = self.norm1(x)
        attn_out, _ = self.attn(x_n, x_n, x_n)
        x = x + self.drop1(attn_out)
        x = x + self.ff(self.norm2(x))
        return x


class TransformerUnipen(nn.Module):
    """Encoder-only Transformer for Unipen (x,y) time-series classification.

    Input : (B, T, 2)
    Output: logits (B, nb_class)
    """

    def __init__(
        self,
        nb_class: int,
        seq_len: int = 50,
        d_model: int = 64,
        num_heads: int = 4,
        num_layers: int = 4,
        ff_dim: int = 128,
        dropout: float = 0.1,
        head_units: int = 256,
        head_dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(2, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            _TransformerBlock(d_model, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model, eps=1e-6)
        self.head = nn.Sequential(
            nn.Linear(d_model, head_units),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_units, nb_class),
        )
        self.register_buffer("_positions", torch.arange(seq_len))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        x = self.input_proj(x) + self.pos_emb(self._positions[:T])
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        return self.head(x)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_NB_CLASS: dict[str, int] = {"1a": 10, "1b": 26, "1c": 26}


def get_unipen_model(arch: str, dataset: str, seq_len: int = 50) -> nn.Module:
    """Build a fresh (untrained) Unipen model.

    Parameters
    ----------
    arch    : 'cnn3' | 'cnn4' | 'blstm2' | 'transformer'
    dataset : '1a' | '1b' | '1c'
    seq_len : input sequence length (default 50)
    """
    if dataset not in _NB_CLASS:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose from: {list(_NB_CLASS)}")
    nb_class = _NB_CLASS[dataset]
    if arch == "cnn3":
        return CNN3Unipen(nb_class, seq_len)
    if arch == "cnn4":
        return CNN4Unipen(nb_class, seq_len)
    if arch == "blstm2":
        return BLSTM2Unipen(nb_class)
    if arch == "transformer":
        return TransformerUnipen(nb_class, seq_len)
    raise ValueError(f"Unknown arch '{arch}'. Choose from: cnn3, cnn4, blstm2, transformer.")
