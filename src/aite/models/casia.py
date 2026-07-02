"""PyTorch model definitions for CASIA-OLHWDB1.1 handwriting classification.

All models use the variable-length batch convention::

    forward(ink, lengths, mask) -> logits [B, num_classes]

where ``ink`` is [B, T, 2], ``lengths`` is [B] (valid timestep counts),
and ``mask`` is a boolean tensor [B, T] (True = valid, False = padding).

Architectures
-------------
CNN1DClassifier     : 1D-CNN with GELU + masked global average pooling
BLSTMClassifier     : Bidirectional LSTM with masked mean pooling
TransformerClassifier: Encoder-only Transformer with sinusoidal positional encoding

Factory
-------
get_casia_model(arch, num_classes=3755) -> nn.Module
"""
from __future__ import annotations

import math
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


# ---------------------------------------------------------------------------
# CNN-1D
# ---------------------------------------------------------------------------

class _ConvBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dropout: float, use_batchnorm: bool) -> None:
        super().__init__()
        padding = kernel_size // 2
        layers: List[nn.Module] = [
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, stride=1, padding=padding)
        ]
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(out_ch))
        layers.extend([nn.GELU(), nn.Dropout(dropout)])
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CNN1DClassifier(nn.Module):
    """1D-CNN with masked global average pooling.

    Default config matches the trained checkpoint (channels=[64,128,256,256],
    kernel_sizes=[5,5,3,3], dropout=0.1, use_batchnorm=True).

    Input : ink [B, T, 2], lengths [B], mask [B, T]
    Output: logits [B, num_classes]
    """

    def __init__(
        self,
        num_classes: int,
        channels: List[int] = None,
        kernel_sizes: List[int] = None,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()
        if channels is None:
            channels = [64, 128, 256, 256]
        if kernel_sizes is None:
            kernel_sizes = [5, 5, 3, 3]

        blocks: List[nn.Module] = []
        in_ch = 2
        for out_ch, k in zip(channels, kernel_sizes):
            blocks.append(_ConvBlock1D(in_ch, out_ch, k, dropout, use_batchnorm))
            in_ch = out_ch
        self.encoder = nn.Sequential(*blocks)

        feat_dim = channels[-1]
        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, feat_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, num_classes),
        )

    def forward(self, ink: torch.Tensor, lengths: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = ink.transpose(1, 2)     # [B, 2, T]
        x = self.encoder(x)          # [B, C, T]
        x = x.transpose(1, 2)        # [B, T, C]
        valid = mask.unsqueeze(-1).to(dtype=x.dtype)
        summed = (x * valid).sum(dim=1)
        denom = lengths.clamp_min(1).unsqueeze(1).to(dtype=x.dtype)
        pooled = summed / denom
        return self.head(pooled)


# ---------------------------------------------------------------------------
# BLSTM
# ---------------------------------------------------------------------------

class BLSTMClassifier(nn.Module):
    """Bidirectional LSTM with masked mean pooling.

    Default config matches the trained checkpoint (hidden_size=256, num_layers=3,
    dropout=0.2).

    Input : ink [B, T, 2], lengths [B], mask [B, T]
    Output: logits [B, num_classes]
    """

    def __init__(
        self,
        num_classes: int,
        hidden_size: int = 256,
        num_layers: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, num_classes),
        )

    def forward(self, ink: torch.Tensor, lengths: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        packed = pack_padded_sequence(ink, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=ink.shape[1])
        valid = mask.unsqueeze(-1).to(dtype=out.dtype)
        summed = (out * valid).sum(dim=1)
        denom = lengths.clamp_min(1).unsqueeze(1).to(dtype=out.dtype)
        pooled = summed / denom
        return self.head(pooled)


# ---------------------------------------------------------------------------
# Transformer
# ---------------------------------------------------------------------------

class _SinusoidalPE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048) -> None:
        super().__init__()
        self.d_model = d_model
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerClassifier(nn.Module):
    """Encoder-only Transformer with sinusoidal positional encoding.

    Default config matches the trained checkpoint (d_model=128, num_heads=4,
    num_layers=4, ff_dim=256, dropout=0.1).

    Input : ink [B, T, 2], lengths [B], mask [B, T]
    Output: logits [B, num_classes]
    """

    def __init__(
        self,
        num_classes: int,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
        ff_dim: int = 256,
        dropout: float = 0.1,
        max_pos_len: int = 2048,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(2, d_model)
        self.pos_encoder = _SinusoidalPE(d_model, max_pos_len)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            enc_layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, ink: torch.Tensor, lengths: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(ink)
        x = self.pos_encoder(x)
        padding_mask = ~mask  # TransformerEncoder: True = ignore
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        valid = mask.unsqueeze(-1).to(dtype=x.dtype)
        summed = (x * valid).sum(dim=1)
        denom = lengths.clamp_min(1).unsqueeze(1).to(dtype=x.dtype)
        pooled = summed / denom
        return self.head(pooled)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_casia_model(arch: str, num_classes: int = 3755) -> nn.Module:
    """Build a fresh (untrained) CASIA model with paper-default hyperparameters.

    Parameters
    ----------
    arch        : 'cnn1d' | 'blstm' | 'transformer'
    num_classes : number of output classes (default 3755 for CASIA-OLHWDB1.1)
    """
    if arch == "cnn1d":
        return CNN1DClassifier(num_classes)
    if arch == "blstm":
        return BLSTMClassifier(num_classes)
    if arch == "transformer":
        return TransformerClassifier(num_classes)
    raise ValueError(f"Unknown arch '{arch}'. Choose from: cnn1d, blstm, transformer.")
