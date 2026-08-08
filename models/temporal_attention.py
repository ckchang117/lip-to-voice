"""Temporal-attention block stack over lip features (Run D = flow + attention).

A pre-LN bidirectional transformer encoder applied to the lip-reader features so each
frame gets whole-utterance context - the mechanism we expect to improve phonetic
discrimination (and therefore WER), complementing the flow stream's perceptual gains.

Step-0 invariance: each block uses ReZero - the attention and FFN residuals are
scaled by learnable scalars `alpha`, `beta` initialized to 0. So at init every block
returns its input bit-for-bit, the whole stack is identity, and the frozen denoiser
sees baseline conditioning unchanged. Position information is injected *inside* the
attention residual branch (added to the LN'd query/key/value source) so the skip
path is never modified.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class AttnBlock(nn.Module):
    def __init__(self, dim: int = 512, heads: int = 8, ff_mult: int = 4,
                 dropout: float = 0.15, drop_path: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ff_mult, dim),
            nn.Dropout(dropout),
        )
        # ReZero scalars: at init alpha=beta=0 -> block is identity bit-for-bit.
        self.alpha = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.zeros(1))
        self.drop_path = drop_path

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        # x: (B, T, dim), pos: (T, dim)
        # Stochastic depth (training only): skip the block with probability drop_path.
        if self.training and self.drop_path > 0.0 and torch.rand(1, device=x.device).item() < self.drop_path:
            return x
        # Position info enters only via the LN'd attention input - skip path is untouched.
        h = self.norm1(x) + pos
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.alpha * attn_out

        h = self.norm2(x)
        x = x + self.beta * self.ff(h)
        return x


class TemporalAttention(nn.Module):
    """4-block pre-LN transformer over lip features (B, 512, 1, T).

    Identity at init (ReZero), so `temporal_attention(lip) == lip` bit-exact and the
    frozen denoiser receives baseline conditioning at training step 0.
    """

    def __init__(self, dim: int = 512, depth: int = 4, heads: int = 8, ff_mult: int = 4,
                 dropout: float = 0.15, drop_path_max: float = 0.15, max_len: int = 1024):
        super().__init__()
        # Sinusoidal positional encoding (registered buffer, not a parameter).
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float) * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pos_embed", pe, persistent=False)

        # Stochastic-depth probabilities linearly scaled 0 -> drop_path_max across blocks.
        drop_paths = torch.linspace(0.0, drop_path_max, depth).tolist()
        self.blocks = nn.ModuleList([
            AttnBlock(dim=dim, heads=heads, ff_mult=ff_mult, dropout=dropout, drop_path=drop_paths[i])
            for i in range(depth)
        ])

    def forward(self, lip_feature: torch.Tensor) -> torch.Tensor:
        # (B, dim, 1, T) -> (B, T, dim)
        B, D, _, T = lip_feature.shape
        x = lip_feature.squeeze(2).permute(0, 2, 1).contiguous()
        pos = self.pos_embed[:T].to(dtype=x.dtype)
        for block in self.blocks:
            x = block(x, pos)
        # (B, T, dim) -> (B, dim, 1, T)
        return x.permute(0, 2, 1).unsqueeze(2).contiguous()
