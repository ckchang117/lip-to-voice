"""Audio-visual speaker/style consistency modules (Runs E + F).

Two new modules:
1. VisualStyleEncoder - operates on face-frame summary features ONLY (no mouthroi),
   so it cannot encode phonetic content. Outputs a 192-d speaker/style embedding
   from a sequence of face crops.
2. SpeakerStyleFusion - gated-additive (ReZero) fusion of the visual style embedding
   + a precomputed audio speaker embedding (ECAPA-TDNN, frozen) into the 512-d lip
   stream. Step-0 invariance via alpha = 0 init.

The contrastive InfoNCE alignment between VisualStyleEncoder(target_video) and
ECAPA(audio_reference) is computed in the trainer, not here.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class _PosEnc(nn.Module):
    """Sinusoidal positional encoding for the face-sequence transformer."""

    def __init__(self, dim: int = 128, max_len: int = 64):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float) * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):
        # x: (B, T, D)
        return x + self.pe[: x.size(1)].to(dtype=x.dtype)


class VisualStyleEncoder(nn.Module):
    """Face-only visual speaker/style encoder.

    Input:  (B, M, 128) - per-frame face features from the frozen net_facial
            (face crops, NOT mouthroi -> no phonetic content leakage).
    Output: (B, 192) - L2-normalized speaker/style embedding.

    Bandwidth bottleneck: only face features (which capture identity + broad facial
    motion / expressiveness / head pose), never mouth ROIs. The encoder physically
    cannot encode "which phoneme is being said."
    """

    def __init__(self, in_dim: int = 128, emb_dim: int = 192, max_seq: int = 32,
                 heads: int = 4, ff_mult: int = 4, dropout: float = 0.15):
        super().__init__()
        self.pos = _PosEnc(dim=in_dim, max_len=max_seq)
        self.norm1 = nn.LayerNorm(in_dim)
        self.attn = nn.MultiheadAttention(in_dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(in_dim)
        self.ff = nn.Sequential(
            nn.Linear(in_dim, in_dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_dim * ff_mult, in_dim),
            nn.Dropout(dropout),
        )
        self.proj = nn.Linear(in_dim, emb_dim)

    def forward(self, face_features: torch.Tensor) -> torch.Tensor:
        # face_features: (B, M, 128)
        x = self.pos(face_features)
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        x = x.mean(dim=1)                              # (B, 128) - temporal mean pool
        emb = self.proj(x)                             # (B, 192)
        return F.normalize(emb, p=2, dim=-1)           # L2-normalized


class SpeakerStyleFusion(nn.Module):
    """ReZero gated fusion of the visual-style + audio-speaker embeddings into lip features.

    Mirrors FlowFusion's gated-additive design so the frozen denoiser's 640-d conditioning
    interface is preserved.

    fused_lip = lip + tanh(alpha) * broadcast_T( proj( concat(s_v, s_a) ) )

    alpha is a scalar nn.Parameter initialized to 0 -> contribution is exactly 0 at init
    -> invariance to the frozen baseline conditioning at step 0.
    """

    def __init__(self, sv_dim: int = 192, sa_dim: int = 192, lip_dim: int = 512):
        super().__init__()
        self.proj = nn.Linear(sv_dim + sa_dim, lip_dim)
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, lip_feature: torch.Tensor, s_v: torch.Tensor, s_a: torch.Tensor) -> torch.Tensor:
        # lip_feature: (B, lip_dim, 1, T)
        # s_v, s_a:    (B, 192)
        combined = torch.cat([s_v, s_a], dim=-1)             # (B, 384)
        h = self.proj(combined)                              # (B, 512)
        h = h.unsqueeze(-1).unsqueeze(-1)                    # (B, 512, 1, 1)
        h = h.expand(-1, -1, 1, lip_feature.size(-1))        # (B, 512, 1, T)
        return lip_feature + torch.tanh(self.alpha) * h      # alpha=0 -> identity at init
