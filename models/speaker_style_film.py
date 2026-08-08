"""FiLM (Feature-wise Linear Modulation) speaker/style fusion for Runs G / H.

Replaces the additive ReZero fusion in `speaker_style.py` with per-channel scale (gamma)
and shift (beta) - the textbook way speaker conditioning is done in modern TTS (VALL-E,
NaturalSpeech, StyleTTS).

  fused_lip = gamma(s_v, s_a) * lip + beta(s_v, s_a)

Identity init (gamma=1, beta=0) guarantees the lip features are passed through unchanged at
step 0, preserving the frozen-denoiser baseline behavior. Combined with a LayerFiLMBank
that modulates the denoiser's per-layer residual outputs (also identity-init), the new
runs achieve true multi-layer speaker injection without touching denoiser weights.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpeakerStyleFiLM(nn.Module):
    """FiLM fusion of (s_v, s_a) into the lip feature stream.

    Inputs:
      lip  : (B, lip_dim, 1, T) - content + identity features
      s_v  : (B, sv_dim)        - visual style embedding (already L2-normalized)
      s_a  : (B, sa_dim)        - audio reference ECAPA embedding (L2-normalized inside)

    Identity init: gamma predicts 1.0 (last linear weights=0, bias=1) and beta predicts 0.0
    (last linear weights=0, bias=0). Composes to `out = 1 * lip + 0 = lip`.
    """

    def __init__(self, sv_dim: int = 192, sa_dim: int = 192, lip_dim: int = 512,
                 hidden: int = 384):
        super().__init__()
        self.lip_dim = lip_dim
        self.gamma_mlp = nn.Sequential(
            nn.Linear(sv_dim + sa_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, lip_dim),
        )
        self.beta_mlp = nn.Sequential(
            nn.Linear(sv_dim + sa_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, lip_dim),
        )
        nn.init.zeros_(self.gamma_mlp[-1].weight)
        nn.init.ones_(self.gamma_mlp[-1].bias)
        nn.init.zeros_(self.beta_mlp[-1].weight)
        nn.init.zeros_(self.beta_mlp[-1].bias)

    def forward(self, lip: torch.Tensor, s_v: torch.Tensor, s_a: torch.Tensor) -> torch.Tensor:
        sa_n = F.normalize(s_a, p=2, dim=-1)
        h = torch.cat([s_v, sa_n], dim=-1)                 # (B, sv+sa)
        gamma = self.gamma_mlp(h).view(-1, self.lip_dim, 1, 1)
        beta = self.beta_mlp(h).view(-1, self.lip_dim, 1, 1)
        return gamma * lip + beta


class LayerFiLM(nn.Module):
    """Single-layer FiLM that scales + shifts a 1D feature map by per-channel gamma and beta
    predicted from a global conditioning vector. Identity init so the layer is a no-op
    at step 0 (preserves frozen-denoiser step-0 invariance).
    """

    def __init__(self, cond_dim: int = 384, feat_channels: int = 512, hidden: int = 256):
        super().__init__()
        self.feat_channels = feat_channels
        self.gamma_proj = nn.Sequential(
            nn.Linear(cond_dim, hidden), nn.SiLU(), nn.Linear(hidden, feat_channels),
        )
        self.beta_proj = nn.Sequential(
            nn.Linear(cond_dim, hidden), nn.SiLU(), nn.Linear(hidden, feat_channels),
        )
        nn.init.zeros_(self.gamma_proj[-1].weight); nn.init.ones_(self.gamma_proj[-1].bias)
        nn.init.zeros_(self.beta_proj[-1].weight);  nn.init.zeros_(self.beta_proj[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """x: (B, C, L); cond: (B, cond_dim) -> gamma(cond)*x + beta(cond)."""
        g = self.gamma_proj(cond).view(-1, self.feat_channels, 1)
        b = self.beta_proj(cond).view(-1, self.feat_channels, 1)
        return g * x + b


class LayerFiLMBank(nn.Module):
    """A `LayerFiLM` per denoiser residual block. Lets the speaker conditioning
    re-modulate the post-block activations at every layer of the frozen denoiser.

    Cond input is the joint speaker vector concat(s_v, normalize(s_a)) of dimension
    cond_dim (default 384). At init, every layer is identity, so the denoiser output
    is bit-identical to the baseline.
    """

    def __init__(self, num_layers: int = 12, feat_channels: int = 512,
                 cond_dim: int = 384, hidden: int = 256):
        super().__init__()
        self.num_layers = num_layers
        self.films = nn.ModuleList([
            LayerFiLM(cond_dim=cond_dim, feat_channels=feat_channels, hidden=hidden)
            for _ in range(num_layers)
        ])
