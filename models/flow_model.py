"""Optical-flow visual stream for LipVoicer (flow-only ablation, Run C).

Adds a motion pathway that fuses into LipVoicer's 512-d lip-reading features via
a ZERO-INITIALIZED gate, so at training step 0 the fused features are bit-identical
to the baseline lip features and the frozen DiffWave denoiser stays in-distribution.

Shapes follow the existing conditioning pathway (models/audiovisual_model.py):
the lip-reading feature is (B, 512, 1, T); the flow feature is produced to match
it exactly so a gated add preserves the (B, 512, 1, T) -> 640-d cond interface.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _gn(num_channels: int, max_groups: int = 8) -> nn.GroupNorm:
    """GroupNorm with a group count that divides num_channels. GroupNorm (not
    BatchNorm) keeps the trainable module free of batch-statistic fragility at
    the small batch sizes used for this frozen-backbone fine-tune."""
    groups = max_groups
    while groups > 1 and num_channels % groups != 0:
        groups //= 2
    return nn.GroupNorm(groups, num_channels)


class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1)
        self.norm = _gn(out_ch)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class FlowEncoder(nn.Module):
    """Per-frame 2D CNN over optical-flow fields -> one feature vector per frame.

    Input:  flow (B, 2, T, H, W)   (2 = optical-flow vector channels dx, dy)
    Output: feat (B, feat_dim, 1, T)   aligned to net_lipreading's (B,512,1,T)
    """

    def __init__(self, in_ch: int = 2, feat_dim: int = 512):
        super().__init__()
        self.feat_dim = feat_dim
        self.stem = nn.Sequential(
            _ConvBlock(in_ch, 32),   # H -> H/2
            _ConvBlock(32, 64),      # /4
            _ConvBlock(64, 128),     # /8
            _ConvBlock(128, 256),    # /16
            _ConvBlock(256, feat_dim),  # /32
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, flow: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = flow.shape
        x = flow.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)  # (B*T, 2, H, W)
        x = self.stem(x)
        x = self.pool(x).reshape(B, T, self.feat_dim)            # (B, T, feat_dim)
        return x.permute(0, 2, 1).unsqueeze(2)                   # (B, feat_dim, 1, T)


class FlowProj(nn.Module):
    """1x1 projection of the flow features. Deliberately NOT zero-initialized.

    Invariance at init comes from the zero gate alone (ReZero-style). Zeroing BOTH
    the gate and this projection would create a dead saddle: the gate's gradient is
    proportional to this projection's output, and this projection's gradient is
    proportional to the gate, so if both are zero every flow parameter has zero
    gradient and nothing can ever learn. A normally-initialized projection keeps the
    gate's gradient live while the zero gate still makes the flow term exactly zero."""

    def __init__(self, feat_dim: int = 512):
        super().__init__()
        self.conv = nn.Conv2d(feat_dim, feat_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class FlowFusion(nn.Module):
    """Gated-additive fusion of the flow stream into the lip-reading features.

    fused = lip + tanh(gate) * FlowProj(FlowEncoder(flow))

    The gate is initialized to 0 (ReZero): tanh(0)=0 makes the flow term EXACTLY
    zero at init, so `fused == lip` bit-for-bit and the frozen denoiser stays
    in-distribution. The projection is non-zero so the gate's gradient stays live
    (proportional to the projection output), letting the module learn to ramp the
    flow stream in. Only this module is trained. Output keeps the (B,512,1,T) shape,
    preserving the frozen 640-d conditioning interface after the face concat.
    """

    def __init__(self, in_ch: int = 2, feat_dim: int = 512):
        super().__init__()
        self.encoder = FlowEncoder(in_ch=in_ch, feat_dim=feat_dim)
        self.proj = FlowProj(feat_dim=feat_dim)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, lip_feature: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        f = self.proj(self.encoder(flow))           # (B, 512, 1, T)
        return lip_feature + torch.tanh(self.gate) * f  # gate=0 -> +0 exactly at init
