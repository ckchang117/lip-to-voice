"""AudioVisualFullModel: flow stream + temporal attention combined (Run D).

Subclasses AudioVisualFlowModel and inserts a TemporalAttention pass after the
flow fusion, before the 640-d concat. Both new contributions use ReZero/zero-gate
init, so the conditioning is bit-identical to baseline at step 0 and the frozen
denoiser stays in-distribution.
"""

from __future__ import annotations

import torch

from .audiovisual_flow_model import AudioVisualFlowModel
from .flow_model import FlowFusion
from .temporal_attention import TemporalAttention


class AudioVisualFullModel(AudioVisualFlowModel):
    def name(self):
        return "AudioVisualFullModel"

    def __init__(self, nets, flow_fusion: FlowFusion | None = None,
                 temporal_attention: TemporalAttention | None = None, flow_hw: int = 88):
        super().__init__(nets, flow_fusion=flow_fusion, flow_hw=flow_hw)
        self.temporal_attention = (
            temporal_attention if temporal_attention is not None else TemporalAttention()
        )

    def forward(self, melspec, mouthroi, face_image, diffusion_steps, cond_drop_prob, flow=None):
        batch = melspec.shape[0]
        if cond_drop_prob > 0:
            # Draw the keep-mask ONCE so all three streams (mouthroi, face, flow) drop together.
            prob_keep_mask = self.prob_mask_like((batch, 1, 1, 1, 1), 1.0 - cond_drop_prob, melspec.device)
            _mouthroi = torch.where(prob_keep_mask, mouthroi, self.mouthroi_null)
            _face_image = torch.where(prob_keep_mask.squeeze(1), face_image, self.face_null)
            _flow = torch.where(prob_keep_mask, flow, self.flow_null) if flow is not None else None
        else:
            _mouthroi = mouthroi
            _face_image = face_image
            _flow = flow

        lip = self.net_lipreading(_mouthroi)              # (B,512,1,T)
        if _flow is not None:
            lip = self.flow_fusion(lip, _flow)            # ReZero gate=0 -> +0 at init
        lip = self.temporal_attention(lip)                # ReZero alpha=0 -> identity at init

        identity = self.net_facial(_face_image)
        identity = identity.repeat(1, 1, 1, lip.shape[-1])
        cond = torch.cat((identity, lip), dim=1).squeeze(2)  # (B,640,T)

        return self.net_diffwave((melspec, diffusion_steps), cond=cond)

    @torch.no_grad()
    def compute_cond(self, mouthroi, face_image, flow=None):
        """Pre-denoiser 640-d conditioning (no CFG dropout). Used by the invariance test."""
        lip = self.net_lipreading(mouthroi)
        if flow is not None:
            lip = self.flow_fusion(lip, flow)
        lip = self.temporal_attention(lip)
        identity = self.net_facial(face_image).repeat(1, 1, 1, lip.shape[-1])
        return torch.cat((identity, lip), dim=1).squeeze(2)
