"""AudioVisualFlowModel: AudioVisualModel + an optical-flow stream.

Subclassing (rather than editing the parent) keeps the submodule names
`net_lipreading.*`, `net_facial.*`, `net_diffwave.*` identical, so the released
1M-step checkpoint loads with `strict=False` leaving only `flow_fusion.*` as
freshly-initialized (zero) parameters. The 640-d conditioning interface to the
frozen denoiser is unchanged.
"""

from __future__ import annotations

import torch

from .audiovisual_model import AudioVisualModel
from .flow_model import FlowFusion


class AudioVisualFlowModel(AudioVisualModel):
    def name(self):
        return "AudioVisualFlowModel"

    def __init__(self, nets, flow_fusion: FlowFusion | None = None, flow_hw: int = 88):
        super().__init__(nets)
        self.flow_fusion = flow_fusion if flow_fusion is not None else FlowFusion()
        # Null flow for classifier-free guidance: zeros = "no motion" (and with the
        # zero-init gate this is in-distribution at step 0). Shape broadcasts to
        # (B, 2, T, H, W).
        self.register_buffer("flow_null", torch.zeros(1, 2, 1, flow_hw, flow_hw))

    def forward(self, melspec, mouthroi, face_image, diffusion_steps, cond_drop_prob, flow=None):
        batch = melspec.shape[0]
        if cond_drop_prob > 0:
            # Draw the keep-mask ONCE so all three streams are dropped together.
            prob_keep_mask = self.prob_mask_like((batch, 1, 1, 1, 1), 1.0 - cond_drop_prob, melspec.device)
            _mouthroi = torch.where(prob_keep_mask, mouthroi, self.mouthroi_null)
            _face_image = torch.where(prob_keep_mask.squeeze(1), face_image, self.face_null)
            _flow = torch.where(prob_keep_mask, flow, self.flow_null) if flow is not None else None
        else:
            _mouthroi = mouthroi
            _face_image = face_image
            _flow = flow

        lipreading_feature = self.net_lipreading(_mouthroi)          # (B,512,1,T)
        if _flow is not None:
            lipreading_feature = self.flow_fusion(lipreading_feature, _flow)

        identity_feature = self.net_facial(_face_image)
        identity_feature = identity_feature.repeat(1, 1, 1, lipreading_feature.shape[-1])
        visual_feature = torch.cat((identity_feature, lipreading_feature), dim=1)
        visual_feature = visual_feature.squeeze(2)                   # (B,640,T)

        return self.net_diffwave((melspec, diffusion_steps), cond=visual_feature)

    @torch.no_grad()
    def compute_cond(self, mouthroi, face_image, flow=None):
        """Return the 640-d conditioning tensor (pre-denoiser), no CFG dropout.
        Used by the step-0 invariance test to compare against the baseline."""
        lip = self.net_lipreading(mouthroi)
        if flow is not None:
            lip = self.flow_fusion(lip, flow)
        identity = self.net_facial(face_image).repeat(1, 1, 1, lip.shape[-1])
        return torch.cat((identity, lip), dim=1).squeeze(2)
