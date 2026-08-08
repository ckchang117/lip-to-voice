"""AudioVisualSpkModel / AudioVisualSpkAttnModel: speaker/style consistency variants.

Run E: plain LipVoicer + SpeakerStyleFusion (face-sequence visual style + ECAPA audio
       reference, contrastively aligned).
Run F: same + TemporalAttention (warm-started from Run B).

Both preserve LipVoicer's 640-d conditioning interface; new contributions enter via a
ReZero-gated additive stream on the 512-d lip features (alpha=0 at init -> bit-identical
to baseline at step 0).
"""

from __future__ import annotations

import torch

from .audiovisual_model import AudioVisualModel
from .speaker_style import SpeakerStyleFusion, VisualStyleEncoder
from .temporal_attention import TemporalAttention


class AudioVisualSpkModel(AudioVisualModel):
    """Run E: plain LipVoicer + speaker/style consistency."""

    def name(self):
        return "AudioVisualSpkModel"

    def __init__(self, nets,
                 speaker_style_fusion: SpeakerStyleFusion | None = None,
                 visual_style_encoder: VisualStyleEncoder | None = None,
                 face_seq_n: int = 8, sa_dim: int = 192):
        super().__init__(nets)
        self.speaker_style_fusion = speaker_style_fusion or SpeakerStyleFusion()
        self.visual_style_encoder = visual_style_encoder or VisualStyleEncoder()
        # Null tokens for classifier-free guidance dropout.
        self.register_buffer("face_seq_null", torch.zeros(1, face_seq_n, 3, 224, 224))
        self.register_buffer("audio_ref_null", torch.zeros(1, sa_dim))

    def _encode_visual_style(self, face_seq: torch.Tensor) -> torch.Tensor:
        """face_seq: (B, M, 3, 224, 224) -> (B, 192) visual style embedding."""
        B, M, C, H, W = face_seq.shape
        # Per-frame face features via the frozen net_facial (with_fc=True -> 128-d output).
        flat = face_seq.reshape(B * M, C, H, W)
        feats = self.net_facial(flat)                   # (B*M, 128, 1, 1)
        feats = feats.view(B, M, -1)                    # (B, M, 128)
        return self.visual_style_encoder(feats)         # (B, 192)

    def forward(self, melspec, mouthroi, face_image, face_seq, audio_ref_emb,
                diffusion_steps, cond_drop_prob):
        batch = melspec.shape[0]
        if cond_drop_prob > 0:
            # Mask drawn ONCE; all four streams drop together for coherent CFG.
            keep = self.prob_mask_like((batch, 1, 1, 1, 1), 1.0 - cond_drop_prob, melspec.device)
            _mouthroi = torch.where(keep, mouthroi, self.mouthroi_null)
            _face_image = torch.where(keep.squeeze(1), face_image, self.face_null)
            keep_seq = keep.squeeze(-1).squeeze(-1)                # (B,1,1)
            _face_seq = torch.where(keep_seq.unsqueeze(-1).unsqueeze(-1), face_seq, self.face_seq_null)
            keep_emb = keep.view(batch, 1)
            _audio_ref_emb = torch.where(keep_emb, audio_ref_emb, self.audio_ref_null)
        else:
            _mouthroi = mouthroi
            _face_image = face_image
            _face_seq = face_seq
            _audio_ref_emb = audio_ref_emb

        lip = self.net_lipreading(_mouthroi)                       # (B, 512, 1, T)
        lip = self._maybe_apply_attention(lip)                      # hook for subclass (F)
        s_v = self._encode_visual_style(_face_seq)                 # (B, 192)
        s_a = _audio_ref_emb                                       # (B, 192) ECAPA, frozen
        lip = self.speaker_style_fusion(lip, s_v, s_a)             # ReZero +0 at init

        identity = self.net_facial(_face_image).repeat(1, 1, 1, lip.shape[-1])
        cond = torch.cat((identity, lip), dim=1).squeeze(2)        # (B, 640, T)
        return self.net_diffwave((melspec, diffusion_steps), cond=cond)

    def _maybe_apply_attention(self, lip):
        # Run E: no attention. Run F overrides this.
        return lip

    @torch.no_grad()
    def compute_cond(self, mouthroi, face_image, face_seq, audio_ref_emb):
        """Pre-denoiser 640-d conditioning, no CFG. Used by the invariance test."""
        lip = self.net_lipreading(mouthroi)
        lip = self._maybe_apply_attention(lip)
        s_v = self._encode_visual_style(face_seq)
        lip = self.speaker_style_fusion(lip, s_v, audio_ref_emb)
        identity = self.net_facial(face_image).repeat(1, 1, 1, lip.shape[-1])
        return torch.cat((identity, lip), dim=1).squeeze(2)


class AudioVisualSpkAttnModel(AudioVisualSpkModel):
    """Run F: plain LipVoicer + temporal attention (warm-started from Run B) + speaker/style consistency."""

    def name(self):
        return "AudioVisualSpkAttnModel"

    def __init__(self, nets,
                 speaker_style_fusion: SpeakerStyleFusion | None = None,
                 visual_style_encoder: VisualStyleEncoder | None = None,
                 temporal_attention: TemporalAttention | None = None,
                 face_seq_n: int = 8, sa_dim: int = 192):
        super().__init__(nets, speaker_style_fusion=speaker_style_fusion,
                         visual_style_encoder=visual_style_encoder,
                         face_seq_n=face_seq_n, sa_dim=sa_dim)
        self.temporal_attention = temporal_attention or TemporalAttention()

    def _maybe_apply_attention(self, lip):
        return self.temporal_attention(lip)
