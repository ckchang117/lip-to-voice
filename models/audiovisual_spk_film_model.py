"""AudioVisualSpkFilmModel / AudioVisualSpkAttnFilmModel - Runs G and H.

G: plain LipVoicer + FiLM speaker/style + multi-layer FiLM injection into the frozen denoiser.
H: same as G + TemporalAttention (warm-started from Run B).

Both replace `SpeakerStyleFusion` (ReZero additive) with `SpeakerStyleFiLM` (per-channel
gamma scale + beta shift), and additionally apply a per-layer `LayerFiLM` via forward hooks on
every `Residual_block` of the frozen 12-layer WaveNet denoiser. Identity init at every
FiLM module guarantees step-0 invariance with the LipVoicer baseline.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .audiovisual_spk_model import AudioVisualSpkModel
from .speaker_style import VisualStyleEncoder
from .speaker_style_film import SpeakerStyleFiLM, LayerFiLMBank
from .temporal_attention import TemporalAttention


class AudioVisualSpkFilmModel(AudioVisualSpkModel):
    """Run G: LipVoicer + FiLM speaker/style + per-layer FiLM into the denoiser."""

    def name(self):
        return "AudioVisualSpkFilmModel"

    def __init__(self, nets,
                 speaker_style_fusion: SpeakerStyleFiLM | None = None,
                 visual_style_encoder: VisualStyleEncoder | None = None,
                 layer_film_bank: LayerFiLMBank | None = None,
                 face_seq_n: int = 8, sa_dim: int = 192,
                 num_denoiser_layers: int = 12, denoiser_channels: int = 512,
                 cond_dim: int = 384, film_hidden: int = 256):
        # Initialize parent with the FiLM fusion replacing SpeakerStyleFusion.
        super().__init__(
            nets,
            speaker_style_fusion=speaker_style_fusion or SpeakerStyleFiLM(),
            visual_style_encoder=visual_style_encoder or VisualStyleEncoder(),
            face_seq_n=face_seq_n, sa_dim=sa_dim,
        )
        self.layer_film_bank = layer_film_bank or LayerFiLMBank(
            num_layers=num_denoiser_layers, feat_channels=denoiser_channels,
            cond_dim=cond_dim, hidden=film_hidden,
        )

    def _attach_layer_film_hooks(self, cond: torch.Tensor):
        """Register a forward hook on each frozen Residual_block that applies the
        corresponding LayerFiLM to the block's `(out, skip)` residual output.
        Returns the list of handles for caller to remove after the forward pass.
        """
        handles = []
        blocks = self.net_diffwave.residual_layer.residual_blocks

        def make_hook(idx: int, c: torch.Tensor):
            film = self.layer_film_bank.films[idx]

            def hook(_module, _inp, output):
                out, skip = output
                return (film(out, c), skip)
            return hook

        for i, block in enumerate(blocks):
            handles.append(block.register_forward_hook(make_hook(i, cond)))
        return handles

    def forward(self, melspec, mouthroi, face_image, face_seq, audio_ref_emb,
                diffusion_steps, cond_drop_prob):
        batch = melspec.shape[0]
        if cond_drop_prob > 0:
            keep = self.prob_mask_like((batch, 1, 1, 1, 1), 1.0 - cond_drop_prob, melspec.device)
            _mouthroi = torch.where(keep, mouthroi, self.mouthroi_null)
            _face_image = torch.where(keep.squeeze(1), face_image, self.face_null)
            keep_seq = keep.squeeze(-1).squeeze(-1)
            _face_seq = torch.where(keep_seq.unsqueeze(-1).unsqueeze(-1), face_seq, self.face_seq_null)
            keep_emb = keep.view(batch, 1)
            _audio_ref_emb = torch.where(keep_emb, audio_ref_emb, self.audio_ref_null)
        else:
            _mouthroi = mouthroi
            _face_image = face_image
            _face_seq = face_seq
            _audio_ref_emb = audio_ref_emb

        lip = self.net_lipreading(_mouthroi)                       # (B, 512, 1, T)
        lip = self._maybe_apply_attention(lip)
        s_v = self._encode_visual_style(_face_seq)                 # (B, 192) L2-normalized
        s_a = _audio_ref_emb                                       # (B, 192)
        lip = self.speaker_style_fusion(lip, s_v, s_a)             # FiLM fusion (identity at init)

        # Joint speaker conditioning for the per-layer FiLM bank.
        sa_n = F.normalize(s_a, p=2, dim=-1)
        s = torch.cat([s_v, sa_n], dim=-1)                          # (B, 384)

        identity = self.net_facial(_face_image).repeat(1, 1, 1, lip.shape[-1])
        cond = torch.cat((identity, lip), dim=1).squeeze(2)         # (B, 640, T)

        handles = self._attach_layer_film_hooks(s)
        try:
            eps = self.net_diffwave((melspec, diffusion_steps), cond=cond)
        finally:
            for h in handles:
                h.remove()
        return eps

    @torch.no_grad()
    def compute_cond(self, mouthroi, face_image, face_seq, audio_ref_emb):
        """Pre-denoiser 640-d conditioning, no CFG. Used by the invariance test.
        Note: doesn't include the per-layer FiLM (those modulate inside the denoiser);
        the invariance test verifies them by running eps with hooks attached."""
        lip = self.net_lipreading(mouthroi)
        lip = self._maybe_apply_attention(lip)
        s_v = self._encode_visual_style(face_seq)
        lip = self.speaker_style_fusion(lip, s_v, audio_ref_emb)
        identity = self.net_facial(face_image).repeat(1, 1, 1, lip.shape[-1])
        return torch.cat((identity, lip), dim=1).squeeze(2)


class AudioVisualSpkAttnFilmModel(AudioVisualSpkFilmModel):
    """Run H: G + TemporalAttention (warm-started from Run B)."""

    def name(self):
        return "AudioVisualSpkAttnFilmModel"

    def __init__(self, nets,
                 speaker_style_fusion: SpeakerStyleFiLM | None = None,
                 visual_style_encoder: VisualStyleEncoder | None = None,
                 layer_film_bank: LayerFiLMBank | None = None,
                 temporal_attention: TemporalAttention | None = None,
                 face_seq_n: int = 8, sa_dim: int = 192,
                 num_denoiser_layers: int = 12, denoiser_channels: int = 512,
                 cond_dim: int = 384, film_hidden: int = 256):
        super().__init__(
            nets,
            speaker_style_fusion=speaker_style_fusion,
            visual_style_encoder=visual_style_encoder,
            layer_film_bank=layer_film_bank,
            face_seq_n=face_seq_n, sa_dim=sa_dim,
            num_denoiser_layers=num_denoiser_layers,
            denoiser_channels=denoiser_channels,
            cond_dim=cond_dim, film_hidden=film_hidden,
        )
        self.temporal_attention = temporal_attention or TemporalAttention()

    def _maybe_apply_attention(self, lip):
        return self.temporal_attention(lip)
