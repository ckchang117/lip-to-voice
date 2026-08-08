"""Step-0 invariance gate for every fine-tuned variant.

variant selects the model under test:
  "flow"     (Run C): zero-gate FlowFusion
  "full"     (Run D): FlowFusion + ReZero temporal attention
  "spk"      (Run E): ReZero speaker/style fusion
  "spk_attn" (Run F): speaker/style + temporal attention
  "g"        (Run G): FiLM speaker/style + per-layer FiLM bank
  "h"        (Run H): G + temporal attention

Each new module is identity-initialized (ReZero scalars at 0; FiLM last-linears
predict gamma=1, beta=0), so at training step 0 the model must reproduce the
baseline exactly. Checks, in order:
1. Symbolic: the identity-init parameters are actually at their init values.
2. Deterministic primary: the pre-denoiser lip chain equals the raw lip features
   bit-for-bit (single pass, no GPU non-determinism).
3. End-to-end sanity: eps prediction matches the baseline within the measured
   baseline-vs-baseline non-determinism floor (two instances of a deep net on GPU
   are only equal up to cuDNN non-determinism).
"""

from __future__ import annotations

import os
import sys

import torch
from omegaconf import OmegaConf

MELGEN_CFG = dict(
    _name_="melgen", in_channels=80, out_channels=80,
    diffusion_step_embed_dim_in=128, diffusion_step_embed_dim_mid=512,
    diffusion_step_embed_dim_out=512, res_channels=512, skip_channels=512,
    num_res_layers=12, dilation_cycle=1, mel_upsample=[2, 2],
)

FLOW_VARIANTS = ("flow", "full")
SPK_VARIANTS = ("spk", "spk_attn", "g", "h")
VARIANTS = FLOW_VARIANTS + SPK_VARIANTS


def _setup_repo():
    if "/root/repo" not in sys.path:
        sys.path.insert(0, "/root/repo")
    os.chdir("/root/repo")
    from experiments.scripts.run_baseline_inference import _link_checkpoints
    _link_checkpoints()


def _build_backbones():
    from models.model_builder import ModelBuilder
    b = ModelBuilder()
    return b.build_lipreadingnet(), b.build_facial(fc_out=128, with_fc=True), \
        b.build_diffwave_model(OmegaConf.create(MELGEN_CFG))


def _build_variant(variant: str):
    from models.audiovisual_flow_model import AudioVisualFlowModel
    from models.audiovisual_full_model import AudioVisualFullModel
    from models.audiovisual_spk_model import AudioVisualSpkModel, AudioVisualSpkAttnModel
    from models.audiovisual_spk_film_model import (
        AudioVisualSpkFilmModel, AudioVisualSpkAttnFilmModel,
    )
    from models.flow_model import FlowFusion
    from models.speaker_style import SpeakerStyleFusion, VisualStyleEncoder
    from models.speaker_style_film import SpeakerStyleFiLM, LayerFiLMBank
    from models.temporal_attention import TemporalAttention

    backbones = _build_backbones()
    if variant == "flow":
        return AudioVisualFlowModel(backbones, FlowFusion())
    if variant == "full":
        return AudioVisualFullModel(backbones, flow_fusion=FlowFusion(),
                                    temporal_attention=TemporalAttention())
    if variant == "spk":
        return AudioVisualSpkModel(backbones, speaker_style_fusion=SpeakerStyleFusion(),
                                   visual_style_encoder=VisualStyleEncoder())
    if variant == "spk_attn":
        return AudioVisualSpkAttnModel(backbones, speaker_style_fusion=SpeakerStyleFusion(),
                                       visual_style_encoder=VisualStyleEncoder(),
                                       temporal_attention=TemporalAttention())
    if variant == "g":
        return AudioVisualSpkFilmModel(backbones, speaker_style_fusion=SpeakerStyleFiLM(),
                                       visual_style_encoder=VisualStyleEncoder(),
                                       layer_film_bank=LayerFiLMBank())
    if variant == "h":
        return AudioVisualSpkAttnFilmModel(backbones, speaker_style_fusion=SpeakerStyleFiLM(),
                                           visual_style_encoder=VisualStyleEncoder(),
                                           layer_film_bank=LayerFiLMBank(),
                                           temporal_attention=TemporalAttention())
    raise ValueError(f"unknown variant: {variant}")


def _assert_film_identity_init(module, name: str):
    """A SpeakerStyleFiLM or LayerFiLM must have gamma->1, beta->0 at init."""
    g_last = module.gamma_mlp[-1] if hasattr(module, "gamma_mlp") else module.gamma_proj[-1]
    b_last = module.beta_mlp[-1] if hasattr(module, "beta_mlp") else module.beta_proj[-1]
    assert torch.all(g_last.weight == 0), f"{name}: gamma last-linear weight not zero"
    assert torch.all(g_last.bias == 1), f"{name}: gamma last-linear bias not ones"
    assert torch.all(b_last.weight == 0), f"{name}: beta last-linear weight not zero"
    assert torch.all(b_last.bias == 0), f"{name}: beta last-linear bias not zero"


def _assert_init(net, variant: str):
    if variant in ("flow", "full"):
        gate = float(torch.tanh(net.flow_fusion.gate).item())
        print(f"tanh(gate)={gate:.6f}")
        assert gate == 0.0, "flow gate must be zero-initialized!"
    if variant in ("spk", "spk_attn"):
        alpha_spk = float(torch.tanh(net.speaker_style_fusion.alpha).item())
        print(f"tanh(alpha_spk)={alpha_spk:.6f}")
        assert alpha_spk == 0.0, "speaker_style_fusion alpha must be zero-initialized!"
    if variant in ("g", "h"):
        _assert_film_identity_init(net.speaker_style_fusion, "SpeakerStyleFiLM")
        for i, film in enumerate(net.layer_film_bank.films):
            _assert_film_identity_init(film, f"LayerFiLM[{i}]")
    if variant in ("full", "spk_attn", "h"):
        alphas = [float(b.alpha.item()) for b in net.temporal_attention.blocks]
        betas = [float(b.beta.item()) for b in net.temporal_attention.blocks]
        assert all(a == 0.0 for a in alphas), "every attention alpha must be zero-initialized!"
        assert all(b == 0.0 for b in betas), "every attention beta must be zero-initialized!"
    print(f"variant={variant}  identity-init checks PASS")


def run(variant: str, melgen_ckpt, videos_dir, mouthrois_dir, audios_dir, flow_dir,
        speaker_ref_dir=None, ref_clips_json=None):
    assert variant in VARIANTS, f"variant must be one of {VARIANTS}"
    _setup_repo()
    from models.audiovisual_model import AudioVisualModel
    from experiments.checkpoint_compat import load as compat_load
    from dataloaders.dataset_lipvoicer import LipVoicerDataset

    sd = compat_load(melgen_ckpt, map_location="cpu")
    sd = sd["model_state_dict"] if isinstance(sd, dict) and "model_state_dict" in sd else sd

    base = AudioVisualModel(_build_backbones()).cuda().eval()
    base.load_state_dict(sd)

    net = _build_variant(variant).cuda().eval()
    missing, unexpected = net.load_state_dict(sd, strict=False)
    assert unexpected == [], f"unexpected keys: {unexpected[:8]}"

    _assert_init(net, variant)

    alias = "/tmp/LRS2_videos"
    if not os.path.lexists(alias):
        os.symlink(videos_dir, alias)
    ds_kwargs = dict(videos_dir=alias, mouthrois_dir=mouthrois_dir, audios_dir=audios_dir,
                     sampling_rate=16000, videos_window_size=25, audio_stft_hop=160,
                     flow_dir=flow_dir)
    if variant in SPK_VARIANTS:
        ds_kwargs.update(speaker_ref_dir=speaker_ref_dir, ref_clips_json=ref_clips_json)
    ds = LipVoicerDataset("val", **ds_kwargs)

    if variant in SPK_VARIANTS:
        melspec, mouthroi, face, flow, face_seq, audio_ref_emb = ds[0]
        face_seq = face_seq.unsqueeze(0).cuda()
        audio_ref_emb = audio_ref_emb.unsqueeze(0).cuda()
    else:
        melspec, mouthroi, face, flow = ds[0]
    melspec = melspec.unsqueeze(0).cuda()
    mouthroi = mouthroi.unsqueeze(0).cuda()
    face = face.unsqueeze(0).cuda()
    flow = flow.unsqueeze(0).cuda()
    steps = torch.zeros((1, 1)).cuda()

    with torch.no_grad():
        # PRIMARY (deterministic): the pre-denoiser chain must be an identity on
        # the raw lip features.
        lip = net.net_lipreading(mouthroi)
        if variant == "flow":
            chain = net.flow_fusion(lip, flow)
        elif variant == "full":
            chain = net.temporal_attention(net.flow_fusion(lip, flow))
        else:
            chain = lip.clone()
            if variant in ("spk_attn", "h"):
                chain = net.temporal_attention(chain)
            s_v = net._encode_visual_style(face_seq)
            chain = net.speaker_style_fusion(chain, s_v, audio_ref_emb)
        dchain = (chain - lip).abs().max().item()
        print(f"max|chain - lip| = {dchain:.3e}")
        if variant in ("g", "h"):
            # FiLM identity init: exact zero modulo float noise.
            assert dchain <= 1e-5, f"pre-denoiser chain leaks at init: {dchain:.3e}"
        else:
            assert torch.equal(chain, lip), f"chain LEAKS at init: max|delta|={dchain:.3e}"

        # SANITY: end-to-end eps against the baseline self-non-determinism floor.
        eps_base = base(melspec, mouthroi, face, steps, cond_drop_prob=0)
        eps_base2 = base(melspec, mouthroi, face, steps, cond_drop_prob=0)
        if variant in SPK_VARIANTS:
            eps_new = net(melspec, mouthroi, face, face_seq, audio_ref_emb, steps, cond_drop_prob=0)
        else:
            eps_new = net(melspec, mouthroi, face, steps, cond_drop_prob=0, flow=flow)

    floor = (eps_base - eps_base2).abs().max().item()
    deps = (eps_base - eps_new).abs().max().item()
    print(f"max|eps_base - eps_new| = {deps:.3e}   (baseline self-nondeterminism floor = {floor:.3e})")
    assert deps <= max(floor * 5.0, 1e-2), f"eps differs beyond non-determinism floor: {deps:.3e} vs floor {floor:.3e}"
    print(f"PASS [{variant}]: new modules compose to identity at init; conditioning matches baseline.")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--variant", required=True, choices=list(VARIANTS))
    for a in ("melgen_ckpt", "videos_dir", "mouthrois_dir", "audios_dir", "flow_dir"):
        p.add_argument(f"--{a}", required=True)
    p.add_argument("--speaker_ref_dir", default=None)
    p.add_argument("--ref_clips_json", default=None)
    args = p.parse_args()
    run(args.variant, args.melgen_ckpt, args.videos_dir, args.mouthrois_dir, args.audios_dir,
        args.flow_dir, speaker_ref_dir=args.speaker_ref_dir, ref_clips_json=args.ref_clips_json)
