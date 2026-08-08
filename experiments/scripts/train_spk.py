"""Fine-tune speaker/style consistency variants (Runs E + F).

variant=spk      -> Run E: plain LipVoicer + SpeakerStyleFusion + VisualStyleEncoder
variant=spk_attn -> Run F: LipVoicer + TemporalAttention (warm from Run B) + SpeakerStyleFusion + VisualStyleEncoder

Loss = diffusion epsilon L1 + λ_contrastive * InfoNCE(visual_style_emb, audio_ref_emb)
where positives are the per-example (s_v_i, s_a_i) pair (same speaker), and negatives
are the other in-batch examples (random different talks at batch 16 / 280-talk pool ->
likely different speakers). ReZero alpha=0 in SpeakerStyleFusion provides step-0
invariance to baseline conditioning. Must run with CWD=/root/repo.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

MELGEN_CFG = dict(
    _name_="melgen", in_channels=80, out_channels=80,
    diffusion_step_embed_dim_in=128, diffusion_step_embed_dim_mid=512,
    diffusion_step_embed_dim_out=512, res_channels=512, skip_channels=512,
    num_res_layers=12, dilation_cycle=1, mel_upsample=[2, 2],
)


def _collate(batch):
    """Truncate-to-min on last dim (handles mel +/-1 frame); same as the flow/full trainer."""
    out = []
    for i in range(len(batch[0])):
        tensors = [b[i] for b in batch]
        min_last = min(t.shape[-1] for t in tensors)
        if any(t.shape[-1] != min_last for t in tensors):
            tensors = [t[..., :min_last] for t in tensors]
        out.append(torch.stack(tensors, 0))
    return tuple(out)


def _setup_repo():
    if "/root/repo" not in sys.path:
        sys.path.insert(0, "/root/repo")
    os.chdir("/root/repo")
    from experiments.scripts.run_baseline_inference import _link_checkpoints
    _link_checkpoints()


def _build_net(variant: str, melgen_ckpt: str, attn_ckpt: str | None = None):
    """variant in {'spk','spk_attn'}.
    spk      -> AudioVisualSpkModel; load 1M ckpt strict=False (fresh spk modules).
    spk_attn -> AudioVisualSpkAttnModel; load Run B attn ckpt strict=False (warm-start attention).
    """
    from models.model_builder import ModelBuilder
    from models.audiovisual_spk_model import AudioVisualSpkModel, AudioVisualSpkAttnModel
    from models.speaker_style import SpeakerStyleFusion, VisualStyleEncoder
    from models.temporal_attention import TemporalAttention
    from experiments.checkpoint_compat import load as compat_load

    builder = ModelBuilder()
    net_lip = builder.build_lipreadingnet()
    net_face = builder.build_facial(fc_out=128, with_fc=True)
    net_dw = builder.build_diffwave_model(OmegaConf.create(MELGEN_CFG))

    if variant == "spk":
        net = AudioVisualSpkModel((net_lip, net_face, net_dw),
                                   speaker_style_fusion=SpeakerStyleFusion(),
                                   visual_style_encoder=VisualStyleEncoder()).cuda()
        ckpt_path = melgen_ckpt
    elif variant == "spk_attn":
        net = AudioVisualSpkAttnModel((net_lip, net_face, net_dw),
                                       speaker_style_fusion=SpeakerStyleFusion(),
                                       visual_style_encoder=VisualStyleEncoder(),
                                       temporal_attention=TemporalAttention()).cuda()
        ckpt_path = attn_ckpt if (attn_ckpt and Path(attn_ckpt).exists()) else melgen_ckpt
    else:
        raise ValueError(f"unknown variant: {variant}")

    print(f"loading checkpoint: {ckpt_path}")
    state = compat_load(ckpt_path, map_location="cpu")
    sd = state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state
    missing, unexpected = net.load_state_dict(sd, strict=False)
    # Drop optimizer-only keys / things that don't apply (Run B attn ckpt has flow_fusion.* that
    # we don't have here; AudioVisualFullModel keys we don't carry forward).
    allowed_unexpected = ("flow_fusion.", "flow_null")  # benign - Run B ckpt has these
    bad_unexpected = [k for k in unexpected if not k.startswith(allowed_unexpected)]
    assert not bad_unexpected, f"unexpected ckpt keys: {bad_unexpected[:8]}"
    if unexpected:
        print(f"  ignored unexpected keys: {len(unexpected)} (flow_fusion from Run B ckpt, benign)")
    allowed_prefixes = ("speaker_style_fusion.", "visual_style_encoder.")
    allowed_singletons = {"face_seq_null", "audio_ref_null"}
    if variant == "spk_attn":
        if ckpt_path == melgen_ckpt:
            allowed_prefixes = allowed_prefixes + ("temporal_attention.",)
    bad_missing = [k for k in missing if not (k.startswith(allowed_prefixes) or k in allowed_singletons)]
    assert not bad_missing, f"unexpected MISSING keys: {bad_missing[:8]}"
    print(f"loaded {ckpt_path} strict=False | missing={len(missing)} (spk/style + buffers"
          f"{' + attention' if variant=='spk_attn' and ckpt_path==melgen_ckpt else ''})")
    return net


def _freeze_backbone(net, variant: str):
    # Freeze everything except the new modules (and attention for spk if not present).
    frozen = [net.net_lipreading, net.net_facial, net.net_diffwave]
    if variant == "spk_attn":
        # We warm-started attention from Run B; freeze it now so we isolate the speaker/style gain.
        frozen.append(net.temporal_attention)
    for m in frozen:
        for p in m.parameters():
            p.requires_grad_(False)
        m.eval()
    # Trainable: visual_style_encoder + speaker_style_fusion
    for p in net.visual_style_encoder.parameters():
        p.requires_grad_(True)
    for p in net.speaker_style_fusion.parameters():
        p.requires_grad_(True)
    net.visual_style_encoder.train()
    net.speaker_style_fusion.train()
    n_vse = sum(p.numel() for p in net.visual_style_encoder.parameters())
    n_ssf = sum(p.numel() for p in net.speaker_style_fusion.parameters())
    print(f"trainable params: visual_style_encoder={n_vse/1e6:.2f}M  "
          f"speaker_style_fusion={n_ssf/1e6:.2f}M  total={(n_vse+n_ssf)/1e6:.2f}M")


def _make_dataset(split, videos_dir, mouthrois_dir, audios_dir, flow_dir,
                  speaker_ref_dir, ref_clips_json, face_seq_n=8):
    from dataloaders.dataset_lipvoicer import LipVoicerDataset
    alias = "/tmp/LRS2_videos"
    if not os.path.lexists(alias):
        os.symlink(videos_dir, alias)
    return LipVoicerDataset(
        split, videos_dir=alias, mouthrois_dir=mouthrois_dir, audios_dir=audios_dir,
        sampling_rate=16000, videos_window_size=25, audio_stft_hop=160,
        flow_dir=flow_dir, speaker_ref_dir=speaker_ref_dir,
        ref_clips_json=ref_clips_json, face_seq_n=face_seq_n,
    )


def _diffusion_inputs(melspec, dh):
    T, Alpha_bar = dh["T"], dh["Alpha_bar"]
    B = melspec.shape[0]
    steps = torch.randint(T, size=(B, 1, 1)).cuda()
    z = torch.normal(0, 1, size=melspec.shape).cuda()
    x_t = torch.sqrt(Alpha_bar[steps]) * melspec + torch.sqrt(1 - Alpha_bar[steps]) * z
    return x_t, steps, z


def _main_loss(net, l1, melspec, mouthroi, face, face_seq, audio_ref_emb, dh, cond_drop_prob=0.2):
    B = melspec.shape[0]
    x_t, steps, z = _diffusion_inputs(melspec, dh)
    eps = net(x_t, mouthroi, face, face_seq, audio_ref_emb, steps.view(B, 1), cond_drop_prob)
    return l1(eps, z)


def _info_nce(s_v: torch.Tensor, s_a: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """Symmetric InfoNCE / CLIP loss. Both inputs L2-normalized (B, D)."""
    s_a = F.normalize(s_a, p=2, dim=-1)
    # logits (B, B): row i, col j = sim(s_v_i, s_a_j) / τ
    logits = s_v @ s_a.t() / temperature
    targets = torch.arange(s_v.size(0), device=s_v.device)
    loss_v2a = F.cross_entropy(logits, targets)
    loss_a2v = F.cross_entropy(logits.t(), targets)
    return 0.5 * (loss_v2a + loss_a2v)


@torch.no_grad()
def _val_loss(net, l1, val_batch, dh, seed=1234):
    torch.manual_seed(seed)
    melspec, mouthroi, face, _flow, face_seq, audio_ref_emb = (t.cuda() for t in val_batch)
    return _main_loss(net, l1, melspec, mouthroi, face, face_seq, audio_ref_emb, dh,
                      cond_drop_prob=0.0).item()


def run(variant: str, melgen_ckpt, attn_ckpt, videos_dir, mouthrois_dir, audios_dir, flow_dir,
        speaker_ref_dir, ref_clips_json, save_dir,
        steps=6000, batch=16, lr=2e-4, lam_contrastive=0.5, temperature=0.07,
        log_every=50, val_every=500, patience=3):
    _setup_repo()
    from utils import calc_diffusion_hyperparams

    net = _build_net(variant, melgen_ckpt, attn_ckpt=attn_ckpt)
    _freeze_backbone(net, variant)
    dh = calc_diffusion_hyperparams(T=400, beta_0=0.0001, beta_T=0.02, beta=None, fast=False)

    train_ds = _make_dataset("train", videos_dir, mouthrois_dir, audios_dir, flow_dir,
                              speaker_ref_dir, ref_clips_json)
    val_ds = _make_dataset("val", videos_dir, mouthrois_dir, audios_dir, flow_dir,
                            speaker_ref_dir, ref_clips_json)
    print(f"train clips: {len(train_ds)} | val clips: {len(val_ds)}")
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=4,
                              drop_last=True, pin_memory=False, collate_fn=_collate)
    val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False, num_workers=2,
                            drop_last=True, collate_fn=_collate)
    val_batch = next(iter(val_loader))

    trainable = list(net.visual_style_encoder.parameters()) + list(net.speaker_style_fusion.parameters())
    optimizer = torch.optim.Adam(trainable, lr=lr)
    l1 = nn.L1Loss()
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Cumulative step counter so retries after preemption monotonically advance saved
    # checkpoint filenames (rather than overwriting low numbers and getting stuck in a loop).
    # Parse the resumed checkpoint's step from its filename; fall back to 0 for cold starts.
    start_step = 0
    if attn_ckpt and Path(attn_ckpt).exists() and Path(attn_ckpt).parent == Path(save_dir):
        try:
            start_step = int(Path(attn_ckpt).stem)
            print(f"resuming step counter at {start_step}", flush=True)
        except ValueError:
            start_step = 0
    step = 0  # local step (this run); global = start_step + step
    best_val = math.inf
    bad_evals = 0
    print(f"=== training Run {variant.upper()}: {steps} steps, batch {batch}, lr {lr}, "
          f"lam_contrastive {lam_contrastive}, tau {temperature}, start_step {start_step} ===", flush=True)
    remaining = max(0, steps - start_step)  # steps is global target
    if remaining == 0:
        print(f"start_step {start_step} >= target {steps}; nothing to do.", flush=True)
        return
    pbar = tqdm(total=remaining, desc=f"train_{variant}")
    stop = False
    while step < remaining and not stop:
        for melspec, mouthroi, face, _flow, face_seq, audio_ref_emb in train_loader:
            melspec = melspec.cuda(); mouthroi = mouthroi.cuda(); face = face.cuda()
            face_seq = face_seq.cuda(); audio_ref_emb = audio_ref_emb.cuda()

            optimizer.zero_grad()
            main = _main_loss(net, l1, melspec, mouthroi, face, face_seq, audio_ref_emb, dh)
            # Contrastive: compute s_v on the same face_seq the main loss used; s_a is the audio ref.
            s_v = net._encode_visual_style(face_seq)
            contrastive = _info_nce(s_v, audio_ref_emb, temperature=temperature)
            (main + lam_contrastive * contrastive).backward()
            optimizer.step()

            if step % log_every == 0:
                alpha = float(torch.tanh(net.speaker_style_fusion.alpha).item())
                tqdm.write(f"step {step}: main {main.item():.4f} ctr {contrastive.item():.4f} "
                           f"tanh(alpha_spk) {alpha:+.4f}")

            if step > 0 and step % val_every == 0:
                v = _val_loss(net, l1, val_batch, dh)
                global_step = start_step + step
                tqdm.write(f"  [val] step {step} (global {global_step}): val_loss {v:.4f} (best {best_val:.4f})")
                torch.save({"model_state_dict": net.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict()},
                           str(save_path / f"{global_step}.pkl"))
                if v < best_val - 1e-4:
                    best_val = v
                    bad_evals = 0
                else:
                    bad_evals += 1
                    if bad_evals >= patience:
                        tqdm.write(f"  early stop at step {step}")
                        stop = True
                        break
            step += 1
            pbar.update(1)
            if step >= remaining:
                break
    pbar.close()

    final_global = start_step + step
    torch.save({"model_state_dict": net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict()},
               str(save_path / f"{final_global}.pkl"))
    print(f"\ndone. final step {step} (global {final_global}), best val {best_val:.4f}. ckpts in {save_dir}", flush=True)
