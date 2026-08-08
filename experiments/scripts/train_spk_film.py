"""Fine-tune FiLM speaker/style + multi-layer FiLM injection variants (Runs G + H).

variant=g       -> Run G: LipVoicer + SpeakerStyleFiLM + LayerFiLMBank (multi-layer denoiser FiLM)
variant=h       -> Run H: G + TemporalAttention (warm-started from Run B)

Loss = diffusion epsilon L1 + λ_contrastive * InfoNCE(visual_style_emb, audio_ref_emb).
Identity init at every FiLM module -> frozen denoiser sees baseline conditioning bit-for-bit
at step 0.

Smarter stop criterion: tracks EMAs of BOTH val_loss and contrastive loss; early-stops only
when both EMAs plateau for `patience` consecutive val checkpoints. Fixes the Run F shipping
issue where val_loss plateaued early but contrastive kept dropping.

Cumulative-step checkpoint numbering (same fix as train_spk.py post-Run F): filenames use
`global_step = start_step + local_step` so preemption-resume monotonically advances progress.
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
    """variant in {'g','h'}.
    g -> AudioVisualSpkFilmModel; load melgen ckpt strict=False (fresh film modules).
    h -> AudioVisualSpkAttnFilmModel; load Run B attn ckpt strict=False (warm-start attn)
        OR resume from a prior G/H run if its checkpoint exists.
    """
    from models.model_builder import ModelBuilder
    from models.audiovisual_spk_film_model import (
        AudioVisualSpkFilmModel, AudioVisualSpkAttnFilmModel,
    )
    from models.speaker_style import VisualStyleEncoder
    from models.speaker_style_film import SpeakerStyleFiLM, LayerFiLMBank
    from models.temporal_attention import TemporalAttention
    from experiments.checkpoint_compat import load as compat_load

    builder = ModelBuilder()
    net_lip = builder.build_lipreadingnet()
    net_face = builder.build_facial(fc_out=128, with_fc=True)
    net_dw = builder.build_diffwave_model(OmegaConf.create(MELGEN_CFG))

    if variant == "g":
        net = AudioVisualSpkFilmModel(
            (net_lip, net_face, net_dw),
            speaker_style_fusion=SpeakerStyleFiLM(),
            visual_style_encoder=VisualStyleEncoder(),
            layer_film_bank=LayerFiLMBank(),
        ).cuda()
        ckpt_path = melgen_ckpt
    elif variant == "h":
        net = AudioVisualSpkAttnFilmModel(
            (net_lip, net_face, net_dw),
            speaker_style_fusion=SpeakerStyleFiLM(),
            visual_style_encoder=VisualStyleEncoder(),
            layer_film_bank=LayerFiLMBank(),
            temporal_attention=TemporalAttention(),
        ).cuda()
        ckpt_path = attn_ckpt if (attn_ckpt and Path(attn_ckpt).exists()) else melgen_ckpt
    else:
        raise ValueError(f"unknown variant: {variant}")

    print(f"loading checkpoint: {ckpt_path}")
    state = compat_load(ckpt_path, map_location="cpu")
    sd = state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state
    missing, unexpected = net.load_state_dict(sd, strict=False)
    # Drop optimizer-only / non-applicable keys (Run B attn ckpt may have flow_fusion.*).
    allowed_unexpected = ("flow_fusion.", "flow_null")
    bad_unexpected = [k for k in unexpected if not k.startswith(allowed_unexpected)]
    assert not bad_unexpected, f"unexpected ckpt keys: {bad_unexpected[:8]}"
    if unexpected:
        print(f"  ignored unexpected keys: {len(unexpected)} (legacy flow_fusion from Run B ckpt)")
    allowed_prefixes = ("speaker_style_fusion.", "visual_style_encoder.", "layer_film_bank.")
    allowed_singletons = {"face_seq_null", "audio_ref_null"}
    if variant == "h" and ckpt_path == melgen_ckpt:
        allowed_prefixes = allowed_prefixes + ("temporal_attention.",)
    bad_missing = [k for k in missing if not (k.startswith(allowed_prefixes) or k in allowed_singletons)]
    assert not bad_missing, f"unexpected MISSING keys: {bad_missing[:8]}"
    print(f"loaded {ckpt_path} strict=False | missing={len(missing)} (FiLM + speaker + buffers"
          f"{' + attention' if variant=='h' and ckpt_path==melgen_ckpt else ''})")
    return net


def _freeze_backbone(net, variant: str):
    """Freeze everything except FiLM + visual style modules (and attention for H if warm-started)."""
    frozen = [net.net_lipreading, net.net_facial, net.net_diffwave]
    if variant == "h":
        frozen.append(net.temporal_attention)  # warm-started from Run B; isolate FiLM gain
    for m in frozen:
        for p in m.parameters():
            p.requires_grad_(False)
        m.eval()
    for p in net.visual_style_encoder.parameters():
        p.requires_grad_(True)
    for p in net.speaker_style_fusion.parameters():
        p.requires_grad_(True)
    for p in net.layer_film_bank.parameters():
        p.requires_grad_(True)
    net.visual_style_encoder.train()
    net.speaker_style_fusion.train()
    net.layer_film_bank.train()
    n_vse = sum(p.numel() for p in net.visual_style_encoder.parameters())
    n_ssf = sum(p.numel() for p in net.speaker_style_fusion.parameters())
    n_lfb = sum(p.numel() for p in net.layer_film_bank.parameters())
    print(f"trainable params: visual_style_encoder={n_vse/1e6:.2f}M  "
          f"speaker_style_fusion(FiLM)={n_ssf/1e6:.2f}M  "
          f"layer_film_bank={n_lfb/1e6:.2f}M  "
          f"total={(n_vse+n_ssf+n_lfb)/1e6:.2f}M")


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
    """Symmetric InfoNCE / CLIP loss. s_v already L2-normalized; we normalize s_a here."""
    s_a = F.normalize(s_a, p=2, dim=-1)
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
        log_every=50, val_every=200, patience=8, ema_alpha=0.3, ema_eps=1e-3):
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

    trainable = (
        list(net.visual_style_encoder.parameters())
        + list(net.speaker_style_fusion.parameters())
        + list(net.layer_film_bank.parameters())
    )
    optimizer = torch.optim.Adam(trainable, lr=lr)
    l1 = nn.L1Loss()
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Cumulative-step counter so retries after preemption monotonically advance.
    start_step = 0
    if attn_ckpt and Path(attn_ckpt).exists() and Path(attn_ckpt).parent == Path(save_dir):
        try:
            start_step = int(Path(attn_ckpt).stem)
            print(f"resuming step counter at {start_step}", flush=True)
        except ValueError:
            start_step = 0
    step = 0
    val_ema = None
    ctr_ema = None
    best_val_ema = math.inf
    best_ctr_ema = math.inf
    bad = 0
    print(f"=== training Run {variant.upper()}: target {steps} steps, batch {batch}, lr {lr}, "
          f"lam_contrastive {lam_contrastive}, tau {temperature}, start_step {start_step}, "
          f"val_every {val_every}, patience {patience}, ema_alpha {ema_alpha} ===", flush=True)
    remaining = max(0, steps - start_step)
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
            s_v = net._encode_visual_style(face_seq)
            contrastive = _info_nce(s_v, audio_ref_emb, temperature=temperature)
            (main + lam_contrastive * contrastive).backward()
            optimizer.step()

            if step % log_every == 0:
                tqdm.write(f"step {step}: main {main.item():.4f} ctr {contrastive.item():.4f}")

            if step > 0 and step % val_every == 0:
                v = _val_loss(net, l1, val_batch, dh)
                c = float(contrastive.item())
                val_ema = v if val_ema is None else ema_alpha * v + (1 - ema_alpha) * val_ema
                ctr_ema = c if ctr_ema is None else ema_alpha * c + (1 - ema_alpha) * ctr_ema
                global_step = start_step + step
                tqdm.write(
                    f"  [val] step {step} (global {global_step}): val_loss {v:.4f} "
                    f"(EMA {val_ema:.4f}, best {best_val_ema:.4f})  "
                    f"ctr {c:.4f} (EMA {ctr_ema:.4f}, best {best_ctr_ema:.4f})  bad={bad}"
                )
                torch.save({"model_state_dict": net.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict()},
                           str(save_path / f"{global_step}.pkl"))
                v_improved = val_ema < best_val_ema - ema_eps
                c_improved = ctr_ema < best_ctr_ema - ema_eps
                if v_improved:
                    best_val_ema = val_ema
                if c_improved:
                    best_ctr_ema = ctr_ema
                if v_improved or c_improved:
                    bad = 0
                else:
                    bad += 1
                    if bad >= patience:
                        tqdm.write(f"  early stop at step {step} (both val + ctr EMAs plateaued)")
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
    print(f"\ndone. final step {step} (global {final_global}), "
          f"best val_ema {best_val_ema:.4f}, best ctr_ema {best_ctr_ema:.4f}. "
          f"ckpts in {save_dir}", flush=True)
