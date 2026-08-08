"""Fine-tune one ablation stream (Runs B/C/D) on the TED train split.

run_name selects the ablation:
  "attn" (Run B): temporal attention only; flow_fusion frozen at init and never
                  called (flow=None at every forward).
  "flow" (Run C): optical-flow stream only (FlowEncoder + zero-init proj + gate).
  "full" (Run D): flow + temporal attention; warm-starts flow_fusion from the
                  latest Run C checkpoint when available.

Freezes LipVoicer's pretrained denoiser, lip-reader, and face encoder; trains only
the selected modules. Loss = diffusion epsilon L1 + an annealed curriculum anchor
that keeps the new conditioning near baseline for the first ~2k steps so the frozen
denoiser stays in-distribution.

Standalone (no Hydra/DDP), single A100. Must run with CWD=/root/repo.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

MELGEN_CFG = dict(
    _name_="melgen", in_channels=80, out_channels=80,
    diffusion_step_embed_dim_in=128, diffusion_step_embed_dim_mid=512,
    diffusion_step_embed_dim_out=512, res_channels=512, skip_channels=512,
    num_res_layers=12, dilation_cycle=1, mel_upsample=[2, 2],
)

RUNS = ("attn", "flow", "full")


def _collate(batch):
    """Stack each field with torch.stack (fresh output storage), avoiding
    default_collate's worker shared-memory path (which raises 'Trying to resize
    storage that is not resizable' on numpy-backed tensors). Truncates the last
    dim to the batch-min when it varies: the mel window can differ by +/-1 frame
    across clips (fps rounding); mouthroi/face/flow are fixed-size.
    """
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


def _build_net(run_name: str, melgen_ckpt: str, flow_ckpt: str | None = None):
    from models.model_builder import ModelBuilder
    from models.audiovisual_flow_model import AudioVisualFlowModel
    from models.audiovisual_full_model import AudioVisualFullModel
    from models.flow_model import FlowFusion
    from models.temporal_attention import TemporalAttention
    from experiments.checkpoint_compat import load as compat_load

    builder = ModelBuilder()
    backbones = (builder.build_lipreadingnet(),
                 builder.build_facial(fc_out=128, with_fc=True),
                 builder.build_diffwave_model(OmegaConf.create(MELGEN_CFG)))

    if run_name == "flow":
        net = AudioVisualFlowModel(backbones, FlowFusion()).cuda()
        allowed_prefixes = ("flow_fusion.",)
    else:
        # "attn" hosts an unused flow_fusion so checkpoints stay schema-compatible.
        net = AudioVisualFullModel(
            backbones, flow_fusion=FlowFusion(), temporal_attention=TemporalAttention(),
        ).cuda()
        allowed_prefixes = ("flow_fusion.", "temporal_attention.")

    # "full" warm-starts flow_fusion from the Run C checkpoint when available;
    # everything else loads the 1M-step baseline.
    ckpt_path = melgen_ckpt
    if run_name == "full" and flow_ckpt and Path(flow_ckpt).exists():
        ckpt_path = flow_ckpt
    print(f"loading checkpoint: {ckpt_path}")
    state = compat_load(ckpt_path, map_location="cpu")
    sd = state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state
    missing, unexpected = net.load_state_dict(sd, strict=False)
    assert len(unexpected) == 0, f"unexpected ckpt keys (backbone prefix mismatch!): {unexpected[:8]}"
    bad = [k for k in missing if not (k.startswith(allowed_prefixes) or k == "flow_null")]
    assert not bad, f"unexpected MISSING keys (frozen weights not loaded!): {bad[:8]}"
    print(f"loaded ckpt strict=False | missing={len(missing)} (new modules only), unexpected=0")
    return net


def _trainable_modules(net, run_name: str):
    if run_name == "attn":
        return [net.temporal_attention]
    if run_name == "flow":
        return [net.flow_fusion]
    return [net.flow_fusion, net.temporal_attention]


def _freeze(net, run_name: str):
    trainable = _trainable_modules(net, run_name)
    for m in net.children():
        for p in m.parameters():
            p.requires_grad_(False)
        m.eval()  # freeze BatchNorm running stats + dropout
    for m in trainable:
        for p in m.parameters():
            p.requires_grad_(True)
        m.train()
    n = sum(p.numel() for m in trainable for p in m.parameters())
    print(f"trainable params ({run_name}): {n/1e6:.2f}M")


def _make_dataset(split, videos_dir, mouthrois_dir, audios_dir, flow_dir):
    from dataloaders.dataset_lipvoicer import LipVoicerDataset
    alias = "/tmp/LRS2_videos"
    if not os.path.lexists(alias):
        os.symlink(videos_dir, alias)  # LRS2 detection is case-sensitive on the path
    return LipVoicerDataset(
        split, videos_dir=alias, mouthrois_dir=mouthrois_dir, audios_dir=audios_dir,
        sampling_rate=16000, videos_window_size=25, audio_stft_hop=160, flow_dir=flow_dir,
    )


def _diffusion_inputs(melspec, dh):
    T, Alpha_bar = dh["T"], dh["Alpha_bar"]
    B = melspec.shape[0]
    steps = torch.randint(T, size=(B, 1, 1)).cuda()
    z = torch.normal(0, 1, size=melspec.shape).cuda()
    x_t = torch.sqrt(Alpha_bar[steps]) * melspec + torch.sqrt(1 - Alpha_bar[steps]) * z
    return x_t, steps, z


def _main_loss(net, l1, melspec, mouthroi, face, flow, dh, cond_drop_prob=0.2):
    B = melspec.shape[0]
    x_t, steps, z = _diffusion_inputs(melspec, dh)
    eps = net(x_t, mouthroi, face, steps.view(B, 1), cond_drop_prob, flow=flow)
    return l1(eps, z)


def _aux_loss(net, run_name: str, mouthroi, flow):
    """Curriculum anchor: squared norm of the new modules' perturbation of the
    lip features. Gradients flow only through the trainable modules."""
    if run_name == "flow":
        # ||c_new - c_baseline||^2 over the lip channels (face channels cancel).
        f = net.flow_fusion.proj(net.flow_fusion.encoder(flow))
        flow_term = torch.tanh(net.flow_fusion.gate) * f
        return (flow_term ** 2).mean()
    with torch.no_grad():
        lip0 = net.net_lipreading(mouthroi)
    if run_name == "attn":
        lip2 = net.temporal_attention(lip0)
    else:  # full: combined flow + attention perturbation
        lip2 = net.temporal_attention(net.flow_fusion(lip0, flow))
    return ((lip2 - lip0) ** 2).mean()


@torch.no_grad()
def _val_loss(net, run_name, l1, val_batch, dh, seed=1234):
    torch.manual_seed(seed)  # fixed noise/steps so the proxy is comparable across evals
    melspec, mouthroi, face, flow = (t.cuda() for t in val_batch)
    if run_name == "attn":
        flow = None
    return _main_loss(net, l1, melspec, mouthroi, face, flow, dh, cond_drop_prob=0.0).item()


def _log_line(net, run_name, step, main, aux, lam):
    parts = [f"step {step}: main {main.item():.4f} aux {aux.item():.5f} lam {lam:.3f}"]
    if run_name in ("flow", "full"):
        parts.append(f"tanh(gate) {float(torch.tanh(net.flow_fusion.gate).item()):+.4f}")
    if run_name in ("attn", "full"):
        alphas = [float(b.alpha.item()) for b in net.temporal_attention.blocks]
        parts.append(f"alpha0 {alphas[0]:+.4f} alphaN {alphas[-1]:+.4f}")
    tqdm.write(" ".join(parts))


def run(run_name, melgen_ckpt, videos_dir, mouthrois_dir, audios_dir, flow_dir, save_dir,
        flow_ckpt: str | None = None,
        steps=6000, batch=16, lr=2e-4, lam0=1.0, anneal=2000,
        log_every=50, val_every=500, patience=3):
    assert run_name in RUNS, f"run_name must be one of {RUNS}"
    _setup_repo()
    from utils import calc_diffusion_hyperparams

    net = _build_net(run_name, melgen_ckpt, flow_ckpt=flow_ckpt)
    _freeze(net, run_name)
    dh = calc_diffusion_hyperparams(T=400, beta_0=0.0001, beta_T=0.02, beta=None, fast=False)

    train_ds = _make_dataset("train", videos_dir, mouthrois_dir, audios_dir, flow_dir)
    val_ds = _make_dataset("val", videos_dir, mouthrois_dir, audios_dir, flow_dir)
    print(f"train clips: {len(train_ds)} | val clips: {len(val_ds)}")
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=4,
                              drop_last=True, pin_memory=False, collate_fn=_collate)
    val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False, num_workers=2,
                            drop_last=True, collate_fn=_collate)
    val_batch = next(iter(val_loader))  # one fixed batch for the comparable proxy

    trainable = [p for m in _trainable_modules(net, run_name) for p in m.parameters()]
    optimizer = torch.optim.Adam(trainable, lr=lr)
    l1 = nn.L1Loss()
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    def _save(step):
        torch.save({"model_state_dict": net.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict()},
                   str(save_path / f"{step}.pkl"))

    step = 0
    best_val = math.inf
    bad_evals = 0
    print(f"=== training {run_name}: {steps} steps, batch {batch}, lr {lr} ===", flush=True)
    pbar = tqdm(total=steps, desc=f"train_{run_name}")
    stop = False
    while step < steps and not stop:
        for melspec, mouthroi, face, flow in train_loader:
            melspec, mouthroi, face, flow = melspec.cuda(), mouthroi.cuda(), face.cuda(), flow.cuda()
            fwd_flow = None if run_name == "attn" else flow
            optimizer.zero_grad()
            main = _main_loss(net, l1, melspec, mouthroi, face, fwd_flow, dh)
            aux = _aux_loss(net, run_name, mouthroi, flow)
            lam = max(0.0, lam0 * (1.0 - step / anneal))
            (main + lam * aux).backward()
            optimizer.step()

            if step % log_every == 0:
                _log_line(net, run_name, step, main, aux, lam)

            if step > 0 and step % val_every == 0:
                v = _val_loss(net, run_name, l1, val_batch, dh)
                tqdm.write(f"  [val] step {step}: val_loss {v:.4f} (best {best_val:.4f})")
                _save(step)
                if v < best_val - 1e-4:
                    best_val = v
                    bad_evals = 0
                else:
                    bad_evals += 1
                    if bad_evals >= patience:
                        tqdm.write(f"  early stop at step {step} (no val improvement x{patience})")
                        stop = True
                        break
            step += 1
            pbar.update(1)
            if step >= steps:
                break
    pbar.close()

    _save(step)
    print(f"\ndone. final step {step}, best val {best_val:.4f}. ckpts in {save_dir}", flush=True)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--run_name", required=True, choices=list(RUNS))
    for a in ("melgen_ckpt", "videos_dir", "mouthrois_dir", "audios_dir", "flow_dir", "save_dir"):
        p.add_argument(f"--{a}", required=True)
    p.add_argument("--flow_ckpt", default=None)
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    args = p.parse_args()
    run(args.run_name, args.melgen_ckpt, args.videos_dir, args.mouthrois_dir, args.audios_dir,
        args.flow_dir, args.save_dir, flow_ckpt=args.flow_ckpt,
        steps=args.steps, batch=args.batch, lr=args.lr)
