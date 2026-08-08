"""Cross-speaker swap test - SCORING PHASE.

Reads the metadata JSON written by swap_test.py (generation phase), runs ECAPA-TDNN
on every swap.wav + nonswap.wav, and computes:
  - sim(gen_swap, B's reference)    - should be HIGH if speaker conditioning works
  - sim(gen_swap, A's reference)    - should be LOW (visible speaker, not heard)
  - sim(gen_nonswap, A's reference) - calibration ceiling (A's video + A's ref)

This script runs in SPEAKERSIM_IMAGE (torch 2.2 + torchaudio 2.2 + speechbrain 1.0)
where speechbrain's torchaudio.io.StreamReader requirement is satisfied.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torchaudio
import torch.nn.functional as F
from tqdm import tqdm

_ECAPA = None


def _ecapa(device: str = "cuda:0"):
    global _ECAPA
    if _ECAPA is None:
        from speechbrain.inference.speaker import EncoderClassifier
        print("loading ECAPA-TDNN…")
        _ECAPA = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": device},
        )
        _ECAPA.eval()
    return _ECAPA


@torch.no_grad()
def _embed_wav(wav_path: Path, device: str = "cuda:0", max_seconds: float = 10.0) -> torch.Tensor:
    audio, sr = torchaudio.load(str(wav_path))
    if audio.dim() == 2 and audio.size(0) > 1:
        audio = audio.mean(dim=0, keepdim=True)
    if sr != 16000:
        audio = torchaudio.functional.resample(audio, sr, 16000)
        sr = 16000
    if audio.size(-1) > int(max_seconds * sr):
        audio = audio[..., : int(max_seconds * sr)]
    audio = audio.to(device)
    emb = _ecapa(device).encode_batch(audio)
    return emb.squeeze().detach()


def _load_precomputed(emb_root: Path, talk: str, clip: str) -> torch.Tensor | None:
    p = emb_root / "main" / talk / f"{clip}.npy"
    if not p.exists():
        return None
    return torch.from_numpy(np.load(str(p))).float()


def run(meta_json_path: str, speaker_emb_root: str, out_json_path: str,
        device: str = "cuda:0"):
    meta = json.loads(Path(meta_json_path).read_text())
    variant = meta.get("variant", "unknown")
    per_pair_in = meta["per_pair"]
    emb_root = Path(speaker_emb_root)

    per_pair = {}
    sims_to_B, sims_to_A, sims_nonswap = [], [], []

    for vid_A, info in tqdm(per_pair_in.items(), desc="score"):
        talk_A = info["talk_A"]
        talk_B = info["talk_B"]
        ref_A_clip = info["ref_A_clip"]
        ref_B_clip = info["ref_B_clip"]
        ref_A = _load_precomputed(emb_root, talk_A, ref_A_clip)
        ref_B = _load_precomputed(emb_root, talk_B, ref_B_clip)
        if ref_A is None or ref_B is None:
            continue
        ref_A = ref_A.to(device); ref_B = ref_B.to(device)

        swap_wav = Path(info["swap_wav"])
        if not swap_wav.is_absolute():
            swap_wav = Path(meta["audio_root"]) / swap_wav.name
        ns_wav = Path(info["nonswap_wav"])
        if not ns_wav.is_absolute():
            ns_wav = Path(meta["audio_root"]) / ns_wav.name

        if not swap_wav.exists() or not ns_wav.exists():
            continue

        gen_swap = _embed_wav(swap_wav, device=device)
        gen_ns = _embed_wav(ns_wav, device=device)
        sim_B = float(F.cosine_similarity(gen_swap.unsqueeze(0), ref_B.unsqueeze(0), dim=-1).item())
        sim_A = float(F.cosine_similarity(gen_swap.unsqueeze(0), ref_A.unsqueeze(0), dim=-1).item())
        sim_ns = float(F.cosine_similarity(gen_ns.unsqueeze(0), ref_A.unsqueeze(0), dim=-1).item())
        sims_to_B.append(sim_B); sims_to_A.append(sim_A); sims_nonswap.append(sim_ns)
        per_pair[vid_A] = {"talk_B": talk_B, "sim_to_B": sim_B, "sim_to_A": sim_A, "sim_nonswap_AtoA": sim_ns}

    agg = {
        "mean_sim_to_B": float(np.mean(sims_to_B)) if sims_to_B else float("nan"),
        "mean_sim_to_A": float(np.mean(sims_to_A)) if sims_to_A else float("nan"),
        "mean_sim_nonswap_AtoA": float(np.mean(sims_nonswap)) if sims_nonswap else float("nan"),
        "n_pairs": len(per_pair),
    }
    payload = {"variant": variant, "aggregate": agg, "per_pair": per_pair}
    out = Path(out_json_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"saved -> {out}")
    print(f"  mean sim(gen_swap, target B)  = {agg['mean_sim_to_B']:.4f}   (higher = good)")
    print(f"  mean sim(gen_swap, visible A) = {agg['mean_sim_to_A']:.4f}   (lower  = good)")
    print(f"  mean sim(gen_nonswap, A)      = {agg['mean_sim_nonswap_AtoA']:.4f}   (ceiling)")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--meta_json_path", required=True)
    p.add_argument("--speaker_emb_root", required=True)
    p.add_argument("--out_json_path", required=True)
    args = p.parse_args()
    run(args.meta_json_path, args.speaker_emb_root, args.out_json_path)
