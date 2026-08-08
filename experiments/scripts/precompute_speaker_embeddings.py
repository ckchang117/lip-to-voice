"""Precompute ECAPA-TDNN speaker embeddings for the top-K clean clips per talk.

Loads `speechbrain/spkrec-ecapa-voxceleb` (192-d speaker embedding, frozen). For each
talk in reference_clips.json, takes its top-K clips by quality score, reads their
16 kHz wavs, runs them through ECAPA, and saves the 192-d embedding as a .npy at:

  <speaker_emb_root>/main/<talk_id>/<clip_id>.npy

Idempotent (skip if file exists). Designed to run in a dedicated Modal image that
includes speechbrain (the main LIPVOICER_IMAGE doesn't need it - only the .npy
embeddings are read at training/inference time).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torchaudio
from tqdm import tqdm

_ECAPA = None


def _ecapa(device: str = "cuda:0"):
    global _ECAPA
    if _ECAPA is None:
        from speechbrain.inference.speaker import EncoderClassifier
        print("loading ECAPA-TDNN (speechbrain/spkrec-ecapa-voxceleb)...")
        _ECAPA = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": device},
        )
        _ECAPA.eval()
        print("ECAPA loaded (192-d embedding, frozen)")
    return _ECAPA


@torch.no_grad()
def _embed_wav(wav_path: Path, device: str, max_seconds: float = 10.0) -> np.ndarray:
    """Read wav, resample to 16 kHz mono if needed, trim to max_seconds, run ECAPA -> 192-d."""
    audio, sr = torchaudio.load(str(wav_path))
    if audio.dim() == 2 and audio.size(0) > 1:
        audio = audio.mean(dim=0, keepdim=True)
    if sr != 16000:
        audio = torchaudio.functional.resample(audio, sr, 16000)
        sr = 16000
    if audio.size(-1) > int(max_seconds * sr):
        audio = audio[..., : int(max_seconds * sr)]
    audio = audio.to(device)
    emb = _ecapa(device).encode_batch(audio)              # (1, 1, 192)
    return emb.squeeze().detach().cpu().numpy().astype(np.float32)


def run(audio_root: str, raw_root: str, embed_root: str, top_k: int = 5,
        device: str = "cuda:0") -> None:
    ref_json = Path(raw_root) / "reference_clips.json"
    if not ref_json.exists():
        raise SystemExit(f"missing: {ref_json} - run select_reference_clips first")
    refs: dict[str, list] = json.loads(ref_json.read_text())

    audio_main = Path(audio_root) / "main"
    emb_main = Path(embed_root) / "main"
    emb_main.mkdir(parents=True, exist_ok=True)

    # Targets: top-K clean clips per talk
    targets: list[tuple[str, str]] = []
    for talk_id, scored in refs.items():
        for clip_id, _score in scored[:top_k]:
            targets.append((talk_id, clip_id))
    print(f"precomputing ECAPA embeddings for {len(targets)} clips ({len(refs)} talks × top-{top_k})")

    n_ok = 0
    n_skip = 0
    n_fail = 0
    for talk_id, clip_id in tqdm(targets, desc="ECAPA"):
        out_path = emb_main / talk_id / f"{clip_id}.npy"
        if out_path.exists() and out_path.stat().st_size > 0:
            n_skip += 1
            continue
        wav = audio_main / talk_id / f"{clip_id}.wav"
        if not wav.exists():
            n_fail += 1
            tqdm.write(f"  miss {wav}")
            continue
        try:
            emb = _embed_wav(wav, device=device)
        except Exception as e:  # noqa: BLE001
            n_fail += 1
            tqdm.write(f"  FAIL {talk_id}/{clip_id}: {type(e).__name__}: {e}")
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(out_path), emb)
        n_ok += 1

    print(f"\ndone. ok={n_ok}  skip={n_skip}  fail={n_fail}  -> {emb_main}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--audio_root", required=True)
    p.add_argument("--raw_root", required=True)
    p.add_argument("--embed_root", required=True)
    p.add_argument("--top_k", type=int, default=5)
    args = p.parse_args()
    run(args.audio_root, args.raw_root, args.embed_root, top_k=args.top_k)
