"""ECAPA-TDNN speaker similarity evaluation.

For each generated clip in `gen_audio_root`, compute ECAPA embedding of the generated
audio AND a held-out clean clip of the same speaker (one not used as the generation
reference, drawn from the speaker's clean-clip pool). Report mean cosine similarity per
run - higher means the generated audio sounds more like the actual speaker.

Saves a JSON with per-clip similarities and aggregate mean.
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
        print("loading ECAPA-TDNN for speaker similarity eval...")
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
    """Return precomputed ECAPA embedding (192-d) for the clean clip if available."""
    p = emb_root / "main" / talk / f"{clip}.npy"
    if not p.exists():
        return None
    arr = np.load(str(p))
    return torch.from_numpy(arr).float()


def run(eval_split_path: str, gen_audio_root: str, ref_clips_json: str,
        speaker_emb_root: str, out_path: str, top_k: int = 5, device: str = "cuda:0"):
    eval_ids = [x.strip().removeprefix("main/") for x in Path(eval_split_path).read_text().splitlines() if x.strip()]
    refs = json.loads(Path(ref_clips_json).read_text())
    gen_root = Path(gen_audio_root)
    emb_root = Path(speaker_emb_root)

    per_clip: dict[str, dict] = {}
    sims: list[float] = []
    n_miss = 0
    for vid in tqdm(eval_ids, desc="ecapa sim"):
        talk, clip = vid.split("/")
        gen_wav = gen_root / talk / f"{clip}.wav"
        if not gen_wav.exists():
            n_miss += 1
            continue

        # Held-out reference: pick a top-K clean clip of this speaker that is NOT the target itself.
        pool = [cid for cid, _ in refs.get(talk, [])[:top_k] if cid != clip]
        if not pool:
            n_miss += 1
            continue
        ref_clip = pool[0]
        ref_emb = _load_precomputed(emb_root, talk, ref_clip)
        if ref_emb is None:
            n_miss += 1
            continue
        ref_emb = ref_emb.to(device)

        gen_emb = _embed_wav(gen_wav, device=device)
        sim = float(F.cosine_similarity(gen_emb.unsqueeze(0), ref_emb.unsqueeze(0), dim=-1).item())
        per_clip[vid] = {"sim": sim, "ref_clip": ref_clip}
        sims.append(sim)

    if sims:
        agg = {"mean": float(np.mean(sims)), "n": len(sims), "std": float(np.std(sims))}
    else:
        agg = {"mean": float("nan"), "n": 0, "std": float("nan")}
    payload = {"per_clip": per_clip, "aggregate": agg, "missing": n_miss}
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"saved -> {out}  | mean speaker similarity = {agg['mean']:.4f}  (n={agg['n']}, missing={n_miss})")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    for a in ("eval_split_path", "gen_audio_root", "ref_clips_json", "speaker_emb_root", "out_path"):
        p.add_argument(f"--{a}", required=True)
    args = p.parse_args()
    run(args.eval_split_path, args.gen_audio_root, args.ref_clips_json,
        args.speaker_emb_root, args.out_path)
