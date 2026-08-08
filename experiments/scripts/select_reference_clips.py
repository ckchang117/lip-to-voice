"""Score each TED clip for "speaker reference quality" and write a per-talk ranking.

Scoring (CPU-only, fast): combine three signals from the clip's 16 kHz wav:
  - voiced_ratio       - energy + zero-crossing-rate VAD (higher = more speech, less silence/applause/music)
  - speech_purity      - 1 − spectral_flatness over voiced frames (higher = more harmonic / voice-like)
  - duration_score     - normalized clip length (longer = more stable speaker info)
combined = 0.5*voiced + 0.3*speech_purity + 0.2*duration

Output: <raw_root>/reference_clips.json
  { "<talk_id>": [["<clip_id>", <score>], ...sorted desc...], ... }

At training/inference time, sample a reference clip from each talk's top-K (default 5),
excluding the target clip itself.
"""

from __future__ import annotations

import json
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm


def _voiced_ratio(audio: np.ndarray, sr: int = 16000, frame_ms: int = 25, hop_ms: int = 10,
                  energy_pct: float = 30.0, zcr_thresh: float = 0.20) -> float:
    """Fraction of frames with high energy AND low zero-crossing rate (voiced speech)."""
    n_frame = int(sr * frame_ms / 1000)
    n_hop = int(sr * hop_ms / 1000)
    if len(audio) < n_frame:
        return 0.0
    frames = np.lib.stride_tricks.sliding_window_view(audio, n_frame)[::n_hop]
    energy = (frames ** 2).mean(axis=1)
    zcr = (np.abs(np.diff(np.sign(frames), axis=1)) > 0).mean(axis=1)
    if energy.size == 0:
        return 0.0
    e_thresh = np.percentile(energy, energy_pct)  # clip-relative
    voiced = (energy > e_thresh) & (zcr < zcr_thresh)
    return float(voiced.mean())


def _speech_purity(audio: np.ndarray, sr: int = 16000, n_fft: int = 512) -> float:
    """1 − mean spectral flatness over the clip; voice (harmonic) has low flatness, noise/music high."""
    if len(audio) < n_fft:
        return 0.0
    n_frames = len(audio) // n_fft
    if n_frames == 0:
        return 0.0
    spec = np.abs(np.fft.rfft(audio[: n_frames * n_fft].reshape(n_frames, n_fft), axis=1)) + 1e-10
    # spectral flatness per frame = geom mean / arith mean
    log_mean = np.log(spec).mean(axis=1)
    arith_mean = spec.mean(axis=1)
    flatness = np.exp(log_mean) / np.maximum(arith_mean, 1e-10)
    return float(1.0 - flatness.mean())


def _duration_score(seconds: float, ref: float = 8.0) -> float:
    """Sigmoid-ish on length; longer than ~ref seconds is fully credited."""
    return float(1.0 / (1.0 + math.exp(-(seconds - ref / 2) / (ref / 4))))


def _score_clip(wav_path: Path) -> tuple[float, float, float, float, float]:
    """Return (combined, voiced, purity, duration_seconds, duration_score)."""
    try:
        audio, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
    except Exception:
        return (0.0, 0.0, 0.0, 0.0, 0.0)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    duration = len(audio) / max(sr, 1)
    voiced = _voiced_ratio(audio, sr=sr)
    purity = _speech_purity(audio, sr=sr)
    dur_s = _duration_score(duration)
    combined = 0.5 * voiced + 0.3 * purity + 0.2 * dur_s
    return (combined, voiced, purity, duration, dur_s)


def _score_one(args: tuple[str, str]) -> tuple[str, str, float]:
    """ProcessPool worker. Returns (talk_id, clip_id, combined_score)."""
    talk_id, wav_str = args
    combined, *_ = _score_clip(Path(wav_str))
    return (talk_id, Path(wav_str).stem, combined)


def run(audio_root: str, raw_root: str, top_k: int = 5, n_workers: int = 8) -> None:
    audio_main = Path(audio_root) / "main"
    if not audio_main.exists():
        raise SystemExit(f"missing: {audio_main}")
    out_path = Path(raw_root) / "reference_clips.json"

    talks = sorted(p for p in audio_main.iterdir() if p.is_dir())
    # Gather all (talk_id, wav_path) work items so we can scatter them across workers.
    work: list[tuple[str, str]] = []
    for talk_dir in talks:
        for wav in sorted(talk_dir.glob("*.wav")):
            work.append((talk_dir.name, str(wav)))
    print(f"scoring reference quality across {len(talks)} talks, {len(work)} clips ({n_workers} workers)")

    grouped: dict[str, list[tuple[str, float]]] = {t.name: [] for t in talks}
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        for talk_id, clip_id, score in tqdm(
            ex.map(_score_one, work, chunksize=8), total=len(work), desc="clips"
        ):
            grouped[talk_id].append((clip_id, score))

    out: dict[str, list[tuple[str, float]]] = {}
    for talk_id, scored in grouped.items():
        scored.sort(key=lambda x: -x[1])
        out[talk_id] = scored

    out_path.write_text(json.dumps(out, indent=2))
    n_clips = sum(len(v) for v in out.values())
    avg_topk = np.mean([
        np.mean([s for _, s in v[:top_k]]) if v else 0.0 for v in out.values()
    ]) if out else 0.0
    print(f"wrote {out_path}: {len(out)} talks, {n_clips} clips, mean top-{top_k} score = {avg_topk:.3f}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--audio_root", required=True)
    p.add_argument("--raw_root", required=True)
    p.add_argument("--top_k", type=int, default=5)
    args = p.parse_args()
    run(args.audio_root, args.raw_root, top_k=args.top_k)
