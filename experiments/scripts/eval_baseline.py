"""Evaluate generated speech against ground truth for LipVoicer on LRS2.

Designed to take an arbitrary directory of generated .wav files and produce
WER + STOI + (LSE-C/LSE-D once SyncNet is wired in) on a fixed eval split.
The same harness will be used for our ablations later.

The harness is built around a per-clip metrics dict so individual metrics can
be NaN if a particular evaluator isn't configured yet (e.g. SyncNet on first
pass) without poisoning the aggregate.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Iterable, Optional

# Heavy imports are deferred to inside `run()` so the module can be imported
# on the local Mac for static checks without dragging in torch/transformers.


# --------------------------------------------------------------------------
# Eval-split + audio-file discovery
# --------------------------------------------------------------------------
def _load_split(eval_split_path: str) -> list[str]:
    ids = Path(eval_split_path).read_text().splitlines()
    return [x.strip() for x in ids if x.strip()]


def _index_baseline_audio(root: str) -> dict[str, Path]:
    """LipVoicer's released LRS2 audio zip has an unknown internal layout.

    We index every .wav under `root` by both basename (e.g. "00001") and by
    the relative path *minus* its top-level dir (e.g. "<set>/00001"), so we
    can look up clips regardless of nesting.
    """
    index: dict[str, Path] = {}
    for wav in Path(root).rglob("*.wav"):
        rel = wav.relative_to(root)
        # Drop the very first path part (often "LRS2" or similar wrapper dir).
        parts = rel.parts
        if len(parts) > 1:
            keyed = "/".join(parts[1:]).removesuffix(".wav")
            index[keyed] = wav
        # Always also index by the "<parent_dir>/<basename>" since LipVoicer
        # uses that as the video_id convention.
        if len(parts) >= 2:
            two_level = f"{parts[-2]}/{parts[-1].removesuffix('.wav')}"
            index[two_level] = wav
        # And by basename alone as a last resort.
        index.setdefault(rel.stem, wav)
    return index


def _read_transcript(txt_path: Path) -> str:
    """LRS2 .txt files start with 'Text:  <SENTENCE>\\n'."""
    raw = txt_path.read_text().splitlines()[0]
    if raw.startswith("Text:"):
        raw = raw[5:].lstrip()
    return raw.lower().strip()


# --------------------------------------------------------------------------
# Metric evaluators (each returns NaN if not configured / not applicable)
# --------------------------------------------------------------------------
@dataclass
class ClipMetrics:
    wer_whisper: float = math.nan
    wer_wav2vec2: float = math.nan
    stoi: float = math.nan
    estoi: float = math.nan
    lse_c: float = math.nan  # SyncNet integration left as future work
    lse_d: float = math.nan
    # DNSMOS - Microsoft's no-reference MOS estimator. The paper reports OVRL.
    dnsmos_p808: float = math.nan
    dnsmos_sig: float = math.nan
    dnsmos_bak: float = math.nan
    dnsmos_ovrl: float = math.nan
    # UTMOS (Saeki et al. 2022) - closest off-the-shelf substitute for STOI-Net.
    # No-reference MOS estimator focused on naturalness + intelligibility, 1-5 scale.
    utmos: float = math.nan

    def as_dict(self) -> dict[str, float]:
        return {
            "wer_whisper": self.wer_whisper,
            "wer_wav2vec2": self.wer_wav2vec2,
            "stoi": self.stoi,
            "estoi": self.estoi,
            "lse_c": self.lse_c,
            "lse_d": self.lse_d,
            "dnsmos_p808": self.dnsmos_p808,
            "dnsmos_sig": self.dnsmos_sig,
            "dnsmos_bak": self.dnsmos_bak,
            "dnsmos_ovrl": self.dnsmos_ovrl,
            "utmos": self.utmos,
        }


@dataclass
class Evaluators:
    whisper_pipeline: object | None = None
    wav2vec2_pipeline: object | None = None
    dnsmos_estimator: object | None = None  # speechmos.dnsmos.run
    utmos_estimator: object | None = None  # speechmos.utmos.run
    device: str = "cuda"

    def score_utmos(self, audio_path: Path) -> float:
        """No-reference MOS estimator (Saeki et al. 2022). Closest off-the-shelf
        substitute for STOI-Net. Returns 1-5 score, NaN on failure."""
        if self.utmos_estimator is None:
            return math.nan
        try:
            import soundfile as sf
            import numpy as np
            audio, sr = sf.read(str(audio_path))
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            # UTMOS expects 16 kHz mono
            if sr != 16000:
                import librosa
                audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=16000)
            result = self.utmos_estimator(audio.astype(np.float32), 16000)
            for k in ("utmos", "UTMOS", "mos"):
                if k in result:
                    return float(result[k])
            return math.nan
        except Exception as e:
            print(f"  [utmos] {audio_path.name}: {type(e).__name__}: {e}")
            return math.nan

    def score_dnsmos(self, audio_path: Path) -> tuple[float, float, float, float]:
        """Return (p808_mos, sig_mos, bak_mos, ovrl_mos). All NaN on failure.

        No-reference: only feeds the generated audio. Wraps Microsoft's
        official DNSMOS ONNX model via the `speechmos` package.
        Paper reports OVRL.
        """
        if self.dnsmos_estimator is None:
            return math.nan, math.nan, math.nan, math.nan
        try:
            import soundfile as sf
            import numpy as np
            audio, sr = sf.read(str(audio_path))
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != 16000:
                import librosa
                audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=16000)
            # speechmos.dnsmos.run signature: run(audio: np.ndarray, sr: int)
            # Returns dict with both UPPERCASE (OVRL, SIG, BAK, P808_MOS) and
            # lowercase-suffixed (ovrl_mos, sig_mos, bak_mos, p808_mos) keys
            # depending on version. Be defensive.
            result = self.dnsmos_estimator(audio.astype(np.float32), 16000)
            def _g(*keys):
                for k in keys:
                    if k in result:
                        return float(result[k])
                return math.nan
            return (
                _g("p808_mos", "P808_MOS"),
                _g("sig_mos", "mos_sig", "SIG"),
                _g("bak_mos", "mos_bak", "BAK"),
                _g("ovrl_mos", "mos_ovr", "OVRL"),
            )
        except Exception as e:
            print(f"  [dnsmos] {audio_path.name}: {type(e).__name__}: {e}")
            return math.nan, math.nan, math.nan, math.nan

    def score_wer(self, audio_path: Path, reference: str) -> tuple[float, float]:
        import jiwer

        out_whisper, out_wav2vec2 = math.nan, math.nan
        if self.whisper_pipeline is not None:
            try:
                pred = self.whisper_pipeline(str(audio_path))["text"].lower().strip()
                out_whisper = float(jiwer.wer(reference, pred))
            except Exception as e:
                print(f"  [whisper] {audio_path.name}: {type(e).__name__}: {e}")
        if self.wav2vec2_pipeline is not None:
            try:
                pred = self.wav2vec2_pipeline(str(audio_path))["text"].lower().strip()
                out_wav2vec2 = float(jiwer.wer(reference, pred))
            except Exception as e:
                print(f"  [wav2vec2] {audio_path.name}: {type(e).__name__}: {e}")
        return out_whisper, out_wav2vec2

    @staticmethod
    def score_stoi(generated_wav: Path, reference_video: Path) -> tuple[float, float]:
        """Extract GT audio from the LRS2 video on the fly, then STOI vs generated.

        Returns (STOI, ESTOI). Both NaN on failure.
        """
        import soundfile as sf
        import librosa
        import numpy as np
        from pystoi import stoi as _stoi

        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                gt_wav_path = f.name
            subprocess.run(
                [
                    "ffmpeg", "-loglevel", "error", "-y", "-i", str(reference_video),
                    "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", gt_wav_path,
                ],
                check=True,
            )
            gt_audio, gt_sr = sf.read(gt_wav_path)
            gen_audio, gen_sr = sf.read(str(generated_wav))
            os.unlink(gt_wav_path)

            # Mono
            if gt_audio.ndim > 1:
                gt_audio = gt_audio.mean(axis=1)
            if gen_audio.ndim > 1:
                gen_audio = gen_audio.mean(axis=1)
            # Resample if mismatched
            target_sr = 16000
            if gt_sr != target_sr:
                gt_audio = librosa.resample(gt_audio.astype(np.float32), orig_sr=gt_sr, target_sr=target_sr)
            if gen_sr != target_sr:
                gen_audio = librosa.resample(gen_audio.astype(np.float32), orig_sr=gen_sr, target_sr=target_sr)
            # Trim to shortest
            n = min(len(gt_audio), len(gen_audio))
            if n < target_sr // 4:  # under 0.25s is not meaningful
                return math.nan, math.nan
            gt_audio = gt_audio[:n]
            gen_audio = gen_audio[:n]
            s = float(_stoi(gt_audio, gen_audio, target_sr, extended=False))
            es = float(_stoi(gt_audio, gen_audio, target_sr, extended=True))
            return s, es
        except Exception as e:
            print(f"  [stoi] {generated_wav.name}: {type(e).__name__}: {e}")
            return math.nan, math.nan


def _build_evaluators(device: str) -> Evaluators:
    """Lazily build the Whisper + wav2vec2 pipelines.

    Loading happens once per `run()`. Each pipeline ~3 GB VRAM at fp16.
    """
    import torch
    from transformers import pipeline

    print(f"[setup] loading Whisper-large-v3 on {device}")
    whisper = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-large-v3",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device=device,
    )
    print(f"[setup] loading wav2vec2-large-960h on {device}")
    wav2vec2 = pipeline(
        "automatic-speech-recognition",
        model="facebook/wav2vec2-large-960h",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device=device,
    )

    dnsmos = None
    try:
        print(f"[setup] loading DNSMOS estimator (speechmos)")
        from speechmos import dnsmos as _dnsmos_mod
        dnsmos = _dnsmos_mod.run
        print(f"[setup] DNSMOS loaded via speechmos.dnsmos.run")
    except Exception as e:
        print(f"[setup] DNSMOS unavailable ({type(e).__name__}: {e}); skipping")

    utmos = None
    try:
        print(f"[setup] loading UTMOS estimator (speechmos)")
        from speechmos import utmos as _utmos_mod
        utmos = _utmos_mod.run
        print(f"[setup] UTMOS loaded via speechmos.utmos.run")
    except Exception as e:
        print(f"[setup] UTMOS unavailable ({type(e).__name__}: {e}); skipping")

    return Evaluators(
        whisper_pipeline=whisper,
        wav2vec2_pipeline=wav2vec2,
        dnsmos_estimator=dnsmos,
        utmos_estimator=utmos,
        device=device,
    )


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------
def _aggregate(per_clip: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    """Return mean / count / coverage for every metric across clips."""
    keys = set()
    for d in per_clip.values():
        keys.update(d.keys())
    out: dict[str, dict[str, float]] = {}
    for k in sorted(keys):
        vals = [d[k] for d in per_clip.values() if not math.isnan(d.get(k, math.nan))]
        if not vals:
            out[k] = {"mean": math.nan, "count": 0, "coverage": 0.0}
            continue
        out[k] = {
            "mean": float(mean(vals)),
            "count": len(vals),
            "coverage": len(vals) / max(1, len(per_clip)),
        }
    return out


# --------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------
def run(
    eval_split: str,
    baseline_audio_root: str,
    videos_root: str,
    out_path: str,
    device: str | None = None,
    max_clips: int | None = None,
) -> dict:
    """Score every clip in `eval_split` against the audio at `baseline_audio_root`.

    Output JSON has shape:
        {"per_clip": {<id>: {<metric>: <value>}}, "aggregate": {<metric>: {...}}}
    """
    if device is None:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"

    eval_ids = _load_split(eval_split)
    if max_clips is not None:
        eval_ids = eval_ids[:max_clips]
    print(f"[run] {len(eval_ids)} clips, device={device}")

    audio_index = _index_baseline_audio(baseline_audio_root)
    print(f"[run] indexed {len(audio_index)} baseline .wav files under {baseline_audio_root}")

    videos_root_p = Path(videos_root)
    evaluators = _build_evaluators(device)

    per_clip: dict[str, dict[str, float]] = {}
    missing = 0

    for i, vid in enumerate(eval_ids):
        if i % 20 == 0:
            print(f"[run] {i}/{len(eval_ids)}  ({vid})")
        m = ClipMetrics()

        # Locate audio
        wav = audio_index.get(vid) or audio_index.get(vid.split("/")[-1])
        if wav is None:
            missing += 1
            print(f"  [missing] no baseline audio found for id={vid!r}")
            per_clip[vid] = m.as_dict()
            continue

        # Locate transcript + video
        txt = videos_root_p / f"{vid}.txt"
        mp4 = videos_root_p / f"{vid}.mp4"
        if not txt.exists():
            print(f"  [missing] transcript missing: {txt}")
            per_clip[vid] = m.as_dict()
            continue
        reference = _read_transcript(txt)

        m.wer_whisper, m.wer_wav2vec2 = evaluators.score_wer(wav, reference)
        if mp4.exists():
            m.stoi, m.estoi = evaluators.score_stoi(wav, mp4)
        m.dnsmos_p808, m.dnsmos_sig, m.dnsmos_bak, m.dnsmos_ovrl = evaluators.score_dnsmos(wav)
        m.utmos = evaluators.score_utmos(wav)

        per_clip[vid] = m.as_dict()

    aggregate = _aggregate(per_clip)
    payload = {
        "eval_split": eval_split,
        "baseline_audio_root": baseline_audio_root,
        "videos_root": videos_root,
        "n_total": len(eval_ids),
        "n_missing_audio": missing,
        "per_clip": per_clip,
        "aggregate": aggregate,
    }

    out_path_p = Path(out_path)
    out_path_p.parent.mkdir(parents=True, exist_ok=True)
    out_path_p.write_text(json.dumps(payload, indent=2))

    print()
    print("=" * 60)
    print(f"Aggregate metrics (n={len(eval_ids)}, missing audio={missing})")
    print("=" * 60)
    for k, v in aggregate.items():
        print(f"  {k:15s}  mean={v['mean']:.4f}   coverage={v['coverage']:.0%}")
    print(f"saved -> {out_path_p}")

    return payload


if __name__ == "__main__":
    # Allow direct invocation for local debugging on a tiny split.
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--eval-split", required=True)
    p.add_argument("--baseline-audio-root", required=True)
    p.add_argument("--videos-root", required=True)
    p.add_argument("--out-path", required=True)
    p.add_argument("--device", default=None)
    p.add_argument("--max-clips", type=int, default=None)
    args = p.parse_args()
    run(
        eval_split=args.eval_split,
        baseline_audio_root=args.baseline_audio_root,
        videos_root=args.videos_root,
        out_path=args.out_path,
        device=args.device,
        max_clips=args.max_clips,
    )
