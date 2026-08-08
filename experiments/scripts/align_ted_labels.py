"""Produce TED-transcript labels for the eval clips, correctly aligned.

Background: TED's GraphQL cue *timestamps* are offset from the downloaded MP4
timeline (~3 s intro lead-in), so segment_ted cut each clip at a window whose
audio doesn't match the cue text it was labeled with. The TED *text* is correct;
only the time alignment is wrong.

This re-derives each eval clip's label from the TED transcript: we anchor on the
clip's actual content (a quick ASR pass over the clip audio), then find the
contiguous span of the TED transcript that best matches that content and use
**TED's own words** as the label. ASR is used only to locate the span - the
emitted label text is TED's, not the ASR's. Falls back to the ASR text only if
no TED span matches well (rare; flagged).

Only the eval clips need this (WER ground truth); training uses no text.
"""

from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from pathlib import Path

from tqdm import tqdm

_ASR = None


def _whisper():
    global _ASR
    if _ASR is None:
        import torch
        from transformers import pipeline
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        print(f"loading Whisper-large-v3 on {device} for ALIGNMENT ONLY...")
        _ASR = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3",
                        torch_dtype=dtype, device=device, return_timestamps=False)
    return _ASR


def _norm(text: str) -> str:
    """Uppercase, keep A-Z and apostrophes, collapse whitespace (LRS2 label form)."""
    text = re.sub(r"[^A-Z' ]+", " ", text.upper())
    return re.sub(r"\s+", " ", text).strip()


def _ted_words(transcript_path: Path) -> list[str]:
    cues = json.loads(transcript_path.read_text()).get("cues") or []
    text = " ".join(_norm(c.get("text") or "") for c in cues)
    return [w for w in text.split() if w]


def _best_span(hyp_words: list[str], ted_words: list[str]) -> tuple[str, float]:
    """Find the TED word-window best matching the ASR hypothesis; return its TED text."""
    n = len(hyp_words)
    if n == 0 or not ted_words:
        return "", 0.0
    best_score, best_lo, best_hi = 0.0, 0, min(n, len(ted_words))
    # Slide a window roughly the length of the hypothesis (+/- slack).
    for lo in range(len(ted_words)):
        for length in range(max(1, n - 4), n + 5):
            hi = lo + length
            if hi > len(ted_words):
                break
            score = SequenceMatcher(None, hyp_words, ted_words[lo:hi]).ratio()
            if score > best_score:
                best_score, best_lo, best_hi = score, lo, hi
        if lo > 0 and lo % 1 == 0 and best_score > 0.95:
            break  # near-perfect; stop early
    return " ".join(ted_words[best_lo:best_hi]), best_score


def run(eval_split_path: str, audio_root: str, raw_root: str, dl_root: str,
        min_match: float = 0.55) -> None:
    eval_ids = [x.strip().removeprefix("main/") for x in Path(eval_split_path).read_text().splitlines() if x.strip()]
    print(f"aligning TED labels for {len(eval_ids)} eval clips")

    asr = _whisper()
    audio_main = Path(audio_root) / "main"
    raw_main = Path(raw_root) / "main"

    ted_cache: dict[str, list[str]] = {}
    n_ted, n_fallback, n_fail = 0, 0, 0
    for vid in tqdm(eval_ids, desc="align"):
        talk, sent = vid.split("/")
        wav = audio_main / talk / f"{sent}.wav"
        dst = raw_main / talk / f"{sent}.txt"
        if not wav.exists():
            n_fail += 1
            continue
        try:
            hyp = _norm(asr(str(wav), generate_kwargs={"language": "english", "task": "transcribe"})["text"])
        except Exception as e:  # noqa: BLE001
            tqdm.write(f"  ASR fail {vid}: {e}")
            n_fail += 1
            continue

        if talk not in ted_cache:
            tpath = Path(dl_root) / talk / "talk.transcript.json"
            ted_cache[talk] = _ted_words(tpath) if tpath.exists() else []
        ted_words = ted_cache[talk]

        span, score = _best_span(hyp.split(), ted_words)
        if score >= min_match and span:
            label = span  # TED's own words
            n_ted += 1
        else:
            label = hyp   # safety fallback (flagged)
            n_fallback += 1
            tqdm.write(f"  low match {score:.2f} {vid}: using ASR fallback")
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(f"Text:  {label}\n")

    print(f"\nTED-aligned: {n_ted} | ASR-fallback: {n_fallback} | failed: {n_fail}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    for a in ("eval_split_path", "audio_root", "raw_root", "dl_root"):
        p.add_argument(f"--{a}", required=True)
    args = p.parse_args()
    run(args.eval_split_path, args.audio_root, args.raw_root, args.dl_root)
