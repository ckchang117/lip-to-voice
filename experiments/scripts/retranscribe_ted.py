"""Re-transcribe each TED clip's audio with Whisper and overwrite the .txt.

Background: TED's GraphQL cue timestamps are offset from the MP4 timeline
(TED intros + applause shift everything by ~3 sec). Our segment_ted used
cue timestamps verbatim, so the clip audio doesn't align with the cue text.

The clip audio itself is internally consistent with the clip's mouth ROIs -
both come from the same time window of the original video. Re-Whispering
the audio gives us a ground-truth transcript that matches what's actually
spoken in the clip, which is what the eval harness needs.

Reads from /vol/ted_lrs2/proc/audio/main/<talk>/<sent>.wav (already 16 kHz mono)
and overwrites /vol/ted_lrs2/raw/main/<talk>/<sent>.txt in LRS2 "Text:  ..."
format (UPPERCASE, no punctuation).
"""

from __future__ import annotations

import re
from pathlib import Path

from tqdm import tqdm


_ASR = None


def _whisper():
    global _ASR
    if _ASR is None:
        import torch
        from transformers import pipeline
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        print(f"loading Whisper-large-v3 on {device} ({torch_dtype})...")
        _ASR = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-large-v3",
            torch_dtype=torch_dtype,
            device=device,
            return_timestamps=False,
        )
        print("Whisper loaded")
    return _ASR


def _normalize_for_lrs2(text: str) -> str:
    """Strip non-LRS2 characters; convert to UPPERCASE.

    LRS2 transcripts are space-separated UPPERCASE word tokens, no punctuation
    other than apostrophes (for contractions like DON'T).
    """
    text = text.upper().strip()
    # Replace common punctuation with space, preserve apostrophes inside words
    text = re.sub(r"[^A-Z' ]+", " ", text)
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def run(audio_root: str, raw_root: str, batch_size: int = 1) -> None:
    audio_path = Path(audio_root) / "main"
    raw_path = Path(raw_root) / "main"
    if not audio_path.exists():
        raise SystemExit(f"missing: {audio_path}")

    # Collect all wavs (one per clip)
    wavs: list[tuple[str, str, Path, Path]] = []  # (talk, sent, wav, dst_txt)
    for talk_dir in sorted(audio_path.iterdir()):
        if not talk_dir.is_dir():
            continue
        for wav in sorted(talk_dir.glob("*.wav")):
            sent = wav.stem
            dst_txt = raw_path / talk_dir.name / f"{sent}.txt"
            if dst_txt.exists():
                wavs.append((talk_dir.name, sent, wav, dst_txt))

    print(f"found {len(wavs)} clips to retranscribe")
    if not wavs:
        return

    asr = _whisper()

    n_ok = 0
    n_fail = 0
    n_empty = 0
    for talk, sent, wav, dst_txt in tqdm(wavs, desc="whisper"):
        try:
            out = asr(
                str(wav),
                batch_size=batch_size,
                generate_kwargs={"language": "english", "task": "transcribe"},
            )
            raw_text = (out.get("text") or "").strip()
            text = _normalize_for_lrs2(raw_text)
            if not text:
                n_empty += 1
                continue
            dst_txt.write_text(f"Text:  {text}\n")
            n_ok += 1
        except Exception as e:
            n_fail += 1
            tqdm.write(f"  FAIL {talk}/{sent}: {type(e).__name__}: {e}")

    print(f"\nok: {n_ok} / {len(wavs)}")
    print(f"empty: {n_empty}")
    print(f"failed: {n_fail}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--audio_root", required=True)
    p.add_argument("--raw_root", required=True)
    args = p.parse_args()
    run(args.audio_root, args.raw_root)
