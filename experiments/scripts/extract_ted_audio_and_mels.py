"""Extract 16 kHz mono WAVs + 80-bin mel spectrograms from segmented TED clips.

For each /vol/ted_raw/main/<talk>/<sent>.mp4:
  - ffmpeg -> /vol/ted_proc/audio/main/<talk>/<sent>.wav   (16 kHz mono)
  - Tacotron-STFT -> /vol/ted_proc/audio/main/<talk>/<sent>.wav.spec  (torch.save)

The mel spec format must match LipVoicer's `dataloaders/stft.py:TacotronSTFT`
+ `wav2mel.py`. We import those directly from the repo to ensure byte-for-byte
compatibility with the released MelGen checkpoint's expected input distribution.

Requires CWD=/root/repo for relative imports.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm


def _setup_repo_imports() -> None:
    if "/root/repo" not in sys.path:
        sys.path.insert(0, "/root/repo")
    os.chdir("/root/repo")


def _extract_wav(src_mp4: Path, dst_wav: Path) -> bool:
    dst_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(src_mp4),
        "-vn",
        "-ar", "16000", "-ac", "1",
        "-c:a", "pcm_s16le",
        str(dst_wav),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=60)
        return result.returncode == 0 and dst_wav.exists() and dst_wav.stat().st_size > 0
    except subprocess.TimeoutExpired:
        return False


def _compute_mel(stft, wav_path: Path, spec_path: Path) -> bool:
    import torch
    from scipy.io.wavfile import read as wav_read

    try:
        sr, data = wav_read(str(wav_path))
        if sr != 16000:
            return False
        audio = torch.from_numpy(data).float()
        # Match LipVoicer's wav2mel.py normalization
        audio = audio / 1.1 / audio.abs().max()
        audio = audio.unsqueeze(0)
        mel = stft.mel_spectrogram(audio)
        mel = mel.squeeze(0)
        torch.save(mel, str(spec_path))
        return True
    except Exception as e:
        print(f"  mel compute failed for {wav_path}: {e}")
        return False


def run(raw_root: str, audio_root: str) -> None:
    _setup_repo_imports()

    # Import LipVoicer's STFT - same one used to compute training mels.
    from dataloaders.stft import TacotronSTFT

    # Audio config from configs/config.yaml (matches the released MelGen checkpoint).
    stft = TacotronSTFT(
        filter_length=640,
        hop_length=160,
        win_length=640,
        sampling_rate=16000,
        mel_fmin=20.0,
        mel_fmax=8000.0,
    )

    raw_path = Path(raw_root) / "main"
    audio_path = Path(audio_root) / "main"
    audio_path.mkdir(parents=True, exist_ok=True)

    clips: list[tuple[str, str]] = []
    for talk_dir in sorted(raw_path.iterdir()):
        if not talk_dir.is_dir():
            continue
        for mp4 in sorted(talk_dir.glob("*.mp4")):
            clips.append((talk_dir.name, mp4.stem))

    print(f"found {len(clips)} clips to extract audio + mels")

    n_ok = 0
    n_fail = 0
    for talk_id, sent_id in tqdm(clips, desc="audio+mel"):
        src_mp4 = raw_path / talk_id / f"{sent_id}.mp4"
        dst_wav = audio_path / talk_id / f"{sent_id}.wav"
        dst_spec = audio_path / talk_id / f"{sent_id}.wav.spec"

        if dst_wav.exists() and dst_spec.exists() and dst_wav.stat().st_size > 0:
            n_ok += 1
            continue

        if not _extract_wav(src_mp4, dst_wav):
            n_fail += 1
            tqdm.write(f"  FAIL wav {talk_id}/{sent_id}")
            continue

        if not _compute_mel(stft, dst_wav, dst_spec):
            n_fail += 1
            tqdm.write(f"  FAIL mel {talk_id}/{sent_id}")
            continue

        n_ok += 1

    print(f"\nsuccess: {n_ok} / {len(clips)}")
    print(f"failures: {n_fail}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--raw_root", required=True)
    p.add_argument("--audio_root", required=True)
    args = p.parse_args()
    run(args.raw_root, args.audio_root)
