"""Run LipVoicer's mouth-ROI + lip-reader pipeline over segmented TED clips.

For each /vol/ted_raw/main/<talk>/<sent>.mp4:
  - Extract 88x88 grayscale mouth ROI -> /vol/ted_proc/mouth_roi/main/<talk>/<sent>.npz
  - Run lip-reading inference -> /vol/ted_proc/lipread_text/<talk>/<sent>.txt

The pipeline (LipVoicer's `mouthroi_processing.InferencePipeline`) is built
ONCE and reused across clips - loading the lip-reader model per clip would
be prohibitively expensive.

Must run with CWD = /root/repo so LipVoicer's relative config paths resolve.
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

import numpy as np
from tqdm import tqdm


# Defer torch import; this runs inside LIPVOICER_IMAGE where torch is available.


def _link_checkpoints(repo_root: str = "/root/repo", ckpt_root: str = "/vol/checkpoints/lipvoicer") -> None:
    """Symlink LipVoicer's pretrained files into the repo at their expected paths.

    LipVoicer's configs use cwd-relative paths like
    `ASR/media/tokenizerbpe256.model` and
    `mouthroi_processing/benchmarks/LRS3/models/LRS3_V_WER19.1/model.pth`.
    Some of the destination directories already exist in the repo with
    real content (e.g. `ASR/media/20words_mean_face.npy`), so we can't
    just symlink whole directories - we link individual files instead.

    For directories that don't exist in the repo at all (e.g. the
    benchmark model dirs that aren't checked in), we link the whole dir.
    """
    from pathlib import Path
    repo = Path(repo_root)
    ckpt = Path(ckpt_root)

    # (relative_path, kind) where kind is "file" or "dir"
    targets = [
        ("exp/LRS2/wnet_h512_d12_T400_betaT0.02/checkpoint/1000000.pkl", "file"),
        ("ASR/callbacks/LRS23/AO/EffConfCTC/checkpoints_ft_lrs2.ckpt", "file"),
        ("ASR/callbacks/LRS23/LM/GPT-Small/checkpoints_epoch_10_step_2860.ckpt", "file"),
        ("ASR/media/tokenizerbpe256.model", "file"),
        ("ASR/media/tokenizerbpe1024.model", "file"),
        ("ASR/media/6gram_lrs23.arpa", "file"),
        ("hifi_gan/g_02400000", "file"),
        ("mouthroi_processing/benchmarks/LRS3/models/LRS3_V_WER19.1", "dir"),
        ("mouthroi_processing/benchmarks/LRS3/language_models/lm_en_subword", "dir"),
    ]

    for rel, kind in targets:
        src = ckpt / rel
        dst = repo / rel
        if not src.exists():
            print(f"  [skip] missing in checkpoints: {src}")
            continue
        if dst.is_symlink() or dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            dst.symlink_to(src)
            print(f"  linked {dst} -> {src}")
        except OSError as e:
            print(f"  WARN symlink failed {dst}: {e}")


def _build_pipeline():
    """Construct LipVoicer's InferencePipeline once. Requires cwd=/root/repo."""
    if "/root/repo" not in sys.path:
        sys.path.insert(0, "/root/repo")
    os.chdir("/root/repo")
    _link_checkpoints()

    from mouthroi_processing.pipelines.pipeline import InferencePipeline

    config_filename = "mouthroi_processing/configs/LRS3_V_WER19.1.ini"
    # mediapipe (not retinaface) - ibug.face_detection's LFS weights are unavailable.
    pipeline = InferencePipeline(
        config_filename,
        detector="mediapipe",
        face_track=True,
        device="cuda:0",
    )
    return pipeline


def process_one_clip(pipeline, clip_path: Path, npz_out: Path, txt_out: Path) -> tuple[bool, str]:
    """Process a single clip. Returns (success, message)."""
    try:
        landmarks = pipeline.process_landmarks(str(clip_path), landmarks_filename=None)
        video = pipeline.dataloader.load_video(str(clip_path))
        mouth_crop = pipeline.dataloader.video_process(video, landmarks)
    except Exception as e:
        return False, f"mouth ROI extract failed: {e}"

    if mouth_crop is None or len(mouth_crop) == 0:
        return False, "empty mouth crop"

    npz_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(npz_out), data=np.asarray(mouth_crop))

    # Run lip-reading inference on the cropped mouth video
    try:
        import torch
        mouth_tensor = torch.tensor(mouth_crop)
        transformed = pipeline.dataloader.video_transform(mouth_tensor)
        transcript = pipeline.model.infer(transformed).lower()
    except Exception as e:
        transcript = ""
        traceback.print_exc()
        return True, f"npz saved; lipread failed: {e}"

    txt_out.parent.mkdir(parents=True, exist_ok=True)
    txt_out.write_text(transcript.strip() + "\n")
    return True, "ok"


def run(raw_root: str, mouthroi_root: str, lipread_root: str) -> None:
    raw_path = Path(raw_root) / "main"
    mouthroi_path = Path(mouthroi_root) / "main"
    lipread_path = Path(lipread_root)
    mouthroi_path.mkdir(parents=True, exist_ok=True)
    lipread_path.mkdir(parents=True, exist_ok=True)

    clips: list[tuple[str, str]] = []
    for talk_dir in sorted(raw_path.iterdir()):
        if not talk_dir.is_dir():
            continue
        for mp4 in sorted(talk_dir.glob("*.mp4")):
            clips.append((talk_dir.name, mp4.stem))

    print(f"found {len(clips)} clips to process")
    if not clips:
        print("nothing to do")
        return

    print("building pipeline (loads lip-reader + face detector - slow)...")
    pipeline = _build_pipeline()
    print("pipeline ready")

    n_ok = 0
    n_fail = 0
    for talk_id, sent_id in tqdm(clips, desc="mouth ROI"):
        clip_path = raw_path / talk_id / f"{sent_id}.mp4"
        npz_out = mouthroi_path / talk_id / f"{sent_id}.npz"
        txt_out = lipread_path / talk_id / f"{sent_id}.txt"

        if npz_out.exists() and txt_out.exists() and npz_out.stat().st_size > 0:
            n_ok += 1
            continue

        ok, msg = process_one_clip(pipeline, clip_path, npz_out, txt_out)
        if ok:
            n_ok += 1
        else:
            n_fail += 1
            tqdm.write(f"  FAIL {talk_id}/{sent_id}: {msg}")

    print(f"\nsuccess: {n_ok} / {len(clips)}")
    print(f"failures: {n_fail}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--raw_root", required=True)
    p.add_argument("--mouthroi_root", required=True)
    p.add_argument("--lipread_root", required=True)
    args = p.parse_args()
    run(args.raw_root, args.mouthroi_root, args.lipread_root)
