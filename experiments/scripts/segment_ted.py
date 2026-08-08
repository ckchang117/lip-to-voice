"""Segment downloaded TED talks into sentence-aligned clips with single-face filter.

Inputs (per talk):
  /vol/ted_dl/<talk_id>/talk.mp4
  /vol/ted_dl/<talk_id>/talk.transcript.json   (from TED GraphQL, cues in ms)

Outputs:
  /vol/ted_lrs2/raw/main/<talk_id>/<sent_idx>.mp4
  /vol/ted_lrs2/raw/main/<talk_id>/<sent_idx>.txt    ("Text:  <transcript>\n")
  /vol/ted_lrs2/raw/main/<talk_id>/_DONE

Pipeline per talk:
  1. Load cached transcript cues (ms timestamps + caption text)
  2. Group adjacent cues into sentences (split on terminal punctuation
     and gaps > 0.5 s)
  3. For each candidate sentence in [2.5, 12] sec:
       - Trim 0.2 sec off each end to absorb cue drift
       - Sample 10 frames; require mediapipe face-detection sees one face
         in >= 80% of samples
       - ffmpeg-cut [start, end] from talk.mp4 to <sent_idx>.mp4
       - Write LRS2-style "Text:  <UPPERCASE>" transcript

No GPU, no Whisper. Just mediapipe + ffmpeg + cached JSON.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm


MIN_CLIP_SEC = 2.5
MAX_CLIP_SEC = 12.0
TRIM_EDGE_SEC = 0.20
SAMPLE_FRAMES = 10
MIN_FACE_FRACTION = 0.8
SENT_GAP_SEC = 0.5  # gap between adjacent cues that forces a sentence break

_FACE_DETECTOR = None


def _detector():
    global _FACE_DETECTOR
    if _FACE_DETECTOR is None:
        _FACE_DETECTOR = mp.solutions.face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5
        )
    return _FACE_DETECTOR


def _clean_cue_text(text: str) -> str:
    """Strip cue formatting tags + collapse whitespace."""
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\[.*?\]", "", text)  # [Applause], [Music]
    text = re.sub(r"\(.*?\)", "", text)  # (laughter)
    return " ".join(text.split()).strip()


def _group_cues_into_sentences(cues: list[dict]) -> list[tuple[float, float, str]]:
    """Group cue dicts (with ms timestamps) into (start_sec, end_sec, text) sentence chunks.

    A new sentence starts when:
      - the previous cue ended in terminal punctuation (.?!), OR
      - the gap between consecutive cues exceeds SENT_GAP_SEC.
    """
    sentences: list[tuple[float, float, str]] = []
    cur_parts: list[str] = []
    cur_start_ms: Optional[int] = None
    cur_end_ms: Optional[int] = None

    SENT_END = set(".?!")

    def flush():
        nonlocal cur_parts, cur_start_ms, cur_end_ms
        if cur_parts and cur_start_ms is not None and cur_end_ms is not None:
            joined = " ".join(cur_parts).strip()
            joined = re.sub(r"\s+([,.!?;:])", r"\1", joined)  # remove space before punct
            if joined:
                sentences.append((cur_start_ms / 1000.0, cur_end_ms / 1000.0, joined))
        cur_parts = []
        cur_start_ms = None
        cur_end_ms = None

    for cue in cues:
        text = _clean_cue_text(cue.get("text") or "")
        if not text:
            continue
        st = cue.get("startTime")
        en = cue.get("endTime")
        if st is None or en is None:
            continue

        # New sentence if there's a gap from previous cue
        if cur_end_ms is not None and (st - cur_end_ms) > SENT_GAP_SEC * 1000:
            flush()

        if cur_start_ms is None:
            cur_start_ms = int(st)
        cur_end_ms = int(en)
        cur_parts.append(text)

        # Flush on terminal punctuation
        if text[-1] in SENT_END:
            flush()

    flush()
    return sentences


def _has_single_face(frame: np.ndarray) -> Optional[bool]:
    if frame is None or frame.size == 0:
        return None
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    try:
        results = _detector().process(rgb)
    except Exception:
        return None
    n = len(results.detections) if results.detections else 0
    return n == 1


def _check_single_face(video_path: Path, start_sec: float, end_sec: float) -> bool:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False
    duration = end_sec - start_sec
    if duration <= 0:
        cap.release()
        return False
    sample_times = np.linspace(start_sec, end_sec, SAMPLE_FRAMES)
    n_single = 0
    n_valid = 0
    for t in sample_times:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if not ok:
            continue
        n_valid += 1
        if _has_single_face(frame) is True:
            n_single += 1
    cap.release()
    if n_valid == 0:
        return False
    return (n_single / n_valid) >= MIN_FACE_FRACTION


def _ffmpeg_cut(src: Path, dst: Path, start_sec: float, end_sec: float) -> bool:
    duration = end_sec - start_sec
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", f"{start_sec:.3f}",
        "-i", str(src),
        "-t", f"{duration:.3f}",
        "-r", "25",
        "-ar", "16000", "-ac", "1",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-c:a", "aac", "-b:a", "64k",
        str(dst),
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=120)
        return r.returncode == 0 and dst.exists() and dst.stat().st_size > 0
    except subprocess.TimeoutExpired:
        return False


def segment_one_talk(talk_id: str, dl_root: Path, out_root: Path) -> dict:
    src_mp4 = dl_root / talk_id / "talk.mp4"
    src_transcript = dl_root / talk_id / "talk.transcript.json"

    if not src_mp4.exists():
        return {"talk_id": talk_id, "skipped": True, "reason": "missing mp4", "kept": 0, "considered": 0, "kept_seconds": 0.0}
    if not src_transcript.exists():
        return {"talk_id": talk_id, "skipped": True, "reason": "missing transcript", "kept": 0, "considered": 0, "kept_seconds": 0.0}

    out_talk_dir = out_root / "main" / talk_id
    out_talk_dir.mkdir(parents=True, exist_ok=True)
    done_marker = out_talk_dir / "_DONE"

    if done_marker.exists():
        existing = list(out_talk_dir.glob("*.mp4"))
        return {
            "talk_id": talk_id,
            "skipped": True,
            "reason": "already done",
            "kept": len(existing),
            "considered": 0,
            "kept_seconds": 0.0,
        }

    try:
        transcript = json.loads(src_transcript.read_text())
        cues = transcript.get("cues") or []
    except Exception as e:
        return {"talk_id": talk_id, "skipped": True, "reason": f"transcript parse failed: {e}", "kept": 0, "considered": 0, "kept_seconds": 0.0}

    if not cues:
        return {"talk_id": talk_id, "skipped": True, "reason": "empty cues", "kept": 0, "considered": 0, "kept_seconds": 0.0}

    sentences = _group_cues_into_sentences(cues)

    n_considered = 0
    n_kept = 0
    total_kept_seconds = 0.0

    for idx, (start, end, text) in enumerate(sentences):
        n_considered += 1
        trimmed_start = max(0.0, start + TRIM_EDGE_SEC)
        trimmed_end = end - TRIM_EDGE_SEC
        duration = trimmed_end - trimmed_start
        if duration < MIN_CLIP_SEC or duration > MAX_CLIP_SEC:
            continue
        if len(text.split()) < 3:
            continue
        if not _check_single_face(src_mp4, trimmed_start, trimmed_end):
            continue

        sent_id = f"{idx:05d}"
        dst_mp4 = out_talk_dir / f"{sent_id}.mp4"
        dst_txt = out_talk_dir / f"{sent_id}.txt"

        if dst_mp4.exists() and dst_mp4.stat().st_size > 0 and dst_txt.exists():
            n_kept += 1
            total_kept_seconds += duration
            continue

        if not _ffmpeg_cut(src_mp4, dst_mp4, trimmed_start, trimmed_end):
            continue

        dst_txt.write_text(f"Text:  {text.upper()}\n")
        n_kept += 1
        total_kept_seconds += duration

    done_marker.touch()
    return {
        "talk_id": talk_id,
        "skipped": False,
        "considered": n_considered,
        "kept": n_kept,
        "kept_seconds": total_kept_seconds,
    }


def run(dl_root: str, out_root: str, summary_path: Optional[str] = None) -> None:
    dl_path = Path(dl_root)
    out_path = Path(out_root)
    out_path.mkdir(parents=True, exist_ok=True)

    summary_file = dl_path / "download_summary.txt"
    if summary_file.exists():
        talk_ids = [t.strip() for t in summary_file.read_text().splitlines() if t.strip()]
    else:
        talk_ids = sorted(p.name for p in dl_path.iterdir() if p.is_dir())
    print(f"considering {len(talk_ids)} talks")

    stats = []
    for tid in tqdm(talk_ids, desc="segment"):
        s = segment_one_talk(tid, dl_path, out_path)
        stats.append(s)
        if s.get("skipped"):
            tqdm.write(f"  {tid}: skipped ({s.get('reason')})")
        else:
            tqdm.write(f"  {tid}: kept {s['kept']}/{s['considered']} ({s.get('kept_seconds', 0):.0f}s)")

    total_kept = sum(s["kept"] for s in stats)
    total_considered = sum(s["considered"] for s in stats)
    total_seconds = sum(s.get("kept_seconds", 0) for s in stats)
    retention = total_kept / max(total_considered, 1)
    print()
    print(f"=== segmentation summary ===")
    print(f"talks processed: {len([s for s in stats if not s.get('skipped')])}")
    print(f"clips kept: {total_kept} / {total_considered} ({retention:.1%} retention)")
    print(f"total kept duration: {total_seconds:.0f} sec ({total_seconds / 3600:.2f} hr)")

    if summary_path:
        Path(summary_path).write_text(
            f"talks: {len(stats)}\n"
            f"clips: {total_kept}\n"
            f"hours: {total_seconds / 3600:.2f}\n"
            f"retention: {retention:.3f}\n"
        )


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dl_root", required=True)
    p.add_argument("--out_root", required=True)
    p.add_argument("--summary", default=None)
    args = p.parse_args()
    run(args.dl_root, args.out_root, summary_path=args.summary)
