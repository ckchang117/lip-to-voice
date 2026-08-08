"""Build train/val/test manifest files for the TED dataset.

Speaker-disjoint split: hold out whole talks for val/test (each TED talk =
one unique speaker, so this is trivially speaker-disjoint).

Output layout (matches what LipVoicer's `LipVoicerDataset` expects for LRS2):
  /vol/ted_raw/train.txt        (list of "main/<talk_id>/<sent_id>" per line)
  /vol/ted_raw/val.txt
  /vol/ted_raw/test.txt
  /vol/ted_raw/eval_200utt.txt  (deterministic 200 IDs sampled from test)

A clip ID is included only if all three of its files exist:
  /vol/ted_raw/main/<talk_id>/<sent_id>.mp4
  /vol/ted_proc/mouth_roi/main/<talk_id>/<sent_id>.npz
  /vol/ted_proc/audio/main/<talk_id>/<sent_id>.wav.spec
"""

from __future__ import annotations

import random
from pathlib import Path


def _collect_complete_clips(raw_root: Path, mouthroi_root: Path, audio_root: Path) -> dict[str, list[str]]:
    """Return mapping {talk_id: [sent_id, ...]} for clips that have all 3 files."""
    raw_main = raw_root / "main"
    out: dict[str, list[str]] = {}

    if not raw_main.exists():
        print(f"WARN: raw_main missing: {raw_main}")
        return out

    for talk_dir in sorted(raw_main.iterdir()):
        if not talk_dir.is_dir():
            continue
        complete: list[str] = []
        for mp4 in sorted(talk_dir.glob("*.mp4")):
            sent_id = mp4.stem
            npz = mouthroi_root / "main" / talk_dir.name / f"{sent_id}.npz"
            spec = audio_root / "main" / talk_dir.name / f"{sent_id}.wav.spec"
            if npz.exists() and spec.exists() and mp4.stat().st_size > 0:
                complete.append(sent_id)
        if complete:
            out[talk_dir.name] = complete
    return out


def run(
    raw_root: str,
    mouthroi_root: str,
    audio_root: str,
    test_talks: int = 10,
    val_talks: int = 5,
    eval_n: int = 200,
    seed: int = 1234,
) -> None:
    raw_path = Path(raw_root)
    clips_by_talk = _collect_complete_clips(raw_path, Path(mouthroi_root), Path(audio_root))
    talks = sorted(clips_by_talk.keys())
    print(f"complete talks: {len(talks)}")
    total_clips = sum(len(v) for v in clips_by_talk.values())
    print(f"total complete clips: {total_clips}")

    if len(talks) < (test_talks + val_talks + 1):
        print(f"WARN: not enough talks ({len(talks)}) for split (need >={test_talks + val_talks + 1})")
        test_talks = max(1, len(talks) // 10)
        val_talks = max(1, len(talks) // 20)
        print(f"  scaled to test={test_talks}, val={val_talks}")

    rng = random.Random(seed)
    shuffled = list(talks)
    rng.shuffle(shuffled)
    test_set = sorted(shuffled[:test_talks])
    val_set = sorted(shuffled[test_talks : test_talks + val_talks])
    train_set = sorted(shuffled[test_talks + val_talks :])

    def _ids_for(talks_list: list[str]) -> list[str]:
        # LipVoicerDataset prepends "main/" to each ID itself, so manifest IDs
        # are just "<talk_id>/<sent_id>" (matches LRS2 convention).
        ids: list[str] = []
        for t in talks_list:
            for s in clips_by_talk[t]:
                ids.append(f"{t}/{s}")
        return ids

    train_ids = _ids_for(train_set)
    val_ids = _ids_for(val_set)
    test_ids = _ids_for(test_set)

    (raw_path / "train.txt").write_text("\n".join(train_ids) + "\n")
    (raw_path / "val.txt").write_text("\n".join(val_ids) + "\n")
    (raw_path / "test.txt").write_text("\n".join(test_ids) + "\n")

    # Deterministic eval split
    rng2 = random.Random(seed + 1)
    eval_ids = sorted(rng2.sample(test_ids, k=min(eval_n, len(test_ids))))
    (raw_path / f"eval_{eval_n}utt.txt").write_text("\n".join(eval_ids) + "\n")

    print()
    print(f"=== manifest ===")
    print(f"train: {len(train_set)} talks, {len(train_ids)} clips")
    print(f"val:   {len(val_set)} talks, {len(val_ids)} clips")
    print(f"test:  {len(test_set)} talks, {len(test_ids)} clips")
    print(f"eval:  {len(eval_ids)} clips (sampled from test)")
    print(f"wrote -> {raw_path}/{{train,val,test,eval_{eval_n}utt}}.txt")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--raw_root", required=True)
    p.add_argument("--mouthroi_root", required=True)
    p.add_argument("--audio_root", required=True)
    p.add_argument("--test_talks", type=int, default=10)
    p.add_argument("--val_talks", type=int, default=5)
    p.add_argument("--eval_n", type=int, default=200)
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args()
    run(
        args.raw_root,
        args.mouthroi_root,
        args.audio_root,
        test_talks=args.test_talks,
        val_talks=args.val_talks,
        eval_n=args.eval_n,
        seed=args.seed,
    )
