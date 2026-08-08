"""Precompute RAFT-small optical flow for each cached mouth-ROI clip.

Mirrors process_ted_mouth_rois.py: per-clip, idempotent, builds the model once.
RAFT is far too slow to run live in the dataloader, so we cache flow to the
volume and the training dataloader just slices it (co-indexed with the mouthroi).

Input  (per clip):  <mouthroi_root>/main/<talk>/<sent>.npz  key 'data' = [T_full, H, W] uint8
Output (per clip):  <flow_root>/main/<talk>/<sent>.npz       key 'data' = [T_full, 2, H, W] float16

Alignment contract: flow[0] is zero, flow[i] = RAFT(frame[i-1] -> frame[i]) for i>=1.
This makes flow[st:st+W] co-indexed 1:1 with mouthroi[st:st+W] (no off-by-one).
Flow vectors are expressed in native-ROI pixel units (downsampled + rescaled from
the 224px RAFT resolution) and clipped to +/-20 px to suppress small-crop outliers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


_RAFT = None
_RAFT_TF = None
MAG_CLIP = 20.0


def _raft(device: str = "cuda:0"):
    global _RAFT, _RAFT_TF
    if _RAFT is None:
        from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
        weights = Raft_Small_Weights.DEFAULT
        print("loading RAFT-small...")
        _RAFT = raft_small(weights=weights, progress=False).eval().to(device)
        _RAFT_TF = weights.transforms()
        print("RAFT loaded")
    return _RAFT, _RAFT_TF


def _to_rgb(frames_u8: np.ndarray, upsample: int, device: str) -> torch.Tensor:
    """[T,H,W] uint8 -> [T,3,upsample,upsample] float in [0,1] on device."""
    x = torch.from_numpy(frames_u8.astype(np.float32) / 255.0).unsqueeze(1)  # [T,1,H,W]
    x = F.interpolate(x, size=(upsample, upsample), mode="bilinear", align_corners=False)
    return x.repeat(1, 3, 1, 1).to(device)


@torch.no_grad()
def compute_clip_flow(npz_in: Path, npz_out: Path, device: str, upsample: int, chunk: int) -> str:
    if npz_out.exists() and npz_out.stat().st_size > 0:
        return "cached"
    frames = np.load(npz_in)["data"]  # [T,H,W] uint8
    if frames.ndim != 3 or frames.shape[0] < 2:
        return f"skip (shape {frames.shape})"
    T, H, W = frames.shape

    model, tf = _raft(device)
    rgb = _to_rgb(frames, upsample, device)         # [T,3,up,up]
    img1_all, img2_all = rgb[:-1], rgb[1:]          # [T-1, ...]

    flows = []
    for s in range(0, img1_all.shape[0], chunk):
        a, b = img1_all[s:s + chunk], img2_all[s:s + chunk]
        a, b = tf(a, b)
        pred = model(a, b)[-1]                       # [n,2,up,up]
        pred = F.interpolate(pred, size=(H, W), mode="bilinear", align_corners=False)
        pred = pred * (float(H) / float(upsample))   # rescale vectors to native px
        flows.append(pred.clamp_(-MAG_CLIP, MAG_CLIP).cpu())
    flow = torch.cat(flows, dim=0)                   # [T-1, 2, H, W]

    zero = torch.zeros(1, 2, H, W)
    flow = torch.cat([zero, flow], dim=0)            # [T, 2, H, W], flow[0]=0
    npz_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(npz_out), data=flow.numpy().astype(np.float16))
    return "ok"


def run_talk(talk_id: str, mouthroi_root: str, flow_root: str,
             device: str = "cuda:0", upsample: int = 224, chunk: int = 16) -> dict:
    """Process every clip of one talk. Idempotent; safe for parallel .map()."""
    src_dir = Path(mouthroi_root) / "main" / talk_id
    out_dir = Path(flow_root) / "main" / talk_id
    if not src_dir.is_dir():
        return {"talk_id": talk_id, "ok": 0, "fail": 0, "reason": "no mouthroi dir"}
    n_ok = n_fail = 0
    for npz_in in sorted(src_dir.glob("*.npz")):
        npz_out = out_dir / npz_in.name
        try:
            compute_clip_flow(npz_in, npz_out, device, upsample, chunk)
            n_ok += 1
        except Exception as e:  # noqa: BLE001
            n_fail += 1
            tqdm.write(f"  FAIL {talk_id}/{npz_in.stem}: {type(e).__name__}: {e}")
    return {"talk_id": talk_id, "ok": n_ok, "fail": n_fail}


def run(mouthroi_root: str, flow_root: str,
        device: str = "cuda:0", upsample: int = 224, chunk: int = 16) -> None:
    """Serial fallback over all talks (Modal uses the parallel .map path instead)."""
    src_path = Path(mouthroi_root) / "main"
    talk_ids = sorted(p.name for p in src_path.iterdir() if p.is_dir())
    print(f"precompute flow over {len(talk_ids)} talks")
    tot_ok = tot_fail = 0
    for tid in tqdm(talk_ids, desc="flow"):
        s = run_talk(tid, mouthroi_root, flow_root, device, upsample, chunk)
        tot_ok += s["ok"]
        tot_fail += s["fail"]
    print(f"\nflow done: {tot_ok} clips, {tot_fail} failed")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--mouthroi_root", required=True)
    p.add_argument("--flow_root", required=True)
    p.add_argument("--upsample", type=int, default=224)
    p.add_argument("--chunk", type=int, default=16)
    args = p.parse_args()
    run(args.mouthroi_root, args.flow_root, upsample=args.upsample, chunk=args.chunk)
