"""Compare flow-tuned (C) vs. frozen baseline (A) eval JSONs.

Reads two files written by eval_baseline.run (schema:
{"per_clip": {<id>: {<metric>: val}}, "aggregate": {<metric>: {"mean",...}}}).
Prints the per-metric mean delta (C - A) over the aggregate, plus a PAIRED delta
restricted to clip IDs scored in both runs.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean

# Lower is better for these; higher for everything else.
LOWER_BETTER = {"wer_whisper", "wer_wav2vec2"}


def _paired(a_per, b_per, metric):
    vals_a, vals_b = [], []
    for vid in a_per.keys() & b_per.keys():
        va, vb = a_per[vid].get(metric, math.nan), b_per[vid].get(metric, math.nan)
        if not (math.isnan(va) or math.isnan(vb)):
            vals_a.append(va)
            vals_b.append(vb)
    if not vals_a:
        return math.nan, math.nan, 0
    return mean(vals_a), mean(vals_b), len(vals_a)


def run(baseline_json: str, flow_json: str) -> None:
    A = json.loads(Path(baseline_json).read_text())
    B = json.loads(Path(flow_json).read_text())
    a_agg, b_agg = A["aggregate"], B["aggregate"]
    a_per, b_per = A.get("per_clip", {}), B.get("per_clip", {})

    metrics = [k for k in a_agg.keys() if k in b_agg]
    print(f"\n{'metric':16s} {'A(base)':>9s} {'C(flow)':>9s} {'delta':>9s} {'better?':>8s}   "
          f"{'paired_A':>9s} {'paired_C':>9s} {'n':>4s}")
    print("-" * 90)
    for k in metrics:
        a_mean, b_mean = a_agg[k]["mean"], b_agg[k]["mean"]
        delta = b_mean - a_mean
        if k in LOWER_BETTER:
            better = "C" if delta < 0 else ("A" if delta > 0 else "=")
        else:
            better = "C" if delta > 0 else ("A" if delta < 0 else "=")
        pa, pc, n = _paired(a_per, b_per, k)
        print(f"{k:16s} {a_mean:9.4f} {b_mean:9.4f} {delta:+9.4f} {better:>8s}   "
              f"{pa:9.4f} {pc:9.4f} {n:4d}")
    print("\n(WER lower=better; STOI/ESTOI/DNSMOS/UTMOS higher=better)")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--baseline_json", required=True)
    p.add_argument("--flow_json", required=True)
    args = p.parse_args()
    run(args.baseline_json, args.flow_json)
