"""Fetch a list of usable TED talk URLs.

Strategy:
  1. Pull TED's master sitemap index, follow year-sharded `talks-YYYY.xml.gz`.
  2. Shuffle deterministically.
  3. For each candidate, fetch its page and validate that it (a) has a direct
     h264 MP4 on TED's CDN, (b) is mainline TED (not TEDx), and (c) is in English.
     ~95% of TED's catalog is TEDx with no CDN MP4, so we oversample heavily.
  4. Stop after N validated talks.

Output: one URL per line at <out_path>. Idempotent (overwrites).
"""

from __future__ import annotations

import gzip
import json
import random
import re
from pathlib import Path
from typing import Iterable

import requests
from tqdm import tqdm

SITEMAP_INDEX = "https://www.ted.com/sitemap.xml"
_BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) "
    "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15"
)


def _fetch(url: str) -> bytes:
    resp = requests.get(url, headers={"User-Agent": _BROWSER_UA}, timeout=30)
    resp.raise_for_status()
    raw = resp.content
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return raw


def fetch_talk_urls() -> list[str]:
    print(f"GET {SITEMAP_INDEX}")
    index_bytes = _fetch(SITEMAP_INDEX)
    locs = re.findall(rb"<loc>(.*?)</loc>", index_bytes)
    locs_str = [u.decode("utf-8") for u in locs]
    talk_shards = [u for u in locs_str if "/talks-" in u]
    print(f"  found {len(talk_shards)} year-sharded talk sitemaps")

    talk_urls: list[str] = []
    for shard in tqdm(talk_shards, desc="sitemap shards"):
        try:
            shard_bytes = _fetch(shard)
            for u in re.findall(rb"<loc>(.*?)</loc>", shard_bytes):
                u = u.decode("utf-8")
                if "/talks/" in u:
                    talk_urls.append(u)
        except Exception as e:
            print(f"  WARN: failed {shard}: {e}")

    out = sorted(set(talk_urls))
    print(f"  total unique talk URLs: {len(out)}")
    return out


def _validate_url(url: str) -> tuple[bool, str]:
    """Return (ok, reason). ok=True means the talk has a CDN MP4 and is mainline-TED English."""
    try:
        resp = requests.get(url, headers={"User-Agent": _BROWSER_UA}, timeout=15)
        resp.raise_for_status()
    except Exception as e:
        return False, f"page fetch failed: {e}"

    m = re.search(
        r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>',
        resp.text,
        re.DOTALL,
    )
    if not m:
        return False, "no __NEXT_DATA__"

    try:
        data = json.loads(m.group(1))
        talk = data["props"]["pageProps"]["videoData"]
    except (KeyError, ValueError, json.JSONDecodeError) as e:
        return False, f"videoData missing: {e}"

    if talk.get("internalLanguageCode") not in (None, "en"):
        return False, f"non-English ({talk.get('internalLanguageCode')})"

    try:
        pd = json.loads(talk["playerData"])
    except (KeyError, ValueError) as e:
        return False, f"playerData parse: {e}"

    event = (pd.get("event") or "").strip()
    if event.lower().startswith("tedx"):
        return False, f"TEDx event ({event})"

    h264 = pd.get("resources", {}).get("h264") or []
    if not h264:
        return False, "no h264 MP4"

    return True, "ok"


def select_validated(urls: list[str], n: int, seed: int, max_candidates: int) -> list[str]:
    rng = random.Random(seed)
    shuffled = list(urls)
    rng.shuffle(shuffled)
    candidates = shuffled[:max_candidates]
    print(f"validating up to {len(candidates)} candidates (target: {n} good)")

    accepted: list[str] = []
    rejected = 0
    reason_counts: dict[str, int] = {}

    for url in tqdm(candidates, desc="validate"):
        ok, reason = _validate_url(url)
        if ok:
            accepted.append(url)
            if len(accepted) >= n:
                break
        else:
            rejected += 1
            reason_key = reason.split(" ")[0] + " " + reason.split(" ")[1] if " " in reason else reason
            reason_counts[reason_key] = reason_counts.get(reason_key, 0) + 1

    print(f"\naccepted: {len(accepted)} / candidates: {len(candidates)}")
    print(f"rejected: {rejected}")
    if reason_counts:
        print("top rejection reasons:")
        for r, c in sorted(reason_counts.items(), key=lambda x: -x[1])[:8]:
            print(f"  {c}× {r}")
    return accepted


def run(out_path: str, n: int = 80, seed: int = 1234, max_candidates: int = 1200) -> None:
    urls = fetch_talk_urls()
    if not urls:
        raise SystemExit("no TED URLs returned - sitemap fetch may have changed")
    chosen = select_validated(urls, n=n, seed=seed, max_candidates=max_candidates)
    if not chosen:
        raise SystemExit("no validated URLs - TED page format may have changed")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text("\n".join(chosen) + "\n")
    print(f"wrote {len(chosen)} URLs -> {out_path}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=80)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--max_candidates", type=int, default=1200)
    args = p.parse_args()
    run(args.out, n=args.n, seed=args.seed, max_candidates=args.max_candidates)
