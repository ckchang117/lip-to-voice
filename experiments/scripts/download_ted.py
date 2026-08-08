"""Download TED talks + transcripts directly from TED.

For each talk URL:
  1. GET the talk page, parse `__NEXT_DATA__` for playerData
  2. Extract the talk numeric id and direct h264 MP4 URL
  3. curl the MP4 to /vol/ted_dl/<slug>/talk.mp4
  4. POST to https://graphql.ted.com/ for the English transcript
  5. Save transcript JSON to /vol/ted_dl/<slug>/talk.transcript.json

Bypasses yt-dlp and YouTube entirely - TED talks now embed YouTube behind
anti-bot, but the direct CDN URLs in __NEXT_DATA__ still work, and TED's
GraphQL exposes human-curated English transcripts with cue-level timestamps
(in milliseconds). No Whisper needed downstream.
"""

from __future__ import annotations

import json
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from tqdm import tqdm


_BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) "
    "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15"
)
_GRAPHQL_URL = "https://graphql.ted.com/"
_TRANSCRIPT_QUERY = """
query($id: ID!, $lang: String!) {
  translation(videoId: $id, language: $lang) {
    id
    paragraphs {
      cues {
        startTime
        endTime
        time
        text
      }
    }
  }
}
""".strip()


def _talk_id_from_url(url: str) -> str:
    m = re.search(r"/talks/([^/?#]+)", url)
    if not m:
        return re.sub(r"[^A-Za-z0-9_-]", "_", url)[-40:]
    return m.group(1)


def _extract_video_info(talk_html: str) -> tuple[str | None, str | None]:
    """Return (graphql_video_id, mp4_url). Either may be None on failure.

    Note: there are two IDs in the page. `videoData.id` is the GraphQL Video
    entity id (needed for translation queries); `playerData.id` is a different
    internal media identifier. Earlier code mixed these up and queried GraphQL
    with playerData.id, which silently returned the wrong record (different
    talk entirely).
    """
    m = re.search(
        r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>',
        talk_html,
        re.DOTALL,
    )
    if not m:
        return None, None
    try:
        data = json.loads(m.group(1))
        talk = data["props"]["pageProps"]["videoData"]
        graphql_id = talk.get("id")
        player_data = json.loads(talk["playerData"])
        h264 = player_data.get("resources", {}).get("h264", [])
        if not h264:
            return graphql_id, None
        mp4_url = sorted(h264, key=lambda r: r.get("bitrate", 0), reverse=True)[0]["file"]
        return graphql_id, mp4_url
    except (KeyError, ValueError, json.JSONDecodeError):
        return None, None


def _fetch_transcript(video_id: str, language: str = "en") -> dict | None:
    """Hit TED's GraphQL endpoint for the transcript. Returns dict with cues or None."""
    try:
        resp = requests.post(
            _GRAPHQL_URL,
            headers={"Content-Type": "application/json", "client-id": "Zenith production"},
            json={
                "query": _TRANSCRIPT_QUERY,
                "variables": {"id": str(video_id), "lang": language},
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        if "errors" in data and data["errors"]:
            return None
        translation = data.get("data", {}).get("translation")
        if not translation:
            return None
        # Flatten all paragraph.cues into a single list (preserve order)
        cues = [c for p in translation.get("paragraphs", []) for c in p.get("cues", [])]
        return {
            "id": translation.get("id"),
            "language": language,
            "cues": cues,  # each: {"startTime", "endTime", "time", "text"} all in ms
        }
    except Exception:
        return None


def download_one(url: str, dl_root: Path, sleep_interval: float = 1.0) -> tuple[str, bool, str]:
    talk_id = _talk_id_from_url(url)
    out_dir = dl_root / talk_id
    out_dir.mkdir(parents=True, exist_ok=True)
    mp4_path = out_dir / "talk.mp4"
    transcript_path = out_dir / "talk.transcript.json"

    have_mp4 = mp4_path.exists() and mp4_path.stat().st_size > 0
    have_transcript = transcript_path.exists() and transcript_path.stat().st_size > 0
    if have_mp4 and have_transcript:
        return talk_id, True, "cached"

    # Fetch page once for both MP4 URL and numeric talk id
    try:
        page = requests.get(url, headers={"User-Agent": _BROWSER_UA}, timeout=30)
        page.raise_for_status()
    except Exception as e:
        return talk_id, False, f"page fetch failed: {e}"

    graphql_id, mp4_url = _extract_video_info(page.text)
    if graphql_id is None and mp4_url is None:
        return talk_id, False, "no playerData in __NEXT_DATA__"
    if mp4_url is None:
        return talk_id, False, "no h264 MP4"

    # Download MP4
    if not have_mp4:
        cmd = [
            "curl", "-sSL", "--fail",
            "-A", _BROWSER_UA,
            "--max-time", "1200",
            "-o", str(mp4_path),
            mp4_url,
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=1500)
            if result.returncode != 0:
                return talk_id, False, f"curl exit {result.returncode}: {result.stderr[:200].decode(errors='replace')}"
        except subprocess.TimeoutExpired:
            return talk_id, False, "curl timeout"
        if not mp4_path.exists() or mp4_path.stat().st_size == 0:
            return talk_id, False, "mp4 missing/empty after curl"

    # Fetch transcript via TED GraphQL using the correct Video entity id.
    if not have_transcript and graphql_id:
        transcript = _fetch_transcript(str(graphql_id), language="en")
        if transcript and transcript.get("cues"):
            transcript_path.write_text(json.dumps(transcript, ensure_ascii=False))
        else:
            return talk_id, False, "no transcript available"

    time.sleep(sleep_interval)
    return talk_id, True, "ok"


def run(url_list_path: str, dl_root: str, max_parallel: int = 4, sleep_interval: float = 1.0) -> None:
    urls = [u.strip() for u in Path(url_list_path).read_text().splitlines() if u.strip()]
    print(f"loaded {len(urls)} URLs from {url_list_path}")
    dl_path = Path(dl_root)
    dl_path.mkdir(parents=True, exist_ok=True)

    successes: list[str] = []
    failures: list[tuple[str, str]] = []

    if max_parallel <= 1:
        for url in tqdm(urls, desc="download"):
            tid, ok, msg = download_one(url, dl_path, sleep_interval=sleep_interval)
            if ok:
                successes.append(tid)
            else:
                failures.append((tid, msg))
                tqdm.write(f"  FAIL {tid}: {msg}")
    else:
        with ThreadPoolExecutor(max_workers=max_parallel) as ex:
            futures = [ex.submit(download_one, url, dl_path, sleep_interval) for url in urls]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="download"):
                tid, ok, msg = fut.result()
                if ok:
                    successes.append(tid)
                else:
                    failures.append((tid, msg))
                    tqdm.write(f"  FAIL {tid}: {msg}")

    print(f"\nsuccess: {len(successes)} / {len(urls)}")
    print(f"failures: {len(failures)}")
    if failures:
        print("first 10 failures:")
        for tid, msg in failures[:10]:
            print(f"  {tid}: {msg}")

    summary_path = dl_path / "download_summary.txt"
    summary_path.write_text("\n".join(successes) + "\n")
    print(f"wrote {len(successes)} successful talk_ids -> {summary_path}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--url_list", required=True)
    p.add_argument("--dl_root", required=True)
    p.add_argument("--max_parallel", type=int, default=4)
    p.add_argument("--sleep_interval", type=float, default=1.0)
    args = p.parse_args()
    run(args.url_list, args.dl_root, max_parallel=args.max_parallel, sleep_interval=args.sleep_interval)
