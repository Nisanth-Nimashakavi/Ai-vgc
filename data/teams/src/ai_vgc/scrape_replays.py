"""Download public Showdown replays for one format, politely.

    uv run python -m ai_vgc.scrape_replays                       # Reg M-C Bo3
    uv run python -m ai_vgc.scrape_replays --mode rating         # top-rated only
    uv run python -m ai_vgc.scrape_replays --rate 5 --workers 6  # faster
    uv run python -m ai_vgc.scrape_replays --format gen9championsvgc2026regmcbo3 --max-new 2000

Uses Showdown's JSON API (documented in pokemon-showdown-client/WEB-API.md),
not the HTML pages:
  - search.json?format=F&sort=rating&page=N  -> 51 replays per page, capped at 100 pages
  - search.json?format=F&before=T            -> newest first, older than upload time T
  - <id>.json                                -> one replay, including its log

The output file uses the vgc-battle-logs layout, {battle_id: [uploadtime, log]},
so `ai_vgc.nn.dataset` reads it unchanged. It is saved every `--save-every`
downloads, and replays already in the file are skipped, so the script can be
stopped with Ctrl+C and rerun at any time.

Staying polite (and unbanned):
  - a few connections at once, but one shared rate limit (`--rate` requests per
    second, default 3, i.e. ~10,000 replays/hour);
  - adaptive: any HTTP 429/503 halves the rate and pauses *every* worker (for
    Retry-After, or 30 s doubling up to 10 min); the rate creeps back up by 5%
    per 100 clean requests, never above `--rate`;
  - an honest User-Agent;
  - stop after `--max-failures` failures in a row instead of hammering a broken server;
  - private and password-protected replays are skipped.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

BASE = "https://replay.pokemonshowdown.com"
USER_AGENT = "ai-vgc-research/0.1 (personal VGC AI project; rate-limited, backs off on 429)"
PAGE_SIZE = 50  # the API returns 51 rows; the 51st only signals that another page exists
RATING_PAGE_CAP = 100
MAX_RATE = 10.0  # hard ceiling on requests per second, whatever --rate says


class Client:
    """Thread-safe JSON getter with one shared, adaptive rate limit."""

    def __init__(self, rate: float, max_failures: int):
        self.max_rate = min(rate, MAX_RATE)
        self.rate = self.max_rate
        self.max_failures = max_failures
        self.lock = threading.Lock()
        self.next_slot = 0.0  # monotonic time of the next allowed request
        self.failures = 0
        self.clean = 0
        self.backoff = 30.0
        self.requests = 0

    def _acquire(self) -> None:
        with self.lock:
            now = time.monotonic()
            slot = max(now, self.next_slot)
            gap = 1 / self.rate
            self.next_slot = slot + gap * random.uniform(0.8, 1.2)
            self.requests += 1
        if slot > now:
            time.sleep(slot - now)

    def _ok(self) -> None:
        with self.lock:
            self.failures = 0
            self.clean += 1
            if self.clean >= 100:
                self.clean = 0
                self.backoff = 30.0
                self.rate = min(self.max_rate, self.rate * 1.05)

    def _fail(self, msg: str, pause: float | None, throttled: bool) -> None:
        with self.lock:
            self.failures += 1
            self.clean = 0
            if self.failures >= self.max_failures:
                raise SystemExit(f"giving up after {self.failures} failures in a row ({msg})")
            pause = pause if pause is not None else self.backoff
            self.backoff = min(self.backoff * 2, 600)
            if throttled:
                self.rate = max(0.3, self.rate / 2)
            # Pause every worker, not just this one.
            self.next_slot = max(self.next_slot, time.monotonic() + pause)
            print(f"  {msg}; pausing {pause:.0f}s, rate now {self.rate:.2f}/s", file=sys.stderr)

    def get_json(self, url: str):
        """GET a JSON URL, rate limited, with retries. Returns None on a 404."""
        while True:
            self._acquire()
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    body = resp.read()
                self._ok()
                # Showdown sometimes prefixes JSON with "]" to prevent JSON hijacking.
                return json.loads(body.decode().lstrip("]"))
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    self._ok()
                    return None
                retry_after = e.headers.get("Retry-After") if e.headers else None
                pause = float(retry_after) if retry_after and retry_after.isdigit() else None
                self._fail(f"HTTP {e.code} for {url}", pause, throttled=e.code in (429, 503))
            except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError) as e:
                self._fail(f"{type(e).__name__}: {e} for {url}", None, throttled=True)


def search_by_rating(client: Client, fmt: str, max_pages: int):
    for page in range(1, min(max_pages, RATING_PAGE_CAP) + 1):
        q = urllib.parse.urlencode({"format": fmt, "sort": "rating", "page": page})
        rows = client.get_json(f"{BASE}/search.json?{q}") or []
        yield from rows[:PAGE_SIZE]
        if len(rows) <= PAGE_SIZE:
            return


def search_by_date(client: Client, fmt: str, max_pages: int, stop_ids: set[str]):
    """Newest first. Stops at the end, at max_pages, or on a page that is all known."""
    before = None
    for _ in range(max_pages):
        params = {"format": fmt}
        if before is not None:
            params["before"] = before
        rows = client.get_json(f"{BASE}/search.json?{urllib.parse.urlencode(params)}") or []
        page = rows[:PAGE_SIZE]
        yield from page
        if len(rows) <= PAGE_SIZE or (stop_ids and all(r["id"] in stop_ids for r in page)):
            return
        before = page[-1]["uploadtime"]


def save(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data))
    os.replace(tmp, path)  # atomic, so Ctrl+C never leaves a half-written file


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--format", default="gen9championsvgc2026regmcbo3")
    ap.add_argument("--mode", choices=["rating", "date", "both"], default="both",
                    help="rating: top ~5,000 by rating; date: every public replay, newest first")
    ap.add_argument("--out", type=Path, default=None, help="default: data/battle_logs/logs_<format>.json")
    ap.add_argument("--rate", type=float, default=3.0,
                    help=f"max requests per second across all workers (capped at {MAX_RATE:g})")
    ap.add_argument("--workers", type=int, default=4, help="concurrent downloads")
    ap.add_argument("--max-pages", type=int, default=10_000, help="search pages per mode")
    ap.add_argument("--max-new", type=int, default=None, help="stop after this many new replays")
    ap.add_argument("--min-rating", type=int, default=0, help="skip replays rated below this (unrated = 0)")
    ap.add_argument("--save-every", type=int, default=200)
    ap.add_argument("--max-failures", type=int, default=8)
    ap.add_argument("--full-walk", action="store_true",
                    help="date mode: keep going past pages that are already downloaded")
    args = ap.parse_args()

    out = args.out or Path(f"data/battle_logs/logs_{args.format}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    data: dict = json.loads(out.read_text()) if out.exists() else {}
    known = set(data)
    client = Client(args.rate, args.max_failures)
    print(f"{out}: {len(data)} replays already downloaded; up to {client.rate:g} requests/s, "
          f"{args.workers} workers")

    def candidates():
        seen: set[str] = set()
        if args.mode in ("rating", "both"):
            for r in search_by_rating(client, args.format, args.max_pages):
                if r["id"] not in seen:
                    seen.add(r["id"])
                    yield r
        if args.mode in ("date", "both"):
            stop = set() if args.full_walk else known
            for r in search_by_date(client, args.format, args.max_pages, stop):
                if r["id"] not in seen:
                    seen.add(r["id"])
                    yield r

    def fetch(r: dict):
        replay = client.get_json(f"{BASE}/{r['id']}.json")
        if not replay or not replay.get("log"):
            return r["id"], None
        return r["id"], [replay.get("uploadtime", r["uploadtime"]), replay["log"]]

    new = skipped = 0
    start = time.monotonic()
    pool = ThreadPoolExecutor(args.workers)
    pending: set = set()

    def collect(done) -> None:
        nonlocal new, skipped
        for f in done:
            rid, entry = f.result()
            if entry is None:
                skipped += 1
                continue
            data[rid] = entry
            new += 1
            if new % args.save_every == 0:
                save(out, data)
                rate = new / (time.monotonic() - start) * 3600
                print(f"  {new} new ({len(data)} total), {client.requests} requests, "
                      f"~{rate:.0f} replays/hour, rate {client.rate:.2f}/s")

    try:
        for r in candidates():
            if args.max_new and new + len(pending) >= args.max_new:
                break
            if r["id"] in data:
                continue
            if r.get("private") or r.get("password") or (r.get("rating") or 0) < args.min_rating:
                skipped += 1
                continue
            pending.add(pool.submit(fetch, r))
            # Keep the queue short so Ctrl+C and --max-new stop promptly.
            if len(pending) >= args.workers * 2:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                collect(done)
        done, pending = wait(pending)
        collect(done)
    except KeyboardInterrupt:
        print("\ninterrupted, saving what finished")
        for f in pending:
            f.cancel()
        collect([f for f in pending if f.done() and not f.cancelled() and f.exception() is None])
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
        save(out, data)
    print(f"done: {new} new, {skipped} skipped, {len(data)} total in {out}")


if __name__ == "__main__":
    main()
