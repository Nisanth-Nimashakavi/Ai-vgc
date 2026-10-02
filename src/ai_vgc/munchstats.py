"""Per-species usage from MunchStats ([Champions] In-Game Doubles, current month): stat point
spreads and natures (which neither team sheets nor Showdown logs show), plus move, item and
ability usage. Search guesses opposing sets from it (`ai_vgc.nn.search`).

    uv run python -m ai_vgc.munchstats                 # refresh entries older than 7 days (~45 min for all)
    uv run python -m ai_vgc.munchstats --species Rillaboom Incineroar

The ladder bot keeps it fresh by itself: `scripts/bot.sh` starts a refresh in the background,
and a search player fetches any opposing species it has no data on as soon as team preview
shows them (`Refresher`). Search re-reads the file whenever it changes.

Cache: data/teams/munchstats_doubles.json,
    {species id: {"spreads": [["32/32/0/0/0/2", 11.6], ...], "natures": [["Adamant", 84.9], ...],
                  "moves": [...], "items": [...], "abilities": [...], "rank": 1, "fetched": unix time}}
Spreads are Stat Points (0-32 per stat, HP/Atk/Def/SpA/SpD/Spe). Requests are 10 s apart, as the
site's robots.txt asks.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path

BASE = "https://www.munchstats.com/champions/doubles/"
CACHE = Path(__file__).resolve().parents[2] / "data" / "teams" / "munchstats_doubles.json"
UA = "ai-vgc research bot (github.com/nimnim111/ai-vgc)"
DELAY = 10.0
MAX_AGE = 7 * 86400
SECTIONS = {"Moves": "moves", "Items": "items", "Abilities": "abilities",
            "Stat Point Spreads": "spreads", "Natures": "natures"}
_lock = threading.Lock()  # one request at a time per process, DELAY apart
_last = 0.0


def to_id(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _get(url: str) -> str:
    global _last
    with _lock:
        time.sleep(max(0.0, _last + DELAY - time.time()))
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read().decode()
        finally:
            _last = time.time()


def parse(page: str) -> dict:
    out = {}
    parts = re.split(r"<h2>([^<]+)</h2>", page)
    for title, body in zip(parts[1::2], parts[2::2]):
        if title.strip() in SECTIONS:
            rows = re.findall(r'<span class="left-text"[^>]*>([^<]+)</span>.*?<span class="right-text">([\d.]+)%',
                              body, re.S)
            out[SECTIONS[title.strip()]] = [[html.unescape(n).strip(), float(p)] for n, p in rows]
    return out


def load() -> dict:
    try:
        return json.loads(CACHE.read_text())
    except (OSError, ValueError):
        return {}


def _save(entries: dict) -> None:
    """Merge `entries` into the cache (re-read first: another bot may have added some)."""
    data = load() | entries
    tmp = CACHE.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data))
    tmp.replace(CACHE)


def species_list() -> list[tuple[str, int]]:
    """(display name, usage rank) of every species the site lists."""
    page = _get(BASE + "Rillaboom")
    return [(n, int(r)) for n, r in re.findall(r'\{ name: "([^"]+)", usage: "#(\d+)"', page)]


def fetch(name: str, rank: int | None = None) -> dict | None:
    try:
        entry = parse(_get(BASE + urllib.parse.quote(name)))
    except Exception as e:
        print(f"munchstats: {name}: {e!r}", flush=True)
        return None
    if not entry.get("spreads"):
        return None
    entry["fetched"] = time.time()
    if rank is not None:
        entry["rank"] = rank
    _save({to_id(name): entry})
    return entry


def refresh(max_age: float = MAX_AGE, names: list[str] | None = None, log: bool = True) -> int:
    """Fetch every listed species (or `names`) whose entry is missing or older than `max_age`
    seconds, most used first. Returns how many were fetched."""
    data = load()
    todo = [(n, None) for n in names] if names else species_list()
    todo = [(n, r) for n, r in todo if time.time() - data.get(to_id(n), {}).get("fetched", 0) > max_age]
    for i, (name, rank) in enumerate(todo, 1):
        fetch(name, rank)
        if log and (i % 20 == 0 or i == len(todo)):
            print(f"munchstats: {i}/{len(todo)} refreshed", flush=True)
    return len(todo)


_CACHED: tuple[float, dict] = (0.0, {})


def cached() -> dict:
    """The cache, re-read only when the file changes (for per-decision lookups)."""
    global _CACHED
    try:
        mtime = CACHE.stat().st_mtime
    except OSError:
        return {}
    if mtime != _CACHED[0]:
        _CACHED = (mtime, load())
    return _CACHED[1]


def likely_ability(species: str, min_share: float = 0.8) -> str | None:
    """The ability a species runs at least `min_share` of the time (e.g. Raichu: Lightning Rod
    97%), for an opposing Pokemon whose ability hasn't shown yet; None when no single one does."""
    abilities = cached().get(to_id(species), {}).get("abilities") or []
    if abilities and abilities[0][1] >= 100 * min_share:
        return to_id(abilities[0][0])
    return None


class Refresher:
    """Background fetches for a running bot: `need(names)` queues species with no data yet."""

    def __init__(self):
        self._queue: list[str] = []
        self._wanted: set[str] = set()
        self._cv = threading.Condition()
        threading.Thread(target=self._run, daemon=True).start()

    def need(self, names: list[str]) -> None:
        have = load()
        with self._cv:
            for n in names:
                if to_id(n) not in have and to_id(n) not in self._wanted:
                    self._wanted.add(to_id(n))
                    self._queue.append(n)
            self._cv.notify()

    def _run(self) -> None:
        while True:
            with self._cv:
                while not self._queue:
                    self._cv.wait()
                name = self._queue.pop(0)
            fetch(name)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-age", type=float, default=7.0, help="days before an entry is refetched")
    ap.add_argument("--species", nargs="+", default=None, help="only these (display names)")
    args = ap.parse_args()
    n = refresh(args.max_age * 86400, args.species)
    print(f"{CACHE}: {len(load())} species ({n} fetched)")


if __name__ == "__main__":
    main()
