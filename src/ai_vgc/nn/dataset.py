"""Build the imitation-learning dataset from Showdown logs.

    uv run python -m ai_vgc.nn.dataset                 # all logs in data/battle_logs
    uv run python -m ai_vgc.nn.dataset --limit 500     # quick test
    uv run python -m ai_vgc.nn.dataset --logs data/bot_logs --out data/nn_bot   # the bot's own games

Every game is replayed from both players' views. Each decision stores the
encoded state (see `encode.py`), the action per slot, the player's rating,
whether they won, the game id (for train/test splits), how many turns were
left, which feeds the value head, and what the opponent's two actives did that
turn (`opp`, for the auxiliary opponent-action head; see `replay.opp_labels`).
Games 2-3 of a Bo3 also carry what both players did earlier in the series (`series.py`).

--closed (closed team sheets, as in Bo1 CTS): each view drops the opponent's |showteam| line, so
their items, abilities and moves are only known once revealed. The player's own sheet stays: it's
what they see in-game. It also drops the series context and keeps game 1s only, which play like a
Bo1: `uv run python -m ai_vgc.nn.dataset --closed --out data/nn_cts`.

A slot's label is -1 (ignored in the loss) when the log doesn't show what
the player chose: it flinched, slept, was fully paralyzed, or fainted first.

Output: data/nn/<format>.npz, one file per format.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

LOGS = Path("data/battle_logs")
OUT = Path("data/nn")
KEYS = ["tok_cat", "tok_num", "mv_cat", "mv_dyn", "dmg", "glob", "act_tok", "mask",
        "ser_tok", "ser_mv", "ser_glob"]


CLOSED = False  # set in each worker by build(closed=True)


def _closed_init() -> None:
    global CLOSED
    CLOSED = True


def _game(item: tuple[str, str, list[str]]) -> dict[str, np.ndarray] | None:
    """One game from both views; `item` = (tag, log, logs of the series' earlier games)."""
    from ai_vgc.nn.encode import encode, joint_mask
    from ai_vgc.nn.replay import player_info, replay, winner
    from ai_vgc.nn.series import Context

    tag, log, earlier = item
    try:
        win = winner(log)
    except Exception:
        return None
    if win is None:
        return None
    rows: dict[str, list] = {k: [] for k in KEYS + ["action", "preview", "turn", "opp"]}
    per_role = []
    for role in ("p1", "p2"):
        name, rating = player_info(log, role)
        them = player_info(log, "p2" if role == "p1" else "p1")[0]
        ctx = Context.of(earlier, name, them)
        start = len(rows["action"])
        other = "p2" if role == "p1" else "p1"
        view = "\n".join(x for x in log.split("\n") if not x.startswith(f"|showteam|{other}|")) if CLOSED else log

        def cb(battle, action, preview, opp, rating=rating, ctx=ctx):
            battle._series = ctx
            e = encode(battle, rating)
            a = [int(action[0]), int(action[1])]
            if not preview:
                masks = [e["mask"][0], joint_mask(e["mask"][1], a[0])]
                a = [x if 0 <= x < masks[s].shape[0] and masks[s][x] else -1 for s, x in enumerate(a)]
            for k in KEYS:
                rows[k].append(e[k])
            rows["action"].append(a)
            rows["preview"].append(preview)
            rows["turn"].append(battle.turn)
            rows["opp"].append(opp)

        try:
            replay(tag, view, role, cb)
        except Exception:
            for k in rows:
                del rows[k][start:]
            continue
        per_role.append((start, len(rows["action"]), rating, name == win))
    if not rows["action"]:
        return None
    n = len(rows["action"])
    out = {k: np.stack(rows[k]) for k in KEYS}
    out["action"] = np.asarray(rows["action"], np.int16)
    out["preview"] = np.asarray(rows["preview"], bool)
    out["opp"] = np.stack(rows["opp"]).astype(np.int8)
    turns = np.asarray(rows["turn"], np.int16)
    out["rating"] = np.zeros(n, np.int16)
    out["won"] = np.zeros(n, bool)
    out["turns_left"] = np.zeros(n, np.int16)
    for s, e, rating, won in per_role:
        out["rating"][s:e] = rating
        out["won"][s:e] = won
        out["turns_left"][s:e] = turns[s:e].max(initial=0) - turns[s:e]
    out["game"] = np.full(n, zlib.crc32(tag.encode()), np.uint32)
    return out


def _chunk(items: list[tuple[str, str, list[str]]]) -> dict[str, np.ndarray] | None:
    parts = [g for g in map(_game, items) if g is not None]
    if not parts:
        return None
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def build(path: Path, workers: int, limit: int | None, chunk: int = 100, out_dir: Path = OUT,
          closed: bool = False) -> None:
    from ai_vgc.nn.series import series_key

    logs = json.loads(path.read_text())
    series: dict[str, dict[int, str]] = {}
    for v in logs.values():
        if key := series_key(v[1]):
            series.setdefault(key[0], {})[key[1]] = v[1]
    items = []
    for tag, v in logs.items():
        key = series_key(v[1])
        games = series.get(key[0], {}) if key else {}
        earlier = [games[n] for n in range(1, key[1]) if n in games] if key else []
        if closed:  # Bo1-like: game 1s only (or games outside a series), no context
            if not key or key[1] == 1:
                items.append((tag, v[1], []))
            continue
        # A series missing an earlier game (22% on M-C) would mislabel the game number: no context.
        items.append((tag, v[1], earlier if key and len(earlier) == key[1] - 1 else []))
    del logs, series
    if limit:
        items = items[:limit]
    chunks = [items[i:i + chunk] for i in range(0, len(items), chunk)]
    fmt = path.stem.removeprefix("logs_")
    t = time.time()
    parts = []
    # "spawn", not fork: poke-env runs battles on a background event-loop thread,
    # and a forked worker inherits the loop without the thread, so it hangs.
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=_closed_init if closed else None) as pool:
        for i, part in enumerate(pool.map(_chunk, chunks), 1):
            if part is not None:
                parts.append(part)
            if i % 50 == 0 or i == len(chunks):
                n = sum(len(p["action"]) for p in parts)
                print(f"  {fmt}: {i}/{len(chunks)} chunks, {n} decisions, {time.time() - t:.0f}s", flush=True)
    data = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{fmt}.npz"
    np.savez(out, **data)
    act = data["action"][~data["preview"]]
    print(f"{out}: {len(data['action'])} decisions from {len(np.unique(data['game']))} games; "
          f"{(act == -1).mean():.1%} of turn slots unlabeled; {out.stat().st_size / 1e9:.1f} GB")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--limit", type=int, default=None, help="games per format (for testing)")
    ap.add_argument("--formats", nargs="*", default=None, help="e.g. gen9championsvgc2026regmcbo3")
    ap.add_argument("--logs", type=Path, default=LOGS, help="folder of logs_<format>.json")
    ap.add_argument("--out", type=Path, default=OUT, help="folder for the .npz files")
    ap.add_argument("--closed", action="store_true",
                    help="closed team sheets: hide the opponent's sheet, game 1s only, no series context")
    args = ap.parse_args()
    from ai_vgc.nn.encode import vocab
    vocab()  # write the vocab file once, before workers race to create it
    for path in sorted(args.logs.glob("logs_*.json")):
        if args.formats and path.stem.removeprefix("logs_") not in args.formats:
            continue
        build(path, args.workers, args.limit, out_dir=args.out, closed=args.closed)


if __name__ == "__main__":
    main()
