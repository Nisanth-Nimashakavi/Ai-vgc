"""Do a human's moves in earlier games of a Bo3 predict their moves in later games better than
the opponent-action head alone? (idea 8, opponent-conditioned search)

    uv run --extra nn python scripts/opp_habits.py data/models/all-bo3-opp-v2.pt data/battle_logs/logs_*bo3.json

For games 2 and 3 of held-out series (the same 5% game split as training), each opposing active's
move is predicted by the head, and by the head blended with that Pokemon's move counts from the
earlier games of the series (on turn 1, earlier turn-1 moves only): mixed in as frequencies, or
as a boost to the moves it used. Mixtures apply only when it used a move earlier. Rows: turn 1 vs later turns, and whether the Pokemon has history.
"""
import argparse
import json
import re
import sys
import zlib
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from poke_env import to_id_str

from ai_vgc.nn.encode import encode
from ai_vgc.nn.player import load_policy
from ai_vgc.nn.replay import player_info, replay

GAME = re.compile(r'Game (\d)</strong> of <a href="/(game-bestof\d-[^"]+)"')
MOVE = re.compile(r"^\|move\|(p[12])[ab]: ([^|]+)\|([^|]+)\|(.*)$", re.M)
# (label, blend): mixtures p' = (1 - w) p + w h, and boosts p' ∝ p (1 + count)^b.
BLENDS = [("head", lambda p, h, hn: p)] + \
    [(f"mix {w}", lambda p, h, hn, w=w: (1 - w) * p + w * hn) for w in (0.1, 0.3)] + \
    [(f"boost {b}", lambda p, h, hn, b=b: p * (1 + h) ** b) for b in (0.25, 0.5, 1.0, 2.0)]


def usage(log: str) -> dict[str, Counter]:
    """Per player name: Counter of (nickname, move id) chosen this game, and ("t1", nickname,
    move id) for turn 1 only."""
    names = {r: player_info(log, r)[0] for r in ("p1", "p2")}
    t2 = log.find("|turn|2\n")
    out = defaultdict(Counter)
    for m in MOVE.finditer(log):
        role, nick, move, rest = m.groups()
        if "[from]" not in rest:
            out[names[role]][(nick, to_id_str(move))] += 1
            if t2 < 0 or m.start() < t2:
                out[names[role]][("t1", nick, to_id_str(move))] += 1
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--limit", type=int, default=0, help="later games to replay (0 = all)")
    args = ap.parse_args()
    model = load_policy(args.model)

    series = defaultdict(dict)
    for path in args.logs:
        for tag, (_, log) in json.load(open(path)).items():
            g = GAME.search(log)
            if g and "|win|" in log:
                series[g.group(2)][int(g.group(1))] = (tag, log)

    obs, labels, habit, meta = [], [], [], []  # meta: (turn 1?, has history)
    done = 0
    for games in series.values():
        for n in sorted(games):
            tag, log = games[n]
            if n == 1 or zlib.crc32(tag.encode()) % 1000 >= args.val_frac * 1000:
                continue
            before = [usage(games[m][1]) for m in range(1, n) if m in games]
            for role in ("p1", "p2"):
                them = player_info(log, "p2" if role == "p1" else "p1")[0]
                counts = sum((u.get(them, Counter()) for u in before), Counter())
                rating = player_info(log, role)[1]

                def cb(battle, action, preview, opp, rating=rating, counts=counts):
                    if preview or (opp[:, 0] < 0).all():
                        return
                    h = np.zeros((2, 5), np.float32)
                    for s, mon in enumerate(battle.opponent_active_pokemon):
                        if mon is not None:
                            for i, mid in enumerate(list(mon.moves)[:4]):
                                # Turn 1 is matched against earlier turn-1 choices only.
                                h[s, i] = counts[("t1", mon.name, mid) if battle.turn == 1 else (mon.name, mid)]
                    obs.append(encode(battle, rating))
                    labels.append(opp[:, 0].astype(np.int64))
                    habit.append(h)
                    meta.append((battle.turn == 1, h.sum(-1) > 0))

                try:
                    replay(tag, log, role, cb)
                except Exception:
                    continue
            done += 1
            if done % 200 == 0:
                print(f"  {done} games, {len(obs)} decisions", file=sys.stderr, flush=True)
            if args.limit and done >= args.limit:
                break
        if args.limit and done >= args.limit:
            break

    probs = []
    with torch.no_grad():
        for i in range(0, len(obs), 512):
            chunk = obs[i:i + 512]
            b = {k: torch.from_numpy(np.stack([o[k] for o in chunk])) for k in chunk[0]}
            mv, _ = model.opp_logits(model.encode(b), b)
            probs.append(F.softmax(mv, -1).numpy())
    p = np.concatenate(probs)                       # [N, 2, 5]
    y = np.stack(labels)                            # [N, 2]
    h = np.stack(habit)
    hist = np.stack([m[1] for m in meta])           # [N, 2]
    t1 = np.array([m[0] for m in meta])[:, None].repeat(2, 1)
    hn = np.where(h.sum(-1, keepdims=True) > 0, h / np.maximum(h.sum(-1, keepdims=True), 1), p)
    lab = y >= 0
    yc = y.clip(min=0)

    print(f"{done} later games, {lab.sum()} labelled slot actions; "
          f"{(hist & lab).mean() / lab.mean():.0%} of them on a Pokemon with earlier-game history")
    print(f"{'rows':<16} {'n':>6}  " + "  ".join(f"{name:>15}" for name, _ in BLENDS))
    for name, sel in [("all", lab), ("with history", lab & hist), ("  turn 1", lab & hist & t1),
                      ("  turn 2+", lab & hist & ~t1), ("no history", lab & ~hist)]:
        cells = []
        for _, blend in BLENDS:
            q = blend(p, h, hn)
            q = q / q.sum(-1, keepdims=True)
            pick = np.take_along_axis(q, yc[..., None], -1)[..., 0]
            top1 = q.argmax(-1) == yc
            cells.append(f"{top1[sel].mean():6.1%} / {-np.log(pick[sel] + 1e-9).mean():.3f}")
        print(f"{name:<16} {sel.sum():>6}  " + "  ".join(cells))
    print("cells: top-1 accuracy / mean negative log-likelihood (lower is better)")


if __name__ == "__main__":
    main()
