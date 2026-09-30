"""Which team is best when each is piloted by its own specialist? Round robin between (team, model)
entries: every pair plays --series matches, each side on its own team with its own model.

    uv run --extra nn python -m ai_vgc.nn.team_rr --teams data/teams/reg_mc_top \\
        --entries MC196=data/models/mc-bo3-rnad-v6-MC196.pt MC147=data/models/mc-bo3-rnad-v6-MC147.pt ... \\
        --format gen9championsvgc2026regmcbo3 --showdown pokemon-showdown-mc --series 50

Both sides play as the ladder bot does: greedy moves, and in Bo3 games 2+ a preview sampled from
the policy's top --vary-k, changing leads after a loss. --shard i/k plays only every k-th pair
(for Slurm); each copy writes one CSV row per pair, and --sum adds CSVs up into a table.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import itertools
import time
from collections import defaultdict
from pathlib import Path

from poke_env import ServerConfiguration
from poke_env.teambuilder import ConstantTeambuilder

from ai_vgc.nn.player import NNPlayer, load_policy
from ai_vgc.showdown import account, ensure_server, wilson


def summarize(paths: list[Path]) -> None:
    rows = [r for p in paths for r in csv.DictReader(open(p))]
    games, series = defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0])
    h2h: dict[tuple[str, str], float] = {}
    for r in rows:
        a, b = r["team_a"], r["team_b"]
        ga, gb, sa, sn = int(r["games_a"]), int(r["games"]), int(r["series_a"]), int(r["series"])
        for t, w, sw in ((a, ga, sa), (b, gb - ga, sn - sa)):
            games[t][0] += w
            games[t][1] += gb
            series[t][0] += sw
            series[t][1] += sn
        h2h[a, b], h2h[b, a] = sa / max(sn, 1), 1 - sa / max(sn, 1)
    teams = sorted(series, key=lambda t: -series[t][0] / max(series[t][1], 1))
    print(f"{len(rows)} pairs")
    print(f"{'team':>8}  {'series':>16}  {'95% CI':>14}  {'games':>7}")
    for t in teams:
        (w, n), (gw, gn) = series[t], games[t]
        lo, hi = wilson(w, n)
        print(f"{t:>8}  {w:>4}/{n:<4} = {w / max(n, 1):5.1%}  [{lo:.0%}, {hi:.0%}]  {gw / max(gn, 1):6.1%}")
    print("\nseries win rate, row vs column:")
    print(" " * 8 + "".join(f"{t:>8}" for t in teams))
    for a in teams:
        print(f"{a:>8}" + "".join(f"{'-' if a == b else f'{h2h.get((a, b), float('nan')):.0%}':>8}" for b in teams))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entries", nargs="+", default=[], metavar="TEAM=MODEL",
                    help="team file stem in --teams and the model that pilots it")
    ap.add_argument("--teams", type=Path, default=Path("data/teams/reg_mc_top"))
    ap.add_argument("--series", type=int, default=50, help="matches per pair (Bo3 series, or Bo1 games)")
    ap.add_argument("--format", default="gen9championsvgc2026regmcbo3")
    ap.add_argument("--showdown", type=Path, default=None, help="Showdown checkout (pokemon-showdown-mc for M-C)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--rating", type=float, default=1700)
    ap.add_argument("--vary-k", type=int, default=3, help="Bo3 games 2+: preview from the policy's top k")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--closed-sheets", action="store_true", help="both sides decline open team sheets (Bo1 CTS)")
    ap.add_argument("--shard", default="0/1", help="i/k: play pairs i, i+k, i+2k, ...")
    ap.add_argument("--out", type=Path, default=Path("data/team_rr/rr.csv"))
    ap.add_argument("--sum", nargs="+", type=Path, default=None, help="only add up these CSVs and print the table")
    args = ap.parse_args()
    if args.sum:
        summarize(args.sum)
        return

    entries = dict(e.split("=", 1) for e in args.entries)
    i, k = map(int, args.shard.split("/"))
    pairs = list(itertools.combinations(sorted(entries), 2))[i::k]
    models = {}

    def pilot(p: NNPlayer, team: str) -> None:
        """Hand `p` this entry's team and model (cached: each loads once)."""
        path = entries[team]
        if path not in models:
            models[path] = load_policy(path)
        p.model = models[path]
        p._team = ConstantTeambuilder((args.teams / f"{team}.txt").read_text())

    proc = ensure_server(args.port, args.showdown)
    common = dict(battle_format=args.format, max_concurrent_battles=args.concurrency,
                  accept_open_team_sheet=not args.closed_sheets,
                  log_level=40, server_configuration=ServerConfiguration(
                      f"ws://localhost:{args.port}/showdown/websocket", "https://play.pokemonshowdown.com/action.php?"))
    pa, pb = (NNPlayer(load_policy(entries[pairs[0][0]]), args.rating, True, account_configuration=account(n),
                       team=ConstantTeambuilder((args.teams / f"{pairs[0][0]}.txt").read_text()), **common)
              for n in ("rra", "rrb"))
    for p in (pa, pb):
        p.vary, p.vary_k, p.change_after_loss = True, args.vary_k, True
    args.out.parent.mkdir(parents=True, exist_ok=True)
    t = time.time()
    try:
        with open(args.out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["team_a", "team_b", "games_a", "games", "series_a", "series"])
            writer.writeheader()
            for a, b in pairs:
                pilot(pa, a)
                pilot(pb, b)
                asyncio.run(pa.battle_against(pb, n_battles=args.series))
                if pa.games:  # Bo3: count series from the per-game records
                    by = defaultdict(list)
                    for g in pa.games:
                        by[g["series"]].append(g["won"])
                    sa, sn = sum(sum(v) >= 2 for v in by.values()), len(by)
                else:
                    sa, sn = pa.n_won_battles, pa.n_finished_battles
                writer.writerow({"team_a": a, "team_b": b, "games_a": pa.n_won_battles,
                                 "games": pa.n_finished_battles, "series_a": sa, "series": sn})
                f.flush()
                print(f"{a} vs {b}: series {sa}/{sn}, games {pa.n_won_battles}/{pa.n_finished_battles}  "
                      f"{time.time() - t:.0f}s", flush=True)
                for p in (pa, pb):
                    p.reset_battles()
                    p.games.clear()
                    p.last_preview.clear()
    finally:
        if proc:
            proc.terminate()


if __name__ == "__main__":
    main()
