"""How good is one team? Lock it in and play every team in a pool the same number of games.

    uv run --extra nn python -m ai_vgc.nn.team_test data/teams/reg_mc/MC101.txt \\
        --model data/models/mc-bo1-rnad-v3.pt --format gen9championsvgc2026regmc \\
        --showdown pokemon-showdown-mc --pool data/teams/reg_mc --games 4

Both sides are played by the same model (--opponent-model changes theirs), so the result compares
teams, not players: an average team in the pool scores about 50%. It prints the overall win rate
and the worst matchups, and writes one row per opponent team to --out (CSV).

--shard i/k plays only every k-th opponent team, starting at i (for Slurm; add the CSVs up).
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import time
from pathlib import Path

from poke_env import ServerConfiguration
from poke_env.teambuilder import ConstantTeambuilder

from ai_vgc.nn.player import NNPlayer
from ai_vgc.showdown import account, ensure_server, wilson


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("team", type=Path, help="Showdown export of the team to test")
    ap.add_argument("--pool", type=Path, default=Path("data/teams/reg_mb"), help="folder of opponent teams")
    ap.add_argument("--games", type=int, default=4, help="games against each opponent team")
    ap.add_argument("--model", default="data/models/archive/mb-bo3-rnad-v3.pt")
    ap.add_argument("--opponent-model", default=None, help="model playing the pool teams (default: --model)")
    ap.add_argument("--format", default="gen9championsvgc2026regmb")
    ap.add_argument("--showdown", type=Path, default=None, help="Showdown checkout (pokemon-showdown-mc for M-C)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--rating", type=float, default=1700)
    ap.add_argument("--shard", default="0/1", help="i/k: play opponent teams i, i+k, i+2k, ...")
    ap.add_argument("--out", type=Path, default=None, help="CSV (default: data/team_tests/<team>.csv)")
    ap.add_argument("--closed-sheets", action="store_true", help="closed team sheets (CTS Bo1)")
    args = ap.parse_args()

    i, k = map(int, args.shard.split("/"))
    pool = sorted(args.pool.glob("*.txt"))[i::k]
    if not pool:
        raise SystemExit(f"no teams in {args.pool}")
    out = args.out or Path("data/team_tests") / f"{args.team.stem}.csv"
    proc = ensure_server(args.port, args.showdown)
    common = dict(battle_format=args.format, max_concurrent_battles=args.games, accept_open_team_sheet=not args.closed_sheets,
                  log_level=40, server_configuration=ServerConfiguration(
                      f"ws://localhost:{args.port}/showdown/websocket", "https://play.pokemonshowdown.com/action.php?"))
    me = NNPlayer(args.model, args.rating, True, account_configuration=account("tt"),
                  team=ConstantTeambuilder(args.team.read_text()), **common)
    opp = NNPlayer(args.opponent_model or args.model, args.rating, True, account_configuration=account("ttopp"),
                   team=ConstantTeambuilder(pool[0].read_text()), **common)
    rows = []
    t = time.time()
    try:
        for n, path in enumerate(pool, 1):
            opp._team = ConstantTeambuilder(path.read_text())
            before = me.n_won_battles, me.n_finished_battles
            asyncio.run(me.battle_against(opp, n_battles=args.games))
            w, g = me.n_won_battles - before[0], me.n_finished_battles - before[1]
            rows.append({"opponent_team": path.stem, "wins": w, "games": g,
                         "their_mons": " / ".join(line.split("@")[0].split("(")[0].strip()
                                                  for line in path.read_text().split("\n\n") if line.strip())})
            if n % 25 == 0:
                print(f"  {n}/{len(pool)} teams, {time.time() - t:.0f}s", flush=True)
    finally:
        if proc:
            proc.terminate()

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    wins, games = sum(r["wins"] for r in rows), sum(r["games"] for r in rows)
    lo, hi = wilson(wins, games)
    print(f"{args.team.stem} vs {len(rows)} teams in {args.pool}: {wins}/{games} = {wins / max(games, 1):.1%}  "
          f"95% CI [{lo:.1%}, {hi:.1%}]  {time.time() - t:.0f}s -> {out}", flush=True)
    worst = sorted(rows, key=lambda r: r["wins"] / max(r["games"], 1))[:10]
    print("worst matchups:")
    for r in worst:
        print(f"  {r['wins']}/{r['games']}  {r['opponent_team']}: {r['their_mons']}")


if __name__ == "__main__":
    main()
