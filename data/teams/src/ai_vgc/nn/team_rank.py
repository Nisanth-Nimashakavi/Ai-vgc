"""Rank a pool of teams: each team plays --games games against random teams from the pool.

    uv run --extra nn python -m ai_vgc.nn.team_rank --pool data/teams/reg_mc --games 100 \\
        --model data/models/mc-bo1-rnad-v3.pt --format gen9championsvgc2026regmc --showdown pokemon-showdown-mc

Both sides are played by the same model (--opponent-model changes theirs), so an average team
scores about 50%. Writes one row per team (team, wins, games, mons) to --out. --shard i/k ranks
only every k-th team (for Slurm); `scripts/team_sum.py` adds the shards up, sorts them and can
copy the best teams into their own folder. --teams narrows the teams being ranked (e.g. a second,
longer round on the top 40) while they still play the whole --pool.
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
from ai_vgc.showdown import account, ensure_server
from ai_vgc.teams import RandomPoolTeambuilder


def mons(path: Path) -> str:
    return " / ".join(block.split("\n")[0].split("@")[0].split("(")[0].strip()
                      for block in path.read_text().split("\n\n") if block.strip())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", type=Path, default=Path("data/teams/reg_mb"), help="opponent teams")
    ap.add_argument("--teams", type=Path, default=None, help="teams to rank (default: --pool)")
    ap.add_argument("--games", type=int, default=100, help="games per ranked team")
    ap.add_argument("--model", default="data/models/archive/mb-bo3-rnad-v3.pt")
    ap.add_argument("--opponent-model", default=None, help="model playing the pool teams (default: --model)")
    ap.add_argument("--format", default="gen9championsvgc2026regmb")
    ap.add_argument("--showdown", type=Path, default=None, help="Showdown checkout (pokemon-showdown-mc for M-C)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--rating", type=float, default=1700)
    ap.add_argument("--shard", default="0/1", help="i/k: rank teams i, i+k, i+2k, ...")
    ap.add_argument("--out", type=Path, required=True, help="CSV to write")
    args = ap.parse_args()

    i, k = map(int, args.shard.split("/"))
    ranked = sorted((args.teams or args.pool).glob("*.txt"))[i::k]
    if not ranked:
        raise SystemExit(f"no teams in {args.teams or args.pool}")
    proc = ensure_server(args.port, args.showdown)
    common = dict(battle_format=args.format, max_concurrent_battles=args.concurrency, accept_open_team_sheet=True,
                  log_level=40, server_configuration=ServerConfiguration(
                      f"ws://localhost:{args.port}/showdown/websocket", "https://play.pokemonshowdown.com/action.php?"))
    me = NNPlayer(args.model, args.rating, True, account_configuration=account("tr"),
                  team=ConstantTeambuilder(ranked[0].read_text()), **common)
    opp = NNPlayer(args.opponent_model or args.model, args.rating, True, account_configuration=account("tropp"),
                   team=RandomPoolTeambuilder(args.pool, seed=i), **common)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    t = time.time()
    try:
        with open(args.out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["team", "wins", "games", "mons"])
            writer.writeheader()
            for n, path in enumerate(ranked, 1):
                me._team = ConstantTeambuilder(path.read_text())
                before = me.n_won_battles, me.n_finished_battles
                asyncio.run(me.battle_against(opp, n_battles=args.games))
                w, g = me.n_won_battles - before[0], me.n_finished_battles - before[1]
                writer.writerow({"team": path.stem, "wins": w, "games": g, "mons": mons(path)})
                f.flush()  # a killed job keeps the teams it finished
                print(f"{n}/{len(ranked)} {path.stem}: {w}/{g}  {time.time() - t:.0f}s", flush=True)
    finally:
        if proc:
            proc.terminate()


if __name__ == "__main__":
    main()
