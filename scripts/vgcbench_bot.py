"""Put vgc-bench's published BC policy on a local Showdown server to accept challenges.

Runs in vgc-bench's own venv, from its checkout (its vocab files are read from ./data):

    cd ~/projects/vgc-bench && .venv/bin/python ../ai-vgc/scripts/vgcbench_bot.py --names vgcbench

Several names accept in parallel, one battle at a time each, so that ai_vgc.nn.player
--challenge can play them concurrently. Species/items/moves/abilities missing from
vgc-bench's vocab (newer than its June 2026 checkpoint) are embedded as "null".
"""

import argparse
import asyncio
import random
import sys
from pathlib import Path

from poke_env import AccountConfiguration, ServerConfiguration
from poke_env.teambuilder import Teambuilder

sys.path.insert(0, str(Path.cwd()))  # the vgc-bench checkout
from vgc_bench.src import policy_player  # noqa: E402
from vgc_bench.src.policy_player import PolicyPlayer  # noqa: E402

AI_VGC = Path(__file__).resolve().parents[1]


class Vocab(list):
    """A vocab list whose unknown entries map to index 0 ("null") instead of raising."""

    missing: set[str] = set()

    def index(self, x, *args):
        try:
            return super().index(x, *args)
        except ValueError:
            if x not in Vocab.missing:
                Vocab.missing.add(x)
                print(f"not in vgc-bench vocab: {x}", flush=True)
            return 0


for name in ("abilities", "items", "moves"):
    setattr(policy_player, name, Vocab(getattr(policy_player, name)))


class PoolTeambuilder(Teambuilder):
    def __init__(self, teams_dir: Path):
        self.teams = [self.join_team(self.parse_showdown_team(p.read_text()))
                      for p in sorted(teams_dir.glob("*.txt"))]

    def yield_team(self) -> str:
        return random.choice(self.teams)


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--names", nargs="+", default=["vgcbench"])
    ap.add_argument("--model", default="results/saves_bc/seed1/100.zip")
    ap.add_argument("--format", default="gen9championsvgc2026regmb")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--teams", type=Path, default=AI_VGC / "data/teams/reg_mb")
    ap.add_argument("--n", type=int, default=1000, help="challenges to accept per name")
    ap.add_argument("--sample", action="store_true", help="sample actions instead of always taking the most likely")
    args = ap.parse_args()

    server = ServerConfiguration(f"ws://localhost:{args.port}/showdown/websocket",
                                 "https://play.pokemonshowdown.com/action.php?")
    policy = None
    players = []
    for name in args.names:
        p = PolicyPlayer(policy=policy, deterministic=not args.sample,
                         account_configuration=AccountConfiguration(name, None), battle_format=args.format,
                         server_configuration=server, team=PoolTeambuilder(args.teams),
                         accept_open_team_sheet=True, max_concurrent_battles=1, log_level=40)
        if policy is None:
            p.set_policy(args.model, "cpu")
            policy = p.policy
        players.append(p)
    print(f"{', '.join(args.names)} accepting {args.format} challenges on port {args.port}", flush=True)
    await asyncio.gather(*(p.accept_challenges(None, args.n) for p in players))
    for p in players:
        print(f"{p.username}: {p.n_won_battles}/{p.n_finished_battles} won", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
