"""Head-to-head: the same searched model/team on both sides, differing only in --opp-model
(the network search uses to guess the opponent's likely reply). Isolates whether a better
opp-model wins more games than a worse one, holding the learner, team, and search config fixed.

    uv run --extra nn python -m ai_vgc.nn.opp_h2h \\
        --model data/models/mc-cts-rnad-v7f-MC301.pt --team data/teams/reg_mc/MC301.txt \\
        --opp-model-a data/models/mc-cts-opp-v6.pt --opp-model-b data/models/mc-cts-opp-v7.pt \\
        --format gen9championsvgc2026regmc --showdown pokemon-showdown-mc --n 30
"""

from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path

from poke_env import ServerConfiguration
from poke_env.teambuilder import ConstantTeambuilder

from ai_vgc.nn.player import configure_inference
from ai_vgc.nn.search import SearchPlayer
from ai_vgc.showdown import account, ensure_server, wilson


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="checkpoint both sides play (same model, same team)")
    ap.add_argument("--team", type=Path, required=True, help="team file both sides use (a mirror match)")
    ap.add_argument("--opp-model-a", required=True)
    ap.add_argument("--opp-model-b", required=True)
    ap.add_argument("--teams-dir", type=Path, default=Path("data/teams/reg_mc"))
    ap.add_argument("--format", default="gen9championsvgc2026regmc")
    ap.add_argument("--showdown", type=Path, default=Path("pokemon-showdown-mc"))
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--port", type=int, default=19500)
    ap.add_argument("--rating", type=float, default=1700)
    ap.add_argument("--search-worlds", type=int, default=4)
    ap.add_argument("--search-seeds", type=int, default=16)
    ap.add_argument("--search-k", type=int, default=4)
    ap.add_argument("--search-opp-k", type=int, default=4)
    ap.add_argument("--search-prior", type=float, default=0.1)
    ap.add_argument("--search-depth", type=int, default=1)
    ap.add_argument("--search-solver", default="lp")
    ap.add_argument("--double-oracle", action="store_true", default=True)
    ap.add_argument("--search-value-device-a", default="cuda:0",
                    help="GPU for side A's search value network (separate from B, so both sides "
                         "searching at once don't contend for the same GPU)")
    ap.add_argument("--search-value-device-b", default="cuda:0")
    args = ap.parse_args()

    proc = ensure_server(args.port, args.showdown)
    team = ConstantTeambuilder(args.team.read_text())
    common = dict(
        max_concurrent_battles=args.concurrency,
        accept_open_team_sheet=False, start_listening=True,
        server_configuration=ServerConfiguration(
            f"ws://localhost:{args.port}/showdown/websocket",
            "https://play.pokemonshowdown.com/action.php?"),
    )
    search_kw = dict(
        fmt=args.format, teams_dir=str(args.teams_dir), showdown=args.showdown,
        k=args.search_k, opp_k=args.search_opp_k, seeds=args.search_seeds, worlds=args.search_worlds,
        depth=args.search_depth, prior=args.search_prior, backend="rust",
        double_oracle=args.double_oracle, solver=args.search_solver, closed_sheets=True,
    )

    a = SearchPlayer(args.model, args.rating, True, opp_model=args.opp_model_a,
                      account_configuration=account("h2ha"), team=team,
                      value_device=args.search_value_device_a, **search_kw, **common)
    b = SearchPlayer(args.model, args.rating, True, opp_model=args.opp_model_b,
                      account_configuration=account("h2hb"), team=team,
                      value_device=args.search_value_device_b, **search_kw, **common)
    for p in (a, b):
        configure_inference(p, args.teams_dir, args.format)

    t0 = time.time()
    asyncio.run(a.battle_against(b, n_battles=args.n))
    wins, n = a.n_won_battles, a.n_finished_battles
    lo, hi = wilson(wins, n)
    print(f"A (opp-model {Path(args.opp_model_a).name}) vs B (opp-model {Path(args.opp_model_b).name}): "
          f"A wins {wins}/{n} = {wins / max(n, 1):.1%}  95% CI [{lo:.1%}, {hi:.1%}]  {time.time() - t0:.0f}s")
    print(f"RESULT wins_a={wins} games={n}")
    if proc is not None:
        proc.terminate()


if __name__ == "__main__":
    main()
