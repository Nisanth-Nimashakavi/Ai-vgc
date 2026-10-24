#!/usr/bin/env python3
"""Showdown Gauntlet Benchmark: Specialist v7f-MC378 vs v6 Policy on Pool Teams.

CTS Bo1 format (gen9championsvgc2026regmc).
- Candidate (Specialist): pilots signature team (MC378) with speed, damage, and set inference.
- Opponent (v6 Generalist): cycles through the pool of 500+ teams in data/teams/reg_mc.
- Side-balanced: half the games as P1, half as P2.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import socket
import time
from pathlib import Path

from poke_env import ServerConfiguration
from poke_env.teambuilder import ConstantTeambuilder

from ai_vgc.nn.player import NNPlayer, configure_inference, load_policy
from ai_vgc.showdown import account, ensure_server, wilson
from ai_vgc.teams import RandomPoolTeambuilder


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


async def run_gauntlet(
    games: int = 500,
    model_path: str = "data/models/mc-cts-rnad-v7f-MC378.pt",
    opp_model_path: str = "data/models/mc-cts-rnad-v6.pt",
    my_team: str = "MC378",
    pool_dir: str = "data/teams/reg_mc",
    concurrency: int = 4,
    device: str = "cpu",
):
    port = find_free_port()
    print("=" * 70)
    print("GAUNTLET BENCHMARK: SPECIALIST (v7f) vs v6 ON POOL TEAMS")
    print("=" * 70)
    print(f"Port:           {port}")
    print(f"Games:          {games} (half P1, half P2)")
    print(f"Specialist:     {model_path} (Team: {my_team})")
    print(f"Opponent:       {opp_model_path} (Pool: {pool_dir})")
    print(f"Concurrency:    {concurrency}")
    print("-" * 70)

    proc = ensure_server(port, Path("pokemon-showdown-mc"))
    t0 = time.time()
    config = ServerConfiguration(f"ws://localhost:{port}/showdown/websocket", "https://play.pokemonshowdown.com/action.php?")
    fmt = "gen9championsvgc2026regmc"

    team_file = Path("data/teams/reg_mc") / f"{my_team}.txt" if not my_team.endswith(".txt") else Path(my_team)
    if not team_file.exists():
        team_file = Path(pool_dir) / f"{my_team}.txt" if not my_team.endswith(".txt") else Path(my_team)
    if not team_file.exists():
        raise FileNotFoundError(f"Team file not found: {team_file}")
    paste_cand = team_file.read_text("utf-8")
    tb_cand = ConstantTeambuilder(paste_cand)
    tb_pool = RandomPoolTeambuilder(Path(pool_dir), cycle=True)

    print(f"Loading Specialist Model from: {model_path} ...")
    m_v7f = load_policy(model_path, device).eval()
    print(f"Loading Opponent Policy from: {opp_model_path} ...")
    m_v6 = load_policy(opp_model_path, device).eval()

    n_p1 = (games + 1) // 2
    n_p2 = games - n_p1

    try:
        # Phase 1: Candidate as P1
        print(f"\n--- Phase 1: Specialist as P1 ({n_p1} games against pool) ---", flush=True)
        cand_p1 = NNPlayer(
            m_v7f, rating=1700, greedy=True, rules=False, team=tb_cand,
            account_configuration=account("v7fgauntp1"), battle_format=fmt,
            accept_open_team_sheet=False, max_concurrent_battles=concurrency,
            server_configuration=config,
        )
        opp_p2 = NNPlayer(
            m_v6, rating=1700, greedy=True, rules=False, team=tb_pool,
            account_configuration=account("v6gauntopp2"), battle_format=fmt,
            accept_open_team_sheet=False, max_concurrent_battles=concurrency,
            server_configuration=config,
        )
        configure_inference(cand_p1, pool_dir, fmt)
        configure_inference(opp_p2, pool_dir, fmt)

        await cand_p1.battle_against(opp_p2, n_battles=n_p1)
        w1, l1 = cand_p1.n_won_battles, cand_p1.n_lost_battles
        print(f"Phase 1 Result: Specialist won {w1} / {w1 + l1} ({w1 / max(1, w1 + l1) * 100:.1f}%) as P1", flush=True)

        # Phase 2: Candidate as P2
        print(f"\n--- Phase 2: Specialist as P2 ({n_p2} games against pool) ---", flush=True)
        opp_p1 = NNPlayer(
            m_v6, rating=1700, greedy=True, rules=False, team=tb_pool,
            account_configuration=account("v6gauntopp1"), battle_format=fmt,
            accept_open_team_sheet=False, max_concurrent_battles=concurrency,
            server_configuration=config,
        )
        cand_p2 = NNPlayer(
            m_v7f, rating=1700, greedy=True, rules=False, team=tb_cand,
            account_configuration=account("v7fgauntp2"), battle_format=fmt,
            accept_open_team_sheet=False, max_concurrent_battles=concurrency,
            server_configuration=config,
        )
        configure_inference(cand_p2, pool_dir, fmt)
        configure_inference(opp_p1, pool_dir, fmt)

        await opp_p1.battle_against(cand_p2, n_battles=n_p2)
        w2, l2 = cand_p2.n_won_battles, cand_p2.n_lost_battles
        print(f"Phase 2 Result: Specialist won {w2} / {w2 + l2} ({w2 / max(1, w2 + l2) * 100:.1f}%) as P2", flush=True)

    finally:
        if proc:
            proc.terminate()

    total_w = w1 + w2
    total_g = (w1 + l1) + (w2 + l2)
    elapsed = time.time() - t0
    lo, hi = wilson(total_w, total_g)

    print("\n" + "=" * 70)
    print(f"FINAL GAUNTLET RESULT: Specialist won {total_w} / {total_g} ({total_w / max(1, total_g) * 100:.1f}%) in {elapsed:.1f}s")
    print(f"95% CI: [{lo * 100:.1f}%, {hi * 100:.1f}%]")
    print(f"P1 win rate: {w1} / {w1 + l1} ({w1 / max(1, w1 + l1) * 100:.1f}%)")
    print(f"P2 win rate: {w2} / {w2 + l2} ({w2 / max(1, w2 + l2) * 100:.1f}%)")
    print("=" * 70, flush=True)

    result_data = {
        "candidate_model": model_path,
        "candidate_team": str(team_file),
        "opponent_model": opp_model_path,
        "opponent_pool": pool_dir,
        "total_games": total_g,
        "wins": total_w,
        "losses": total_g - total_w,
        "win_rate": total_w / max(1, total_g) * 100.0,
        "ci_95": [lo * 100.0, hi * 100.0],
        "p1_win_rate": w1 / max(1, w1 + l1) * 100.0,
        "p2_win_rate": w2 / max(1, w2 + l2) * 100.0,
        "elapsed_s": elapsed,
    }

    out_file = f"showdown_gauntlet_{Path(model_path).stem}_vs_{Path(opp_model_path).stem}_pool_{total_g}g.json"
    with open(out_file, "w") as f:
        json.dump(result_data, f, indent=2)
    print(f"Saved results to {out_file}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=500, help="Total games against pool")
    parser.add_argument("--model", type=str, default="data/models/mc-cts-rnad-v7f-MC378.pt", help="Candidate specialist model")
    parser.add_argument("--opp-model", type=str, default="data/models/mc-cts-rnad-v6.pt", help="Opponent pool model")
    parser.add_argument("--team", type=str, default="MC378", help="Specialist signature team name")
    parser.add_argument("--pool", type=str, default="data/teams/reg_mc", help="Directory of opponent teams")
    parser.add_argument("--concurrency", type=int, default=4, help="Max concurrent battles")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    args = parser.parse_args()

    asyncio.run(run_gauntlet(
        games=args.games,
        model_path=args.model,
        opp_model_path=args.opp_model,
        my_team=args.team,
        pool_dir=args.pool,
        concurrency=args.concurrency,
        device=args.device,
    ))


if __name__ == "__main__":
    main()
