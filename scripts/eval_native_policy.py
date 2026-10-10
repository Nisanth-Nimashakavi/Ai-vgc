#!/usr/bin/env python3
"""Evaluate two frozen policies entirely in the native VGC engine.

This is a deterministic, side-balanced *training-simulator* evaluator. It is
useful for PFSP diagnostics and regression detection, but it is not a ladder
promotion benchmark: evaluate candidates in Pokémon Showdown as well.

Example:
    VGC_DEX_DIR=$HOME/Dev/localdex/data uv run --extra nn python \
      scripts/eval_native_policy.py \
      --learner data/models/candidate.pt \
      --opponent data/models/anchor.pt \
      --learner-team data/teams/reg_mb/MB100.txt \
      --teams data/teams/reg_mb --games 512 --slots 64 --device cuda
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from ai_vgc.nn.player import load_policy
from ai_vgc.nn.vgc_env import NativeBatchSelfPlay, NativeVGCConfig


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--learner", required=True, help="candidate checkpoint")
    ap.add_argument("--opponent", required=True, help="frozen anchor checkpoint")
    ap.add_argument("--teams", required=True, help="opponent team pool")
    ap.add_argument("--learner-team", required=True, help="fixed candidate team export")
    ap.add_argument("--opponent-teams", default=None,
                    help="separate opponent team pool (default: --teams)")
    ap.add_argument("--format", default="gen9championsvgc2026regmb")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--games", type=int, default=512,
                    help="completed native games to target; rounded up to whole horizons")
    ap.add_argument("--slots", type=int, default=64)
    ap.add_argument("--horizon", type=int, default=128)
    ap.add_argument("--max-turns", type=int, default=128)
    ap.add_argument("--team-preview", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch reports no CUDA device")
    if args.games < 1 or args.slots < 1 or args.horizon < 1:
        raise ValueError("--games, --slots, and --horizon must be positive")

    learner = load_policy(args.learner, args.device).eval()
    opponent = load_policy(args.opponent, args.device).eval()
    config = NativeVGCConfig(
        fmt=args.format,
        teams_dir=args.teams,
        learner_team=args.learner_team,
        opponent_teams_dir=args.opponent_teams,
        greedy=True,
        max_turns=args.max_turns,
        team_preview=args.team_preview,
        seed=args.seed,
    )
    total_games = 0
    total_reward = 0.0
    p1_games = p2_games = 0
    p1_reward = p2_reward = 0.0
    total_turns = 0.0
    horizons = 0
    started = time.perf_counter()

    while total_games < args.games:
        # A collection horizon intentionally drops unfinished PPO tails. Start
        # the next evaluator horizon from a new deterministic stream instead
        # of accidentally replaying identical matchups from a fresh runner.
        config.seed = args.seed + horizons * args.slots * 1_000_003
        runner = NativeBatchSelfPlay(config)
        data, stats = runner.collect_persistent_league_data(
            learner, opponent, args.slots, args.horizon,
            learner_first=(horizons * args.slots // 2) % 2 == 0,
        )
        completed = int(stats["completed_games"])
        horizons += 1
        if not completed:
            continue
        terminal = data["reward"][data["last"]]
        if len(terminal) != completed:
            raise RuntimeError("native evaluator terminal rows disagree with completed-game count")
        total_games += completed
        total_reward += float(terminal.sum())
        p1_games += int(stats["learner_p1_games"])
        p2_games += int(stats["learner_p2_games"])
        p1_reward += float(stats["learner_p1_reward"]) * int(stats["learner_p1_games"])
        p2_reward += float(stats["learner_p2_reward"]) * int(stats["learner_p2_games"])
        total_turns += float(stats["mean_turns"]) * completed

    elapsed = time.perf_counter() - started
    def rate(reward: float, games: int) -> float:
        return reward / games if games else float("nan")

    print(
        f"native eval {args.learner} vs {args.opponent}\n"
        f"games={total_games} score={rate(total_reward, total_games):.3%} "
        f"p1={rate(p1_reward, p1_games):.3%} ({p1_games}) "
        f"p2={rate(p2_reward, p2_games):.3%} ({p2_games}) "
        f"mean_turns={total_turns / total_games:.1f} "
        f"games_s={total_games / elapsed:.1f} horizons={horizons}",
        flush=True,
    )


if __name__ == "__main__":
    main()
