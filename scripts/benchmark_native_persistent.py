#!/usr/bin/env python3
"""Benchmark fixed-shape, immediately-refilled native VGC rollout horizons.

Examples:
    VGC_DEX_DIR=~/Dev/localdex/data uv run --extra nn python \
        scripts/benchmark_native_persistent.py --device cuda --slots 1024 2048 4096 \
        --horizon 64 --team-preview

    # Frozen-opponent specialist rollout:
    VGC_DEX_DIR=~/Dev/localdex/data uv run --extra nn python \
        scripts/benchmark_native_persistent.py --league --learner-team data/teams/reg_mb/MB301.txt \
        --opponent-model data/models/mc-cts-rnad-v7-MC301.pt --device cuda
"""

from __future__ import annotations

import argparse
import time

import torch

from ai_vgc.nn.player import load_policy
from ai_vgc.nn.vgc_env import NativeBatchSelfPlay, NativeVGCConfig


def _synchronize(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize(torch.device(device))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="data/models/mc-cts-rnad-v7-MC301.pt")
    ap.add_argument("--opponent-model", default=None,
                    help="frozen policy in --league mode (default: --model)")
    ap.add_argument("--teams", default="data/teams/reg_mb")
    ap.add_argument("--opponent-teams", default=None,
                    help="opponent team pool in --league mode (default: --teams)")
    ap.add_argument("--learner-team", default=None,
                    help="fixed learner team; required by --league")
    ap.add_argument("--format", default="gen9championsvgc2026regmb")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--slots", nargs="+", type=int, default=[256, 1024, 2048, 4096])
    ap.add_argument("--horizon", type=int, default=64, help="normal policy waves per run")
    ap.add_argument("--max-turns", type=int, default=128)
    ap.add_argument("--team-preview", action="store_true")
    ap.add_argument("--compile", action="store_true",
                    help="benchmark torch.compile for the static normal policy batch")
    ap.add_argument("--league", action="store_true",
                    help="learner-only rollout versus --opponent-model")
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch reports no CUDA device")
    if args.league and not args.learner_team:
        raise ValueError("--league requires --learner-team")
    if args.horizon < 1 or any(slots < 1 for slots in args.slots):
        raise ValueError("--horizon and every --slots value must be positive")

    learner = load_policy(args.model, args.device).eval()
    opponent = learner
    if args.league and args.opponent_model:
        opponent = load_policy(args.opponent_model, args.device).eval()
    print(
        f"device={args.device} mode={'league' if args.league else 'self'} "
        f"horizon={args.horizon} preview={args.team_preview} compile={args.compile} model={args.model}",
        flush=True,
    )
    print(
        "slots completed games/s normal-states/s replacement/s preview/s "
        "mean-turns elapsed_s export% sample% step% reset% preview% peak_cuda_mb",
        flush=True,
    )
    for slots in args.slots:
        config = NativeVGCConfig(
            fmt=args.format,
            teams_dir=args.teams,
            learner_team=args.learner_team if args.league else None,
            opponent_teams_dir=args.opponent_teams,
            max_turns=args.max_turns,
            team_preview=args.team_preview,
            compile_inference=args.compile,
            seed=args.seed + slots,
        )
        runner = NativeBatchSelfPlay(config)

        def collect():
            if args.league:
                return runner.collect_persistent_league_data(
                    learner, opponent, slots, args.horizon,
                )
            return runner.collect_persistent_data(learner, slots, args.horizon)

        for _ in range(args.warmup):
            collect()
        totals: dict[str, float] = {
            "completed_games": 0.0, "normal_states": 0.0,
            "replacement_states": 0.0, "preview_states": 0.0,
            "mean_turns_weighted": 0.0, "elapsed_s": 0.0,
            "export_s": 0.0, "sample_s": 0.0, "step_s": 0.0,
            "reset_s": 0.0, "preview_s": 0.0,
        }
        if args.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(torch.device(args.device))
        for _ in range(args.repeats):
            _synchronize(args.device)
            started = time.perf_counter()
            _data, stats = collect()
            _synchronize(args.device)
            elapsed = time.perf_counter() - started
            completed = float(stats["completed_games"])
            totals["completed_games"] += completed
            totals["normal_states"] += float(stats["normal_states"])
            totals["replacement_states"] += float(stats.get("replacement_states", 0))
            totals["preview_states"] += float(stats["preview_states"])
            totals["mean_turns_weighted"] += completed * float(stats["mean_turns"])
            totals["elapsed_s"] += elapsed
            for key in ("export_s", "step_s", "reset_s", "preview_s"):
                totals[key] += float(stats[key])
            totals["sample_s"] += float(stats.get("sample_s", 0))
            totals["sample_s"] += float(stats.get("learner_sample_s", 0))
            totals["sample_s"] += float(stats.get("opponent_sample_s", 0))

        elapsed = totals["elapsed_s"]
        phase_pct = {
            key: 100.0 * totals[key] / elapsed if elapsed else 0.0
            for key in ("export_s", "sample_s", "step_s", "reset_s", "preview_s")
        }
        peak = (torch.cuda.max_memory_allocated(torch.device(args.device)) / 2**20
                if args.device.startswith("cuda") else 0.0)
        mean_turns = (totals["mean_turns_weighted"] / totals["completed_games"]
                      if totals["completed_games"] else 0.0)
        print(
            f"{slots:5d} {int(totals['completed_games']):9d} "
            f"{totals['completed_games'] / elapsed:7.1f} "
            f"{totals['normal_states'] / elapsed:15.1f} "
            f"{totals['replacement_states'] / elapsed:13.1f} "
            f"{totals['preview_states'] / elapsed:9.1f} "
            f"{mean_turns:10.1f} {elapsed:9.2f} "
            f"{phase_pct['export_s']:6.1f} {phase_pct['sample_s']:7.1f} "
            f"{phase_pct['step_s']:5.1f} {phase_pct['reset_s']:6.1f} "
            f"{phase_pct['preview_s']:8.1f} {peak:12.1f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
