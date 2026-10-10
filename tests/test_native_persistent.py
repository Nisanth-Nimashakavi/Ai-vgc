"""End-to-end invariants for fixed-shape native PPO collection."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
import torch

pytest.importorskip("vgc_engine")

from ai_vgc.nn.player import load_policy
from ai_vgc.nn.vgc_env import NativeBatchSelfPlay, NativeVGCConfig


ROOT = Path(__file__).parents[1]
MODEL = ROOT / "data/models/mc-cts-rnad-v7-MC301.pt"
TEAMS = ROOT / "data/teams/reg_mb"
FORMAT = "gen9championsvgc2026regmb"


@pytest.fixture(scope="module")
def policy():
    if not MODEL.exists():
        pytest.skip(f"native rollout fixture is absent: {MODEL}")
    torch.set_num_threads(1)
    return load_policy(MODEL).eval()


def _config(**kwargs) -> NativeVGCConfig:
    return NativeVGCConfig(
        fmt=FORMAT,
        teams_dir=TEAMS,
        max_turns=1,
        seed=41,
        **kwargs,
    )


def test_persistent_self_play_compacts_terminal_episodes_and_refills_slots(policy):
    runner = NativeBatchSelfPlay(_config(team_preview=True))
    data, stats = runner.collect_persistent_data(policy, slots=2, decision_horizon=2)

    # A one-turn cutoff forces every live slot to finish and refill on every
    # normal wave. Both physical perspectives must end exactly once per game.
    assert stats["completed_games"] == 4
    assert stats["completed_player_episodes"] == 8
    assert stats["normal_states"] == 8
    assert int(data["last"].sum()) == 8
    assert len(data["action"]) == len(data["last"])
    assert int(data["preview"].sum()) > 0
    assert np.all(np.isin(data["reward"][data["last"]], (0.0, 0.5, 1.0)))


def test_persistent_league_keeps_learner_rows_and_resets_preview_slots(policy):
    learner_team = TEAMS / "MB100.txt"
    runner = NativeBatchSelfPlay(_config(learner_team=learner_team, team_preview=True))
    data, stats = runner.collect_persistent_league_data(
        policy, policy, slots=2, decision_horizon=2,
    )

    # League trajectories retain only the learner perspective: one terminal
    # marker per finished battle rather than the two self-play perspectives.
    assert stats["completed_games"] == 4
    assert stats["completed_player_episodes"] == 4
    assert stats["normal_states"] == 4
    # Each completed slot swaps physical side before its immediate refill, so
    # the learner's P1/P2 exposure stays balanced even across short horizons.
    assert stats["learner_p1_games"] == stats["learner_p2_games"] == 2
    assert int(data["last"].sum()) == 4
    assert len(data["action"]) == len(data["last"])
    assert int(data["preview"].sum()) > 0
    assert np.all(np.isin(data["reward"][data["last"]], (0.0, 0.5, 1.0)))
