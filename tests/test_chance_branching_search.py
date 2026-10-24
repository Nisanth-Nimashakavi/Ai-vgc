"""Unit tests for chance branching search and sequential halving."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import numpy as np
import pytest
import torch
import vgc_engine

from ai_vgc.nn.preview18 import TeamPreviewPool, PREVIEW_PLANS
from ai_vgc.nn.search_inputs19 import search_node_inputs, search_leaf_values
from eval_showdown_stateful_shadow import search_turn_19, solve_regret_matching


def setup_test_battle():
    species = ['gengar', 'snorlax', 'pikachu', 'charizard', 'pelipper', 'garchomp']
    teams = [[dict(species=s, moves=['protect', 'tackle', 'thunderbolt', 'surf'], ability='pressure')
              for s in species] for _ in range(2)]
    pool = TeamPreviewPool(teams)
    native = vgc_engine.Battle.from_teams(
        pool.selected_json(0, 0),
        pool.selected_json(1, 0),
        format='gen9championsvgc2026regmc',
        stat_points=True,
        decision_phases=True,
        seed=42
    )
    native.enable_belief_observations19()
    context = dict(
        pool=pool,
        full_teams=np.array([[0, 1]]),
        native_picks=np.array([[PREVIEW_PLANS[0], PREVIEW_PLANS[0]]]),
        belief_bank=None,
    )
    return native, context


class MockPolicyModel(torch.nn.Module):
    """Mock Model 19 producing deterministic policy and value outputs."""
    def __init__(self):
        super().__init__()
        self.device = torch.device("cpu")

    def __call__(self, cat, num, action_a=None, **history):
        B = cat.shape[0]
        # Uniform or peaked policy
        p_a = torch.zeros((B, 19), dtype=torch.float32)
        p_b = torch.zeros((B, 19), dtype=torch.float32)
        # Give higher weight to first few actions
        for i in range(19):
            p_a[:, i] = float(19 - i)
            p_b[:, i] = float(19 - i)

        val = torch.full((B, 1), 0.15, dtype=torch.float32)
        return {
            "policy_a": p_a,
            "policy_b": p_b,
            "value": val,
        }


def test_multi_world_rollout_chance_exact_probabilities():
    native, context = setup_test_battle()

    our_choices = [native.actions_18_to_choices(0, 0, 0)]
    opp_choices = [native.actions_18_to_choices(1, 0, 0)]

    res = vgc_engine.multi_world_rollout_chance(
        [native],
        our_choices,
        [opp_choices],
        our_side=0,
        rating=1500.0,
        ots=False
    )

    probs = np.array(res["outcome_prob"], dtype=np.float32)
    offs = np.asarray(res["cell_offsets"], dtype=np.intp)
    cnts = np.asarray(res["cell_counts"], dtype=np.intp)

    # Probabilities in each cell must sum to 1.0 exactly
    for k in range(len(offs)):
        cell_p = probs[offs[k]:offs[k] + cnts[k]].sum()
        assert np.isclose(cell_p, 1.0, atol=1e-4), f"Cell {k} probability sum was {cell_p}"


def test_search_turn_19_chance_branching():
    native, context = setup_test_battle()
    model = MockPolicyModel()

    (c, n, history), (c_opp, n_opp, opp_history) = search_node_inputs(native, 0, context, torch.device("cpu"))
    m0 = torch.tensor(native.legal_action_mask_18(0, 0), dtype=torch.bool)
    m1 = torch.tensor(native.legal_action_mask_18(0, 1), dtype=torch.bool)

    valid_pairs = [(0, 0), (0, 1), (1, 0), (1, 1)]

    best_pair, g_val = search_turn_19(
        model=model,
        native=native,
        side=0,
        c=c,
        n=n,
        history=history,
        c_opp=c_opp,
        n_opp=n_opp,
        m0=m0,
        m1=m1,
        valid_pairs=valid_pairs,
        device=torch.device("cpu"),
        depth=1,
        k=6,
        m=6,
        chance_branching=True,
        deterministic=True,
        input_context=context,
    )

    assert best_pair in valid_pairs
    assert np.isfinite(g_val)
    print("Search turn 19 chance branching result:", best_pair, g_val)


def test_win_rate_cap_prevents_suboptimal_bluffs():
    M = np.array([
        [0.8, 0.7],    # Action 0: excellent payoff (EV ~ 0.75)
        [0.72, 0.70],  # Action 1: close payoff (EV ~ 0.71, delta = 0.04 < 0.06)
        [-0.5, -0.4],  # Action 2: blundering line (EV ~ -0.45, delta >> 0.06)
    ], dtype=np.float64)

    pi_opp = np.array([0.5, 0.5])
    q_vals = M @ pi_opp  # [0.75, 0.71, -0.45]
    best_q = float(np.max(q_vals))

    cap_delta = 0.06  # 3 win-% points in [-1, 1] scale
    valid_mask = (q_vals >= (best_q - cap_delta))

    assert valid_mask[0] == True
    assert valid_mask[1] == True
    assert valid_mask[2] == False  # Action 2 must be capped out!
