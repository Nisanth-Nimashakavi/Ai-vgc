"""One-turn simultaneous-move search engine for Model 18 (Transformer18 with 35 actions).

Features:
- Action space: 35 explicit normal/Mega actions per slot.
- Conditioned Slot B generation with Mega conflict and duplicate switch prevention.
- Ultra-fast Rust parallel branch execution in vgc_engine (~0.1ms for candidate matrices).
- Batched Neural Leaf Evaluation using Model 18 value head.
- Matrix Game Solver: Regret Matching for 2-player zero-sum simultaneous-move games.
- Prior Blending: Blends net's policy prior with minimax search equilibrium.
"""

from __future__ import annotations

import math
from typing import Any
import numpy as np
import torch
import torch.nn.functional as F

import vgc_engine
from ai_vgc.nn.model18 import Transformer18
from ai_vgc.nn.ppo18 import conditioned_b_mask, mask_closed_observation


def solve_regret_matching(
    M: np.ndarray,
    opp_prior: np.ndarray | None = None,
    iterations: int = 1024,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Solve 2-player zero-sum normal-form game with payoff matrix M via Regret Matching.

    Row player maximizes, Column player minimizes.
    M: shape (K, M_cols)
    Returns: (row_strategy, col_strategy, game_value)
    """
    K, M_cols = M.shape
    cum_regret_row = np.zeros(K, dtype=np.float64)
    cum_regret_col = np.zeros(M_cols, dtype=np.float64)
    avg_strat_row = np.zeros(K, dtype=np.float64)
    avg_strat_col = np.zeros(M_cols, dtype=np.float64)

    strat_row = np.ones(K, dtype=np.float64) / K
    if opp_prior is not None:
        strat_col = np.asarray(opp_prior, dtype=np.float64).copy()
        s = strat_col.sum()
        strat_col = strat_col / s if s > 1e-8 else np.ones(M_cols, dtype=np.float64) / M_cols
    else:
        strat_col = np.ones(M_cols, dtype=np.float64) / M_cols

    for _ in range(iterations):
        row_payoffs = M @ strat_col
        col_payoffs = strat_row @ M

        val = float(strat_row @ row_payoffs)

        avg_strat_row += strat_row
        avg_strat_col += strat_col

        cum_regret_row += row_payoffs - val
        cum_regret_col += val - col_payoffs

        pos_row = np.maximum(cum_regret_row, 0.0)
        s_row = pos_row.sum()
        strat_row = pos_row / s_row if s_row > 1e-8 else np.ones(K, dtype=np.float64) / K

        pos_col = np.maximum(cum_regret_col, 0.0)
        s_col = pos_col.sum()
        strat_col = pos_col / s_col if s_col > 1e-8 else np.ones(M_cols, dtype=np.float64) / M_cols

    pi_row = avg_strat_row / iterations
    pi_col = avg_strat_col / iterations
    game_value = float(pi_row @ M @ pi_col)
    return pi_row, pi_col, game_value


@torch.no_grad()
def get_our_candidates_18(
    model: Transformer18,
    tok_cat: torch.Tensor,
    tok_num: torch.Tensor,
    mask_a: list[bool] | np.ndarray,
    mask_b: list[bool] | np.ndarray,
    k: int = 6,
    ka: int = 3,
    history: dict[str, torch.Tensor] | None = None,
    device: torch.device | str = "cpu",
) -> list[tuple[tuple[int, int], float, float]]:
    """Generate top-K joint actions for our side: ((act_a, act_b), log_prob, prob)."""
    model.eval()
    dev = torch.device(device)
    t_cat = tok_cat.to(dev)
    t_num = tok_num.to(dev)
    hist = {k: v.to(dev) for k, v in history.items()} if history else {}

    scorer = getattr(model, "score_slot_b", None)
    if scorer is not None:
        out = model(t_cat, t_num, return_hidden=True, **hist)
    else:
        out = model(t_cat, t_num, **hist)

    logits_a = out["policy_a"][0].clone().float()  # shape (35,)
    mask_a_t = torch.tensor(mask_a, dtype=torch.bool, device=dev)
    logits_a[~mask_a_t] = -1e9
    lp_a = F.log_softmax(logits_a, dim=-1)

    top_ka = min(ka, int(mask_a_t.sum().item()))
    if top_ka == 0:
        return [((18, 18), 0.0, 1.0)]

    cands: list[tuple[tuple[int, int], float]] = []
    top_a_indices = lp_a.topk(top_ka).indices.tolist()

    mask_b_t = torch.tensor(mask_b, dtype=torch.bool, device=dev).unsqueeze(0)

    for a0 in top_a_indices:
        lp_a0 = lp_a[a0].item()
        prev_a = torch.tensor([a0], device=dev)

        curr_mask_b = conditioned_b_mask(mask_b_t, prev_a)[0]
        if curr_mask_b.sum() == 0:
            curr_mask_b[18] = True

        if scorer is not None and "_hidden" in out:
            logits_b = scorer(out["_hidden"][:1], prev_a)[0].clone().float()
        else:
            logits_b = model(t_cat, t_num, action_a=prev_a, **hist)["policy_b"][0].clone().float()

        logits_b[~curr_mask_b] = -1e9
        lp_b = F.log_softmax(logits_b, dim=-1)

        top_kb = min(k, int(curr_mask_b.sum().item()))
        top_b_indices = lp_b.topk(top_kb).indices.tolist()

        for a1 in top_b_indices:
            joint_lp = lp_a0 + lp_b[a1].item()
            cands.append(((a0, a1), joint_lp))

    cands.sort(key=lambda c: -c[1])
    selected = cands[:k]

    max_lp = selected[0][1] if selected else 0.0
    weights = [math.exp(lp - max_lp) for _, lp in selected]
    w_sum = sum(weights) or 1.0
    probs = [w / w_sum for w in weights]

    return [((act[0], act[1]), lp, p) for (act, lp), p in zip(selected, probs)]


@torch.no_grad()
def get_opp_candidates_18(
    model: Transformer18,
    tok_cat: torch.Tensor,
    tok_num: torch.Tensor,
    mask_a: list[bool] | np.ndarray,
    mask_b: list[bool] | np.ndarray,
    m: int = 5,
    ma: int = 3,
    history: dict[str, torch.Tensor] | None = None,
    device: torch.device | str = "cpu",
) -> list[tuple[tuple[int, int], float, float]]:
    """Generate top-M predicted candidate joint actions for opponent side."""
    return get_our_candidates_18(
        model, tok_cat, tok_num, mask_a, mask_b, k=m, ka=ma, history=history, device=device
    )


@torch.no_grad()
def search_simultaneous_turn_18(
    model: Transformer18,
    battle: vgc_engine.Battle,
    our_side: int = 0,
    k: int = 6,
    m: int = 5,
    blend_alpha: float = 0.6,
    temperature: float = 0.5,
    history: dict[str, torch.Tensor] | None = None,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Perform 1-turn simultaneous-move search over 35-action space using Model 18.

    1. Generate top-K candidate joint actions for our side via autoregressive conditioning.
    2. Generate top-M candidate joint actions for opponent.
    3. Step each candidate pairing (our_k, opp_m) in Rust vgc_engine.
    4. Score non-terminal leaf states with Model 18's Value Head.
    5. Solve the resulting K x M matrix game with Regret Matching for the Nash equilibrium.
    6. Blend search equilibrium with network prior and return optimal joint decision.
    """
    dev = torch.device(device)
    opp_side = 1 - our_side

    # 1. Action masks for both sides
    mask_our_0 = battle.legal_action_mask_18(our_side, 0)
    mask_our_1 = battle.legal_action_mask_18(our_side, 1)
    mask_opp_0 = battle.legal_action_mask_18(opp_side, 0)
    mask_opp_1 = battle.legal_action_mask_18(opp_side, 1)

    # 2. Encode current position
    enc_our = battle.encode_17(our_side)
    enc_opp = battle.encode_17(opp_side)

    tok_cat_our = torch.from_numpy(np.frombuffer(enc_our["tok_cat"], dtype=np.uint32).reshape(1, 17, 4).copy())
    tok_num_our = torch.from_numpy(np.frombuffer(enc_our["tok_num"], dtype=np.float32).reshape(1, 17, 16).copy())
    mask_closed_observation(tok_cat_our, tok_num_our)

    tok_cat_opp = torch.from_numpy(np.frombuffer(enc_opp["tok_cat"], dtype=np.uint32).reshape(1, 17, 4).copy())
    tok_num_opp = torch.from_numpy(np.frombuffer(enc_opp["tok_num"], dtype=np.float32).reshape(1, 17, 16).copy())
    mask_closed_observation(tok_cat_opp, tok_num_opp)

    # 3. Candidate joint actions
    our_cands = get_our_candidates_18(model, tok_cat_our, tok_num_our, mask_our_0, mask_our_1, k=k, history=history, device=dev)
    opp_cands = get_opp_candidates_18(model, tok_cat_opp, tok_num_opp, mask_opp_0, mask_opp_1, m=m, history=history, device=dev)

    our_actions = [c[0] for c in our_cands]
    opp_actions = [c[0] for c in opp_cands]
    our_priors = np.array([c[2] for c in our_cands], dtype=np.float64)
    opp_priors = np.array([c[2] for c in opp_cands], dtype=np.float64)

    # 4. Step candidate matrix in Rust engine
    num_our = len(our_actions)
    num_opp = len(opp_actions)
    M = np.zeros((num_our, num_opp), dtype=np.float32)

    leaf_cats = []
    leaf_nums = []
    leaf_coords = []  # (i, j)

    for i, our_act in enumerate(our_actions):
        our_ch = battle.actions_18_to_choices(our_side, our_act[0], our_act[1])
        for j, opp_act in enumerate(opp_actions):
            opp_ch = battle.actions_18_to_choices(opp_side, opp_act[0], opp_act[1])
            leaf = battle.clone()
            p1_ch, p2_ch = (our_ch, opp_ch) if our_side == 0 else (opp_ch, our_ch)
            res = leaf.step_move(p1_ch, p2_ch)

            if res == "p1_win":
                M[i, j] = 1.0 if our_side == 0 else -1.0
            elif res == "p2_win":
                M[i, j] = -1.0 if our_side == 0 else 1.0
            elif res == "draw":
                M[i, j] = 0.0
            else:
                enc_leaf = leaf.encode_17(our_side)
                c_np = np.frombuffer(enc_leaf["tok_cat"], dtype=np.uint32).reshape(1, 17, 4).copy()
                n_np = np.frombuffer(enc_leaf["tok_num"], dtype=np.float32).reshape(1, 17, 16).copy()
                leaf_cats.append(c_np)
                leaf_nums.append(n_np)
                leaf_coords.append((i, j))

    # 5. Batched evaluation of non-terminal leaves
    if leaf_cats:
        batch_cat = torch.from_numpy(np.concatenate(leaf_cats, axis=0)).to(dev)
        batch_num = torch.from_numpy(np.concatenate(leaf_nums, axis=0)).to(dev)
        mask_closed_observation(batch_cat, batch_num)

        with torch.no_grad():
            out_leaves = model(batch_cat, batch_num)
            vals = out_leaves["value"].squeeze(-1).cpu().numpy()

        for idx, (i, j) in enumerate(leaf_coords):
            M[i, j] = float(vals[idx])

    # 6. Solve Game Matrix with Regret Matching
    pi_search, pi_opp, game_val = solve_regret_matching(M, opp_prior=opp_priors)

    # 7. Blend Search Equilibrium with Network Prior
    pi_final = (1.0 - blend_alpha) * our_priors + blend_alpha * pi_search
    s = pi_final.sum()
    pi_final = pi_final / s if s > 1e-8 else our_priors

    if temperature <= 0.0:
        best_idx = int(np.argmax(pi_final))
    else:
        log_pi = np.log(np.maximum(pi_final, 1e-12)) / temperature
        log_pi -= np.max(log_pi)
        exp_pi = np.exp(log_pi)
        exp_pi /= exp_pi.sum()
        best_idx = int(np.random.choice(len(our_actions), p=exp_pi))

    chosen_joint = our_actions[best_idx]
    choice_0, choice_1 = battle.actions_18_to_choices(our_side, chosen_joint[0], chosen_joint[1])

    return {
        "action": chosen_joint,
        "choices": [choice_0, choice_1],
        "best_index": best_idx,
        "pi_final": pi_final,
        "pi_search": pi_search,
        "our_priors": our_priors,
        "game_value": float(game_val),
        "payoff_matrix": M,
        "our_actions": our_actions,
        "opp_actions": opp_actions,
    }
