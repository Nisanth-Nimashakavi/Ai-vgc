"""One-turn simultaneous-move search engine for the 17-token Transformer (Wolfe_Alt6767 blueprint).

Features:
- Candidates: Policy top-K joint actions (Slot A + conditioned Slot B) x Opponent Prediction top-M actions.
- Multi-World Rollouts: Pure Rust parallel branch execution in ~0.7ms for 120 leaves.
- Neural Leaf Evaluation: Batched Value Head scoring over non-terminal game states.
- Matrix Solver: Regret Matching for 2-player zero-sum simultaneous-move games.
- Prior Blending: Blends net's own prior preference with search value advantage to avoid chasing noisy estimates.
"""

from __future__ import annotations

import math
from typing import Any
import numpy as np
import torch
import torch.nn.functional as F

import vgc_engine
from ai_vgc.nn.model17 import Transformer17


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
        # Payoffs against opponent mixed strategy
        row_payoffs = M @ strat_col
        # Payoffs against row mixed strategy
        col_payoffs = strat_row @ M

        val = float(strat_row @ row_payoffs)

        # Accumulate strategies
        avg_strat_row += strat_row
        avg_strat_col += strat_col

        # Update regrets
        cum_regret_row += row_payoffs - val
        cum_regret_col += val - col_payoffs

        # Next mixed strategy from positive regret
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
def get_our_candidates(
    model: Transformer17,
    tok_cat: torch.Tensor,
    tok_num: torch.Tensor,
    mask_a: list[bool],
    mask_b: list[bool],
    k: int = 6,
    ka: int = 3,
    device: torch.device | str = "cpu",
) -> list[tuple[tuple[int, int], float, float]]:
    """Generate top-K joint actions for our side: ((act_a, act_b), log_prob, prob).

    Uses autoregressive conditioning: Slot B is evaluated conditioned on each candidate Slot A action.
    """
    model.eval()
    t_cat = tok_cat.to(device)
    t_num = tok_num.to(device)

    out = model(t_cat, t_num)
    logits_a = out["policy_a"][0]  # shape (19,)

    # Apply legality mask for Slot A
    mask_a_t = torch.tensor(mask_a, dtype=torch.bool, device=device)
    logits_a[~mask_a_t] = -1e9
    lp_a = F.log_softmax(logits_a, dim=-1)

    # Top-ka candidate actions for slot A
    top_ka = min(ka, int(mask_a_t.sum().item()))
    if top_ka == 0:
        return [((18, 18), 0.0, 1.0)]

    cands: list[tuple[tuple[int, int], float]] = []
    top_a_indices = lp_a.topk(top_ka).indices.tolist()

    mask_b_t = torch.tensor(mask_b, dtype=torch.bool, device=device)

    for a0 in top_a_indices:
        lp_a0 = lp_a[a0].item()
        prev_a = torch.tensor([a0], device=device)
        out_b = model(t_cat, t_num, action_a=prev_a)
        logits_b = out_b["policy_b"][0].clone()

        # Legality mask for Slot B
        curr_mask_b = mask_b_t.clone()
        # Avoid switching both slots to the same bench mon (16 or 17)
        if a0 in (16, 17) and a0 < len(curr_mask_b):
            curr_mask_b[a0] = False

        if curr_mask_b.sum() == 0:
            curr_mask_b[18] = True  # Pass if no choices

        logits_b[~curr_mask_b] = -1e9
        lp_b = F.log_softmax(logits_b, dim=-1)

        top_kb = min(k, int(curr_mask_b.sum().item()))
        top_b_indices = lp_b.topk(top_kb).indices.tolist()

        for a1 in top_b_indices:
            joint_lp = lp_a0 + lp_b[a1].item()
            cands.append(((a0, a1), joint_lp))

    cands.sort(key=lambda c: -c[1])
    selected = cands[:k]

    # Convert log-probs to normalized probabilities
    max_lp = selected[0][1] if selected else 0.0
    weights = [math.exp(lp - max_lp) for _, lp in selected]
    w_sum = sum(weights) or 1.0
    probs = [w / w_sum for w in weights]

    return [((act[0], act[1]), lp, p) for (act, lp), p in zip(selected, probs)]


@torch.no_grad()
def get_opp_candidates(
    model: Transformer17,
    tok_cat: torch.Tensor,
    tok_num: torch.Tensor,
    mask_a: list[bool],
    mask_b: list[bool],
    m: int = 5,
    device: torch.device | str = "cpu",
) -> list[tuple[tuple[int, int], float, float]]:
    """Generate top-M joint actions for the opponent from opp_policy_a and opp_policy_b heads."""
    model.eval()
    t_cat = tok_cat.to(device)
    t_num = tok_num.to(device)

    out = model(t_cat, t_num)
    logits_a = out["opp_a"][0]
    logits_b = out["opp_b"][0]

    mask_a_t = torch.tensor(mask_a, dtype=torch.bool, device=device)
    mask_b_t = torch.tensor(mask_b, dtype=torch.bool, device=device)

    if mask_a_t.sum() == 0:
        mask_a_t[18] = True
    if mask_b_t.sum() == 0:
        mask_b_t[18] = True

    logits_a[~mask_a_t] = -1e9
    logits_b[~mask_b_t] = -1e9

    lp_a = F.log_softmax(logits_a, dim=-1)
    lp_b = F.log_softmax(logits_b, dim=-1)

    cands: list[tuple[tuple[int, int], float]] = []
    top_a = lp_a.topk(min(3, int(mask_a_t.sum().item()))).indices.tolist()
    top_b = lp_b.topk(min(3, int(mask_b_t.sum().item()))).indices.tolist()

    for a0 in top_a:
        for a1 in top_b:
            if a0 in (16, 17) and a0 == a1:
                continue
            joint_lp = (lp_a[a0] + lp_b[a1]).item()
            cands.append(((a0, a1), joint_lp))

    cands.sort(key=lambda c: -c[1])
    selected = cands[:m]

    max_lp = selected[0][1] if selected else 0.0
    weights = [math.exp(lp - max_lp) for _, lp in selected]
    w_sum = sum(weights) or 1.0
    probs = [w / w_sum for w in weights]

    return [((act[0], act[1]), lp, p) for (act, lp), p in zip(selected, probs)]


def check_endgame_state(battle: vgc_engine.Battle) -> tuple[bool, str, int, int]:
    """Check if the current battle is in an endgame state suitable for exact mathematical solving.

    Triggers when:
    1. 2v2 endgame with 0 bench on both sides (neither side can switch; finite terminal horizon).
    2. 2v1 or 1v2 endgame (one side down to 1 mon, 0 bench).
    3. 1v1 endgame (both sides down to 1 mon, 0 bench).
    4. Any state where total remaining mons (active + bench) <= 2 on either side.
    """
    p1_active = sum(1 for s in range(2) if battle.active_hp(0, s) > 0)
    p2_active = sum(1 for s in range(2) if battle.active_hp(1, s) > 0)
    p1_bench = len([c for c in battle.legal_choices(0, 0) if c[0] == "switch"])
    p2_bench = len([c for c in battle.legal_choices(1, 0) if c[0] == "switch"])

    p1_total = p1_active + p1_bench
    p2_total = p2_active + p2_bench

    if p1_bench == 0 and p2_bench == 0 and p1_active <= 2 and p2_active <= 2:
        return True, f"{p1_active}v{p2_active} (no bench)", p1_total, p2_total
    if p1_total <= 2 or p2_total <= 2:
        return True, f"{p1_total}v{p2_total} (low count)", p1_total, p2_total
    return False, f"{p1_total}v{p2_total}", p1_total, p2_total


def solve_endgame(
    battle: vgc_engine.Battle,
    our_side: int = 0,
    max_depth: int = 4,
    max_actions: int = 5,
    max_chance_combinations: int = 32,
    node_budget: int = 1000,
) -> dict[str, Any] | None:
    """Solve 2v2, 2v1, 1v1 endgames via backward induction with exact damage rolls & accuracy checks.

    Runs Rust-native solve_endgame_exact over the game tree.
    Returns: search result dict or None if solve could not complete.
    """
    try:
        sol = vgc_engine.solve_endgame_exact(
            battle,
            max_depth=max_depth,
            max_actions=max_actions,
            max_chance_combinations=max_chance_combinations,
            node_budget=node_budget,
        )
        if not sol:
            return None

        joint_policy = sol.get("row_joint_policy" if our_side == 0 else "col_joint_policy")
        if not joint_policy:
            return None

        best_joint, prob = joint_policy[0]
        choice_0 = best_joint[0]
        choice_1 = best_joint[1]

        act_0 = battle.choice_to_action_17(choice_0, side=our_side)
        act_1 = battle.choice_to_action_17(choice_1, side=our_side)

        our_val = sol["value"] if our_side == 0 else -sol["value"]

        return {
            "action": (act_0, act_1),
            "choices": [choice_0, choice_1],
            "game_value": float(our_val),
            "provenance": sol.get("provenance", "exact"),
            "nodes_visited": sol.get("nodes_visited", 0),
            "policy": joint_policy,
            "is_endgame_solve": True,
            "best_index": 0,
            "pi_final": np.array([prob], dtype=np.float64),
            "pi_search": np.array([prob], dtype=np.float64),
            "pi_net": np.array([1.0], dtype=np.float64),
            "payoff_matrix": np.array([[our_val]], dtype=np.float32),
            "our_candidates": [((act_0, act_1), 0.0, 1.0)],
            "opp_candidates": [],
            "n_worlds": 1,
            "n_leaves": sol.get("nodes_visited", 0),
        }
    except Exception:
        return None


@torch.no_grad()
def search_simultaneous_turn(
    model: Transformer17,
    battle: vgc_engine.Battle,
    our_side: int = 0,
    worlds: list[vgc_engine.Battle] | None = None,
    depth: int = 1,
    k: int = 6,
    m: int = 5,
    k2: int = 3,
    m2: int = 3,
    blend_alpha: float = 0.5,
    solve_endgames: bool = True,
    device: torch.device | str = "cpu",
    cts: bool = True,
) -> dict[str, Any]:
    """Execute simultaneous-move search with Endgame Factoring & optional Depth-2 Lookahead.

    1. Checks for 2v2 (no bench), 2v1, or 1v1 endgames. If detected, immediately calls
       Rust's exact backward-induction solver to find the provably optimal minimax move.
    2. Encodes battle and extracts policy candidates for Us and predicted candidates for Opponent.
    3. If Depth=1: runs multi_world_rollout_17 and evaluates non-terminal leaves via Value Head.
    4. If Depth=2: rolls forward Turn 1 candidates, solves subgames that reach endgames, and
       expands top Turn 2 continuation candidates with batched Value Head leaf evaluation.
    5. Solves payoff matrix M via Regret Matching and blends with policy prior.
    """
    model.eval()
    opp_side = 1 - our_side

    # 1. Exact Endgame Factoring (2v2 no bench, 2v1, 1v1)
    if solve_endgames:
        is_eg, eg_desc, _, _ = check_endgame_state(battle)
        if is_eg:
            eg_sol = solve_endgame(battle, our_side=our_side)
            if eg_sol is not None:
                eg_sol["endgame_type"] = eg_desc
                return eg_sol

    # 2. Legal action masks for both sides
    mask_our_0 = battle.legal_action_mask_17(our_side, 0)
    mask_our_1 = battle.legal_action_mask_17(our_side, 1)
    mask_opp_0 = battle.legal_action_mask_17(opp_side, 0)
    mask_opp_1 = battle.legal_action_mask_17(opp_side, 1)

    # 3. Encode current position
    enc = battle.encode_17(our_side)
    tok_cat_np = np.frombuffer(enc["tok_cat"], dtype=np.uint32).reshape(1, 17, 4).copy()
    tok_num_np = np.frombuffer(enc["tok_num"], dtype=np.float32).reshape(1, 17, 16).copy()

    t_cat = torch.from_numpy(tok_cat_np)
    t_num = torch.from_numpy(tok_num_np)
    if cts:
        t_cat[:, 4:8, 1] = 0
        t_cat[:, 4:8, 2] = 0

    # 4. Generate candidate joint actions for Turn 1
    our_cands = get_our_candidates(model, t_cat, t_num, mask_our_0, mask_our_1, k=k, device=device)
    opp_cands = get_opp_candidates(model, t_cat, t_num, mask_opp_0, mask_opp_1, m=m, device=device)

    our_actions = [c[0] for c in our_cands]
    opp_actions = [c[0] for c in opp_cands]
    our_priors = np.array([c[2] for c in our_cands], dtype=np.float64)
    opp_priors = np.array([c[2] for c in opp_cands], dtype=np.float64)

    # 5. Multi-world setup
    if worlds is None or len(worlds) == 0:
        worlds = [battle.clone() for _ in range(4)]
    n_worlds = len(worlds)

    if depth >= 2:
        # ─── DEPTH-2 LOOKAHEAD WITH CONTINUATION REGRET MATCHING ───
        # For each world, step each (our_cand, opp_cand) pair forward 1 turn.
        # Leaves that reach endgames (2v2 no bench, 2v1, 1v1) are solved to optimality.
        # Midgame leaves expand top K2 x M2 continuation candidates.
        num_our = len(our_actions)
        num_opp = len(opp_actions)
        world_matrices = []

        for w_b in worlds:
            M_w = np.zeros((num_our, num_opp), dtype=np.float32)
            continuation_nodes: list[tuple[int, int, vgc_engine.Battle]] = []

            for i, our_act in enumerate(our_actions):
                our_ch = w_b.actions_17_to_choices(our_side, our_act[0], our_act[1])
                for j, opp_act in enumerate(opp_actions):
                    opp_ch = w_b.actions_17_to_choices(opp_side, opp_act[0], opp_act[1])
                    leaf = w_b.clone()
                    p1_ch, p2_ch = (our_ch, opp_ch) if our_side == 0 else (opp_ch, our_ch)
                    res = leaf.step_move(p1_ch, p2_ch)

                    if res == "p1_win":
                        M_w[i, j] = 1.0 if our_side == 0 else -1.0
                    elif res == "p2_win":
                        M_w[i, j] = -1.0 if our_side == 0 else 1.0
                    elif res == "draw":
                        M_w[i, j] = 0.0
                    else:
                        continuation_nodes.append((i, j, leaf))

            # Process continuation nodes
            if continuation_nodes:
                t2_leaf_cat = []
                t2_leaf_num = []
                t2_leaf_map: list[tuple[int, int, int, int]] = []  # (cand_i, cand_j, t2_i, t2_j)
                t2_subgames: dict[tuple[int, int], dict[str, Any]] = {}

                for cand_i, cand_j, leaf_b in continuation_nodes:
                    # Check if continuation node transitioned into an endgame
                    if solve_endgames:
                        is_eg, _, _, _ = check_endgame_state(leaf_b)
                        if is_eg:
                            eg_sol = solve_endgame(leaf_b, our_side=our_side, max_depth=3, max_actions=4)
                            if eg_sol is not None:
                                M_w[cand_i, cand_j] = eg_sol["game_value"]
                                continue

                    # Generate Turn 2 candidates
                    enc2 = leaf_b.encode_17(our_side)
                    t2_cat_np = np.frombuffer(enc2["tok_cat"], dtype=np.uint32).reshape(1, 17, 4).copy()
                    t2_num_np = np.frombuffer(enc2["tok_num"], dtype=np.float32).reshape(1, 17, 16).copy()
                    t2_cat_t = torch.from_numpy(t2_cat_np)
                    t2_num_t = torch.from_numpy(t2_num_np)
                    if cts:
                        t2_cat_t[:, 4:8, 1] = 0
                        t2_cat_t[:, 4:8, 2] = 0

                    m_our0 = leaf_b.legal_action_mask_17(our_side, 0)
                    m_our1 = leaf_b.legal_action_mask_17(our_side, 1)
                    m_opp0 = leaf_b.legal_action_mask_17(opp_side, 0)
                    m_opp1 = leaf_b.legal_action_mask_17(opp_side, 1)

                    cands2_our = get_our_candidates(model, t2_cat_t, t2_num_t, m_our0, m_our1, k=k2, device=device)
                    cands2_opp = get_opp_candidates(model, t2_cat_t, t2_num_t, m_opp0, m_opp1, m=m2, device=device)

                    acts2_our = [c[0] for c in cands2_our]
                    acts2_opp = [c[0] for c in cands2_opp]
                    opp_prior2 = np.array([c[2] for c in cands2_opp], dtype=np.float64)

                    t2_matrix = np.zeros((len(acts2_our), len(acts2_opp)), dtype=np.float32)

                    for u, a2 in enumerate(acts2_our):
                        c2_our = leaf_b.actions_17_to_choices(our_side, a2[0], a2[1])
                        for v, o2 in enumerate(acts2_opp):
                            c2_opp = leaf_b.actions_17_to_choices(opp_side, o2[0], o2[1])
                            leaf2 = leaf_b.clone()
                            p1_c2, p2_c2 = (c2_our, c2_opp) if our_side == 0 else (c2_opp, c2_our)
                            res2 = leaf2.step_move(p1_c2, p2_c2)

                            if res2 == "p1_win":
                                t2_matrix[u, v] = 1.0 if our_side == 0 else -1.0
                            elif res2 == "p2_win":
                                t2_matrix[u, v] = -1.0 if our_side == 0 else 1.0
                            elif res2 == "draw":
                                t2_matrix[u, v] = 0.0
                            else:
                                enc3 = leaf2.encode_17(our_side)
                                t2_leaf_cat.append(np.frombuffer(enc3["tok_cat"], dtype=np.uint32).reshape(1, 17, 4).copy())
                                t2_leaf_num.append(np.frombuffer(enc3["tok_num"], dtype=np.float32).reshape(1, 17, 16).copy())
                                t2_leaf_map.append((cand_i, cand_j, u, v))

                    t2_subgames[(cand_i, cand_j)] = {
                        "matrix": t2_matrix,
                        "opp_prior": opp_prior2,
                    }

                # Single batched GPU forward pass across all Turn 2 continuation leaves
                if t2_leaf_cat:
                    batch_cat = torch.from_numpy(np.concatenate(t2_leaf_cat, axis=0)).to(device)
                    batch_num = torch.from_numpy(np.concatenate(t2_leaf_num, axis=0)).to(device)
                    if cts:
                        batch_cat[:, 4:8, 1] = 0
                        batch_cat[:, 4:8, 2] = 0
                    out2 = model(batch_cat, batch_num)
                    vals2 = out2["value"].squeeze(-1).cpu().numpy()

                    for idx, (ci, cj, u, v) in enumerate(t2_leaf_map):
                        t2_subgames[(ci, cj)]["matrix"][u, v] = vals2[idx]

                # Backward Induction: solve each Turn 2 matrix with Regret Matching
                for (ci, cj), sub in t2_subgames.items():
                    _, _, gv2 = solve_regret_matching(sub["matrix"], opp_prior=sub["opp_prior"])
                    M_w[ci, cj] = gv2

            world_matrices.append(M_w)

        M = np.mean(world_matrices, axis=0)
        n_leaves = sum(len(c) for c in world_matrices)

    else:
        # ─── DEPTH-1 SIMULTANEOUS SEARCH ───
        rollout_res = vgc_engine.multi_world_rollout_17(worlds, our_actions, opp_actions, our_side)
        n_w, num_our, num_opp = rollout_res["shape"]
        terminal_vals = np.array(rollout_res["terminal_values"], dtype=np.float32).reshape(n_w, num_our, num_opp)

        n_leaves = rollout_res["n_leaves"]
        if n_leaves > 0:
            leaf_cat = np.frombuffer(rollout_res["leaf_tok_cat"], dtype=np.uint32).reshape(n_leaves, 17, 4).copy()
            leaf_num = np.frombuffer(rollout_res["leaf_tok_num"], dtype=np.float32).reshape(n_leaves, 17, 16).copy()

            t_leaf_cat = torch.from_numpy(leaf_cat).to(device)
            t_leaf_num = torch.from_numpy(leaf_num).to(device)
            if cts:
                t_leaf_cat[:, 4:8, 1] = 0
                t_leaf_cat[:, 4:8, 2] = 0

            leaf_out = model(t_leaf_cat, t_leaf_num)
            leaf_vals = leaf_out["value"].squeeze(-1).cpu().numpy()

            for idx, (w_i, our_i, opp_i) in enumerate(rollout_res["leaf_coords"]):
                terminal_vals[w_i, our_i, opp_i] = leaf_vals[idx]

        M = np.nanmean(terminal_vals, axis=0)

    # 6. Solve root matrix game with Regret Matching
    pi_search, pi_opp, game_val = solve_regret_matching(M, opp_prior=opp_priors)

    # 7. Blend Search Equilibrium with Network Prior
    pi_final = (1.0 - blend_alpha) * our_priors + blend_alpha * pi_search
    pi_final /= pi_final.sum()

    best_idx = int(np.argmax(pi_final))
    chosen_joint = our_actions[best_idx]

    # Map chosen action back to legal choice tuples
    choice_0, choice_1 = battle.actions_17_to_choices(our_side, chosen_joint[0], chosen_joint[1])

    return {
        "action": chosen_joint,
        "choices": [choice_0, choice_1],
        "best_index": best_idx,
        "pi_final": pi_final,
        "pi_search": pi_search,
        "pi_net": our_priors,
        "game_value": game_val,
        "payoff_matrix": M,
        "our_candidates": our_cands,
        "opp_candidates": opp_cands,
        "n_worlds": n_worlds,
        "n_leaves": n_leaves,
        "is_endgame_solve": False,
    }

