"""Pure Rust multi-world neural search engine.

Inspired by mikumiku37 (#1 bot on the VGC ladder):
- Samples up to 64 hidden-information worlds (guessed sets, moves, items, spreads)
- For each world, tests top-K joint actions against opponent's top-K candidate responses
- Executes all candidate rollouts in-process inside Rust vgc-engine (~5-10 microseconds/step)
- Encodes all non-terminal leaf states into packed policy tensors in Rust
- Evaluates all leaf states with the Neural Network Value Head in a single batched GPU forward pass
- Computes payoff matrix M across worlds and solves for optimal action via expected value or Double Oracle
"""

from __future__ import annotations

import time
import os
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from poke_env.battle import DoubleBattle
from poke_env.environment import DoublesEnv

import vgc_engine
from ai_vgc.nn.player import safe_action_to_order
from ai_vgc.nn.double_oracle import solve_hidden_world_game, solve_hidden_world_regret
from ai_vgc.nn.encode import encode
from ai_vgc.nn.model import Policy, masked, slot_b_mask
from ai_vgc.nn.search import (
    our_candidates,
    opp_candidates,
    position,
    worlds,
    _opp_candidates_from_probs,
)
from ai_vgc.nn.vgc_bridge import VGCBridge
from ai_vgc.nn.vgc_env import packed_policy_arrays, policy_vocab_tables

_VOCAB_INITIALIZED = False


def init_rust_vocab() -> None:
    """Register Python policy checkpoint categorical vocabularies with Rust vgc-engine."""
    global _VOCAB_INITIALIZED
    if _VOCAB_INITIALIZED:
        return
    tables = policy_vocab_tables()
    vgc_engine.set_global_policy_vocab(
        tables["species"], tables["items"], tables["abilities"], tables["moves"]
    )
    _VOCAB_INITIALIZED = True


def _evaluate_leaves(
    model: Policy,
    packed: dict[str, np.ndarray],
    device: torch.device,
) -> np.ndarray:
    """Evaluate packed non-terminal policy observation states with the Value Head."""
    if not packed or packed["tok_cat"].shape[0] == 0:
        return np.array([], dtype=np.float32)
    b_tensors = {k: torch.from_numpy(v) for k, v in packed.items()}
    BATCH = 1024
    N = packed["tok_cat"].shape[0]
    leaf_vals_list = []
    with torch.no_grad():
        for i in range(0, N, BATCH):
            chunk = {k: v[i : i + BATCH].to(device, non_blocking=device.type == "cuda") for k, v in b_tensors.items()}
            if device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    enc = model.encode(chunk)
                    vals = torch.sigmoid(model.value(enc[0])[:, 0]).cpu().numpy()
            else:
                enc = model.encode(chunk)
                vals = torch.sigmoid(model.value(enc[0])[:, 0]).cpu().numpy()
            leaf_vals_list.append(vals)
    return np.concatenate(leaf_vals_list) if leaf_vals_list else np.array([], dtype=np.float32)


@torch.no_grad()
def _generate_turn2_candidates(
    model: Policy,
    packed: dict[str, np.ndarray],
    leaf_indices: list[int],
    k2: int,
    device: torch.device,
    ka: int = 3,
) -> list[list[np.ndarray]]:
    """Batched policy candidate generation for Turn 2 continuation states."""
    if not leaf_indices:
        return []
    b = {k: torch.from_numpy(packed[k][leaf_indices]).to(device) for k in packed}
    m = torch.from_numpy(packed["mask"][leaf_indices]).to(device)
    enc = model.encode(b)
    lp_a_all = F.log_softmax(masked(model.slot_logits(enc, b, 0), m[:, 0]), -1)
    cls, toks, mv = enc
    n = len(leaf_indices)
    out = []
    for i in range(n):
        enc_i = (cls[i : i + 1], toks[i : i + 1], mv[i : i + 1])
        b_i = {key: v[i : i + 1] for key, v in b.items()}
        lp_a = lp_a_all[i]
        cands = []
        n_a0 = min(ka, max(1, int(m[i, 0].sum().item())))
        top_a0 = lp_a.topk(n_a0).indices.tolist()
        for a0 in top_a0:
            prev = torch.tensor([a0], device=device)
            m_b = slot_b_mask(m[i : i + 1, 1], prev)
            n_b = min(k2, max(1, int(m_b.sum().item())))
            lp_b = F.log_softmax(masked(model.slot_logits(enc_i, b_i, 1, prev), m_b), -1)[0]
            top_a1 = lp_b.topk(n_b).indices.tolist()
            for a1 in top_a1:
                cands.append((np.array([a0, a1], dtype=np.uint8), (lp_a[a0] + lp_b[a1]).item()))
        cands.sort(key=lambda c: -c[1])
        out.append([c[0] for c in cands[:k2]])
    return out


def _opp_legal_pairs(battle: Any, opp_side: int, opp_k2: int) -> list[list[tuple]]:
    """Enumerate candidate legal choice pairs for opponent on Turn 2."""
    c0 = battle.legal_choices(opp_side, 0)
    c1 = battle.legal_choices(opp_side, 1)
    if c0 and c1:
        pairs = []
        for x0 in c0:
            for x1 in c1:
                pairs.append([x0, x1])
                if len(pairs) >= opp_k2:
                    return pairs
        return pairs
    elif c0:
        return [[x0] for x0 in c0[:opp_k2]]
    elif c1:
        return [[x1] for x1 in c1[:opp_k2]]
    return [[]]


def filter_opponent_candidates(
    legal: list[tuple[tuple[str, float], Any]],
    k: int = 6,
    br: int = 0,
    ensure_switches: bool = True,
) -> list[tuple[tuple[str, float], Any]]:
    """Select opponent candidates guaranteeing top-6 minimum and full bench switch recall.

    1. Keeps at least max(6, k, br) top legal candidates by prediction head probability (when ensure_switches is True)
       or max(k, br) otherwise.
    2. Preserves full recall of all legal bench switch destinations: for each healthy bench
       Pokemon, guarantees that the highest-probability joint action switching slot 0 to that
       Pokemon is included, that the highest-probability joint action switching slot 1 to that
       Pokemon is included, and if a double switch is legal, that the highest-probability
       double switch is included.
    3. Re-normalizes candidate probabilities over the retained set.
    """
    if not legal:
        return []

    target_k = max(6, k, br) if ensure_switches else max(k, br)
    selected = list(legal[:target_k])
    selected_set = {t[0] for t, _ in selected}

    if ensure_switches:
        best_slot0_switches: dict[str, tuple[tuple[str, float], Any]] = {}
        best_slot1_switches: dict[str, tuple[tuple[str, float], Any]] = {}
        best_double_switch: tuple[tuple[str, float], Any] | None = None

        for item in legal:
            choice_str, _ = item[0]
            parts = [p.strip() for p in choice_str.split(",")]
            if len(parts) != 2:
                continue
            is_sw0 = parts[0].startswith("switch ")
            is_sw1 = parts[1].startswith("switch ")

            if is_sw0 and is_sw1:
                if best_double_switch is None:
                    best_double_switch = item
            elif is_sw0:
                target = parts[0].removeprefix("switch ")
                if target not in best_slot0_switches:
                    best_slot0_switches[target] = item
            elif is_sw1:
                target = parts[1].removeprefix("switch ")
                if target not in best_slot1_switches:
                    best_slot1_switches[target] = item

        for target, item in best_slot0_switches.items():
            has_dest = any(
                t[0].split(",")[0].strip() == f"switch {target}"
                for t, _ in selected
            )
            if not has_dest and item[0][0] not in selected_set:
                selected.append(item)
                selected_set.add(item[0][0])

        for target, item in best_slot1_switches.items():
            has_dest = any(
                len(t[0].split(",")) > 1 and t[0].split(",")[1].strip() == f"switch {target}"
                for t, _ in selected
            )
            if not has_dest and item[0][0] not in selected_set:
                selected.append(item)
                selected_set.add(item[0][0])

        if best_double_switch is not None:
            has_double = any(
                len(t[0].split(",")) == 2
                and t[0].split(",")[0].strip().startswith("switch ")
                and t[0].split(",")[1].strip().startswith("switch ")
                for t, _ in selected
            )
            if not has_double and best_double_switch[0][0] not in selected_set:
                selected.append(best_double_switch)
                selected_set.add(best_double_switch[0][0])

    total_prob = sum(t[1] for t, _ in selected) or 1.0
    return [((t[0], t[1] / total_prob), c) for t, c in selected]


def rust_search(
    model: Policy,
    opp_model: Policy | None,
    battle: DoubleBattle,
    mask: np.ndarray,
    fmt: str,
    teams_dir: str,
    rating: float = 1500.0,
    k: int = 6,
    opp_k: int = 6,
    n_worlds: int = 16,
    depth: int = 1,
    k2: int = 3,
    opp_k2: int = 3,
    refine: int = 0,
    prior: float = 0.1,
    double_oracle: bool = True,
    ots: bool = False,
    device: torch.device | None = None,
    bridge: VGCBridge | None = None,
    seeds: int = 16,
    rng_seed: int | None = None,
    solver: str = "regret",
    worst_case_weight: float = 0.5,
    regret_iterations: int = 2048,
    value_model: Policy | None = None,
    seeds2: int = 1,
    chance_nodes: bool = False,
    opp_br: int = 0,
    ensure_switches: bool = True,
) -> tuple[np.ndarray, dict] | None:
    """One-turn multi-world neural search executed entirely in Rust with batched GPU value evaluation.

    Returns:
        (chosen_action_array, details_dict) or None if search cannot proceed.
    """
    init_rust_vocab()
    value_model = model if value_model is None else value_model
    public_leaf = os.environ.get('VGC_PUBLIC_LEAF', '0') == '1'
    if public_leaf and (depth not in (1, 2) or ots):
        raise ValueError('public leaf adapter requires depth-one/two CTS search')
    if public_leaf and depth == 2 and (solver != 'lp' or not double_oracle):
        raise ValueError('public depth-two currently requires the LP solver')
    if seeds2 < 1:
        raise ValueError('second-turn chance samples must be positive')
    if seeds < 1:
        raise ValueError("search seeds must be positive")
    if chance_nodes and depth != 1:
        # Chance-node analytical frontier branching is depth-one only.
        # If deeper lookahead is requested, fall back to sampled rollouts.
        chance_nodes = False
    # Common random numbers across action pairs, independently repeated for
    # each chance sample. These are not additional hidden-information worlds.
    rng = np.random.default_rng(rng_seed)
    chance_seeds = rng.integers(0, 2**63, size=seeds).tolist()
    t_start = time.perf_counter()

    if bridge is None:
        bridge = VGCBridge()

    obs = encode(battle, rating)
    ours = our_candidates(model, obs | {"mask": mask}, mask, k, ka=max(4, k))
    if len(ours) < 2:
        return None

    choices = [
        safe_action_to_order(a, battle).message.removeprefix("/choose ")
        for a, _ in ours
    ]

    # Sample hidden-information worlds
    sampled_worlds = worlds(battle, teams_dir, n_worlds, fmt, include_preview=True, rng=rng)
    if not sampled_worlds:
        sampled_worlds = [({}, 1.0)]

    # Precompute opponent model probabilities once from the current observation
    opp_probs = [None, None]
    if opp_model is not None and opp_model.config.get("aux"):
        opp_device = next(opp_model.parameters()).device
        b = {key: torch.from_numpy(np.ascontiguousarray(v))[None].to(opp_device) for key, v in obs.items()}
        with torch.no_grad():
            mv, tg = opp_model.opp_logits(opp_model.encode(b), b)
            mv, tg = F.softmax(mv[0], -1).cpu().numpy(), F.softmax(tg[0], -1).cpu().numpy()
            opp_probs = [(mv[0], tg[0]), (mv[1], tg[1])]

    world_battles = []
    world_weights = []
    world_theirs = []
    opp_choices_tuples = []
    our_choices_tuples = None
    us_is_p1 = True
    world_ids = []
    reconstruction_errors = []
    public_encoders = []

    for hidden_id, (world_dict, w_prob) in enumerate(sampled_worlds):
        state = position(battle, fmt, teams_dir, world_dict)
        if state is None:
            reconstruction_errors.append(f"world {hidden_id}: incomplete position")
            continue
        try:
            replicas = [bridge._battle(state, seed)
                        for seed in (chance_seeds[:1] if chance_nodes else chance_seeds)]
            b, us_p1 = replicas[0]
        except Exception as error:
            reconstruction_errors.append(f"world {hidden_id}: {error!r}")
            continue

        us_is_p1 = us_p1
        p1_state, p2_state, _ = bridge._sides(state)
        our_state_side = p1_state if us_is_p1 else p2_state
        their_state_side = p2_state if us_is_p1 else p1_state

        if our_choices_tuples is None:
            try:
                our_choices_tuples = [
                    bridge._choice(c, our_state_side, 0 if us_is_p1 else 1)
                    for c in choices
                ]
            except Exception as error:
                raise ValueError(f"cannot reconstruct our candidate choices: {error!r}") from error

        invalid = [choices[i] for i, c in enumerate(our_choices_tuples)
                   if not bridge.legal_joint(b, 0 if us_is_p1 else 1, c)]
        if invalid:
            legal = [b.legal_choices(0 if us_is_p1 else 1, slot) for slot in (0, 1)]
            raise ValueError(f"live candidates illegal in reconstructed Rust state: {invalid}; legal={legal}")

        opponent_side = 1 if us_is_p1 else 0
        mega_slots = {slot for slot in (0, 1) if any(c[0] == "mega" for c in b.legal_choices(opponent_side, slot))}
        theirs = _opp_candidates_from_probs(battle, obs, state, None, opp_probs,
                                            mega_slots=mega_slots)
        if not theirs:
            reconstruction_errors.append(f"world {hidden_id}: no opponent candidates")
            continue

        try:
            opp_tuples = [
                bridge._choice(t, their_state_side, 1 if us_is_p1 else 0)
                for t, _ in theirs
            ]
            raw_legal = [(t, c) for t, c in zip(theirs, opp_tuples)
                         if bridge.legal_joint(b, 1 if us_is_p1 else 0, c)]
            if not raw_legal:
                raise ValueError("no legal opponent candidates in reconstructed state")
            legal = filter_opponent_candidates(raw_legal, k=opp_k, br=opp_br, ensure_switches=ensure_switches)
            theirs, opp_tuples = map(list, zip(*legal))
        except Exception as error:
            reconstruction_errors.append(f"world {hidden_id}: opponent choices: {error!r}")
            continue

        for replica, _ in replicas:
            world_battles.append(replica)
            world_weights.append(w_prob / (1 if chance_nodes else seeds))
            world_theirs.append(theirs)
            opp_choices_tuples.append(opp_tuples)
            world_ids.append(hidden_id)
            if public_leaf:
                from ai_vgc.nn.public_leaf import PublicLeafEncoder
                public_encoders.append(PublicLeafEncoder(battle, state, replica, rating))

    if reconstruction_errors:
        raise ValueError(f"incomplete reconstructed search worlds: {reconstruction_errors}")
    if not world_battles or our_choices_tuples is None:
        raise ValueError(f"no reconstructed search worlds: {reconstruction_errors}")

    # Re-normalize world weights over successfully reconstructed worlds
    total_w = sum(world_weights) or 1.0
    world_weights = [w / total_w for w in world_weights]

    t_prep = time.perf_counter() - t_start

    # Run multi-world rollouts in Rust
    t0 = time.perf_counter()
    our_side_code = 0 if us_is_p1 else 1
    if chance_nodes:
        # Chance-node search: each cell is the exact expectation over the turn's
        # enumerated chance frontier (damage/crit/accuracy/secondary), no dice.
        res = vgc_engine.multi_world_rollout_chance(
            world_battles, our_choices_tuples, opp_choices_tuples,
            our_side=our_side_code, rating=rating, ots=ots)
        t_rust = time.perf_counter() - t0
        if device is None:
            device = next(value_model.parameters()).device
        t0 = time.perf_counter()
        outcome_vals = np.array(res["terminal_values"], dtype=np.float32)
        nt = np.array(res["non_terminal_indices"], dtype=np.intp)
        raw_packed = packed = packed_policy_arrays(res["packed"]) if len(nt) else {}
        if len(nt):
            if public_leaf:
                # Project every enumerated outcome leaf onto public-only info so
                # the value net scores the frontier the way the live ladder bot
                # sees it (closed sheets), not with the sampled world's full set.
                from ai_vgc.nn.public_leaf import encode_public_leaves
                packed = encode_public_leaves(public_encoders, res, packed)
            outcome_vals[nt] = _evaluate_leaves(value_model, packed, device)
        prob = np.array(res["outcome_prob"], dtype=np.float32)
        offs = np.asarray(res["cell_offsets"], dtype=np.intp)
        cnts = np.asarray(res["cell_counts"], dtype=np.intp)
        # Fold each cell's outcome block into prob-weighted expected value via a
        # prefix sum (frontier probabilities already sum to one within a cell).
        csum = np.concatenate(([np.float32(0)], np.cumsum(prob * outcome_vals, dtype=np.float32)))
        terminal_values = (csum[offs + cnts] - csum[offs]).astype(np.float32)
        non_terminal_indices = np.array([], dtype=np.intp)
        res = {"world_offsets": res["world_offsets"], "world_opp_counts": res["world_opp_counts"],
               "total_rollouts": int(cnts.sum())}
        t_eval = time.perf_counter() - t0
    else:
        res = vgc_engine.multi_world_rollout(
            world_battles,
            our_choices_tuples,
            opp_choices_tuples,
            our_side=our_side_code,
            rating=rating,
            ots=ots,
        )
        t_rust = time.perf_counter() - t0

        # Evaluate non-terminal leaf states with PyTorch Value Head
        if device is None:
            device = next(value_model.parameters()).device

        t0 = time.perf_counter()
        terminal_values = np.array(res["terminal_values"], dtype=np.float32)
        non_terminal_indices = np.array(res["non_terminal_indices"], dtype=np.intp)
        packed = packed_policy_arrays(res["packed"]) if len(non_terminal_indices) > 0 else {}

        raw_packed = packed
        if len(non_terminal_indices) > 0:
            if public_leaf:
                from ai_vgc.nn.public_leaf import encode_public_leaves
                packed = encode_public_leaves(public_encoders, res, packed)
            terminal_values[non_terminal_indices] = _evaluate_leaves(value_model, packed, device)
        t_eval = time.perf_counter() - t0

    depth2_info = None
    if public_leaf and depth == 2 and len(non_terminal_indices):
        from ai_vgc.nn.public_depth2 import public_second_turn
        terminal_values, depth2_info = public_second_turn(
            model, opp_model, value_model, device, res, raw_packed, public_encoders,
            terminal_values, world_weights, world_theirs, k2, opp_k2, seeds2, rng)
        t_eval += depth2_info['seconds']

    # Assemble initial payoff matrix across worlds
    world_offsets = res["world_offsets"]
    world_opp_counts = res["world_opp_counts"]
    K = len(our_choices_tuples)
    expected_values = np.zeros(K, dtype=np.float32)

    first_opp_k = world_opp_counts[0]
    M_agg = np.zeros((K, first_opp_k), dtype=np.float32)
    q_agg = np.zeros(first_opp_k, dtype=np.float32)

    for w_idx, w_prob in enumerate(world_weights):
        offset = world_offsets[w_idx]
        kw = world_opp_counts[w_idx]
        M_w = terminal_values[offset : offset + K * kw].reshape(K, kw)
        theirs_w = world_theirs[w_idx]
        q_w = np.array([p for _, p in theirs_w], dtype=np.float32)
        q_w /= q_w.sum() or 1.0

        expected_values += w_prob * (M_w @ q_w)

        if kw == first_opp_k:
            M_agg += w_prob * M_w
            q_agg += w_prob * q_w

    q_agg /= q_agg.sum() or 1.0
    logp = np.array([lp for _, lp in ours], dtype=np.float32)

    total_rollouts = res["total_rollouts"] + (depth2_info["simulations"] if depth2_info else 0)

    # Optional Depth 2 (2-turn lookahead)
    if depth >= 2 and not public_leaf and len(non_terminal_indices) > 0:
        refine_count = refine if (0 < refine < K) else K
        top_our_indices = set(np.argsort(-(expected_values + prior * logp))[:refine_count].tolist())

        coords = res.get("non_terminal_coords", [])
        leaves = res.get("non_terminal_battles", [])

        deepen_leaf_indices = [
            i
            for i, (w_idx, our_idx, opp_idx) in enumerate(coords)
            if our_idx in top_our_indices and opp_idx < opp_k2
        ]

        if deepen_leaf_indices:
            deepen_battles = [leaves[i] for i in deepen_leaf_indices]
            turn2_cands = _generate_turn2_candidates(value_model, packed, deepen_leaf_indices, k2, device)

            our_choices_per_leaf = []
            opp_choices_per_leaf = []
            opp_side_code = 1 - our_side_code

            for j, b_leaf in enumerate(deepen_battles):
                our_pairs = [
                    b_leaf.joint_action_to_choices(our_side_code, a.tolist())
                    for a in turn2_cands[j]
                ]
                opp_pairs = _opp_legal_pairs(b_leaf, opp_side_code, opp_k2)
                our_choices_per_leaf.append(our_pairs)
                opp_choices_per_leaf.append(opp_pairs)

            t_r2 = time.perf_counter()
            res2 = vgc_engine.multi_world_rollout(
                deepen_battles,
                our_choices_per_leaf,
                opp_choices_per_leaf,
                our_side=our_side_code,
                rating=rating,
                ots=ots,
            )
            t_rust += time.perf_counter() - t_r2
            total_rollouts += res2["total_rollouts"]

            t_e2 = time.perf_counter()
            term2 = np.array(res2["terminal_values"], dtype=np.float32)
            non_term2 = np.array(res2["non_terminal_indices"], dtype=np.intp)
            if len(non_term2) > 0:
                packed2 = packed_policy_arrays(res2["packed"])
                term2[non_term2] = _evaluate_leaves(value_model, packed2, device)
            t_eval += time.perf_counter() - t_e2

            for j in range(len(deepen_battles)):
                offset2 = res2["world_offsets"][j]
                k_our2 = res2["our_choices_counts"][j]
                k_opp2 = res2["world_opp_counts"][j]
                if k_our2 > 0 and k_opp2 > 0:
                    M2 = term2[offset2 : offset2 + k_our2 * k_opp2].reshape(k_our2, k_opp2)
                    q2 = np.ones(k_opp2, dtype=np.float32) / k_opp2
                    v2 = float((M2 @ q2).max())
                else:
                    v2 = 0.5
                orig_rollout_idx = non_terminal_indices[deepen_leaf_indices[j]]
                terminal_values[orig_rollout_idx] = v2

            # Re-assemble payoff matrix across worlds with turn-2 refined values
            expected_values.fill(0.0)
            M_agg.fill(0.0)
            q_agg.fill(0.0)
            for w_idx, w_prob in enumerate(world_weights):
                offset = world_offsets[w_idx]
                kw = world_opp_counts[w_idx]
                M_w = terminal_values[offset : offset + K * kw].reshape(K, kw)
                theirs_w = world_theirs[w_idx]
                q_w = np.array([p for _, p in theirs_w], dtype=np.float32)
                q_w /= q_w.sum() or 1.0
                expected_values += w_prob * (M_w @ q_w)
                if kw == first_opp_k:
                    M_agg += w_prob * M_w
                    q_agg += w_prob * q_w
            q_agg /= q_agg.sum() or 1.0

    if double_oracle:
        matrices = [terminal_values[offset:offset + K * count].reshape(K, count)
                    for offset, count in zip(world_offsets, world_opp_counts)]
        # Average chance BEFORE the minimization: the opponent cannot choose
        # its move after seeing a future accuracy roll or critical hit.
        hidden_matrices, hidden_weights = average_chance_payoffs(matrices, world_weights, world_ids)
        if solver == "regret":
            first_indices = [world_ids.index(wid) for wid in dict.fromkeys(world_ids)]
            opponent_probs = [np.array([p for _, p in world_theirs[i]]) for i in first_indices]
            eq_strat, eq_value, convergence = solve_hidden_world_regret(
                hidden_matrices, hidden_weights, opponent_probs,
                worst_case_weight=worst_case_weight, iterations=regret_iterations, row_bonus=prior*logp)
            equilibrium = eq_strat, eq_value
            eq_info = {"solver": "hidden-world regret matching", "worst_case_weight": worst_case_weight,
                       "restricted_candidates": True, **convergence}
        elif solver == "lp":
            equilibrium = solve_hidden_world_game(hidden_matrices, hidden_weights, prior * logp)
            eq_info = {"solver": "hidden-world LP", "restricted_candidates": True}
        else:
            raise ValueError(f"unknown search solver: {solver}")
        if equilibrium is not None:
            eq_strat, eq_value = equilibrium
            score = eq_strat
        else:
            eq_strat = eq_value = None
            eq_info["fallback"] = "expected value"
            score = expected_values + prior * logp
    else:
        eq_strat = eq_value = eq_info = None
        score = expected_values + prior * logp

    # Taking argmax of a mixed equilibrium destroys its worst-case guarantee.
    # The policy prior is already included in the LP's row utility.
    if eq_strat is not None:
        eq_strat_clean = np.maximum(np.asarray(eq_strat, dtype=np.float64), 0.0)
        tot = eq_strat_clean.sum()
        if tot > 0:
            eq_strat_clean /= tot
        else:
            eq_strat_clean = np.ones(K, dtype=np.float64) / K
        best = int(np.random.choice(K, p=eq_strat_clean))
    else:
        best = int(score.argmax())
    t_total = time.perf_counter() - t_start

    details = {
        "M": M_agg,
        "q": q_agg,
        "ours": choices,
        "theirs": [t for t, _ in world_theirs[0]],
        "actions": np.stack([a for a, _ in ours]),
        "value": expected_values,
        "logp": logp,
        "obs": obs | {"mask": mask},
        "public_leaf": public_leaf,
        "depth2": depth2_info,
        "score": score,
        "selected_index": best,
        "sims": total_rollouts,
        "errors": reconstruction_errors,
        "input_choices_validated": True,
        "reconstruction_errors": reconstruction_errors,
        "n_worlds": len(set(world_ids)),
        "chance_samples": seeds,
        "chance_seeds": chance_seeds,
        "candidate_policy_mass": float(np.exp(logp).sum()),
        "timing": {
            "prep_s": t_prep,
            "rust_s": t_rust,
            "eval_s": t_eval,
            "total_s": t_total,
        },
        "eq_strat": eq_strat,
        "eq_value": eq_value,
        "eq_info": eq_info,
    }
    if os.environ.get("VGC_SEARCH_DIAGNOSTICS"):
        # Column identities differ by hidden world; M_agg alone cannot explain
        # which opponent responses were evaluated for a historical decision.
        details["world_audit"] = [
            {
                "world_id": world_ids[i],
                "weight": world_weights[i],
                "theirs": world_theirs[i],
                "payoffs": terminal_values[offset:offset + K * count].reshape(K, count),
            }
            for i, (offset, count) in enumerate(zip(world_offsets, world_opp_counts))
        ]
    return ours[best][0], details


def average_chance_payoffs(matrices, weights, world_ids):
    """Keep private-set worlds separate, averaging chance within each world."""
    grouped = {}
    for matrix, weight, world_id in zip(matrices, weights, world_ids, strict=True):
        matrix = np.asarray(matrix, dtype=np.float64)
        if world_id not in grouped:
            grouped[world_id] = [np.zeros_like(matrix), 0.0]
        total, mass = grouped[world_id]
        if total.shape != matrix.shape:
            raise ValueError("chance replicas must have identical candidate actions")
        grouped[world_id] = [total + weight * matrix, mass + weight]
    return ([total / mass for total, mass in grouped.values()],
            [mass for _, mass in grouped.values()])
