"""Experimental second-turn search using public observations and Rust continuations.

Continuation strategies are shared across identical public information states.
Belief weights condition the root world weights on the opponent-head action
probabilities; this is an approximate belief update, not a full extensive-form
solution. Replacement boundaries retain the first-turn value and are counted.
"""
import hashlib
import itertools
import json
import pickle
import time

import numpy as np
import vgc_engine
import torch

from ai_vgc.nn.encode import N_ACT, encode
from ai_vgc.nn.public_leaf import PublicLeafEncoder, encode_public_leaves
from ai_vgc.nn.vgc_bridge import VGCBridge
from ai_vgc.nn.vgc_env import packed_policy_arrays, _action_index
from ai_vgc.nn.double_oracle import solve_hidden_world_game


def public_action_map(public, adapter, engine):
    """Map engine indices to the public policy's roster/move ordering exactly."""
    raw = json.loads(engine.to_json())
    side = raw['p1' if adapter.us == 0 else 'p2']
    sources = adapter.state['sides']['us']['mons']
    team = list(public.team.values())
    names = {m.name: i for i, m in enumerate(team)}
    mask = np.zeros((2, N_ACT), dtype=bool)
    maps = []
    for slot in range(2):
        mapping = {}
        choices = engine.legal_choices(adapter.us, slot)
        if not choices:
            choices = [('pass', slot, 0, -1, -1)]
        for choice in choices:
            kind, actor, idx, ts, target = choice
            if kind == 'switch':
                action = names[sources[idx]['set']['name']] + 1
            elif kind in ('move', 'mega'):
                monidx = side['active'][slot]
                name = sources[monidx]['set']['name']
                moves = list(team[names[name]].moves)
                if idx == 4:  # Struggle uses the exclusive first move plane.
                    moveidx = 0
                else:
                    mid = sources[monidx]['set']['moves'][idx]
                    if mid not in moves:
                        raise ValueError(f'own move absent from public continuation: {mid}')
                    moveidx = moves.index(mid)
                action = _action_index((kind, actor, moveidx, ts, target), adapter.us)
            elif kind == 'pass':
                action = 0
            else:
                raise ValueError(f'unsupported public continuation action: {choice}')
            if action is None or action >= N_ACT:
                raise ValueError(f'cannot map public continuation action: {choice}')
            if action in mapping and mapping[action] != choice:
                raise ValueError('distinct engine choices collide in public action mapping')
            mapping[action] = choice
            mask[slot, action] = True
        maps.append(mapping)
    return mask, maps


def opponent_pairs(public, adapter, engine, probs, k):
    """Rank legal opponent joint replies by the existing public opponent head."""
    raw = json.loads(engine.to_json())
    sidecode = 1 - adapter.us
    side = raw['p1' if sidecode == 0 else 'p2']
    sources = adapter.state['sides']['them']['mons']
    options = []
    for slot in range(2):
        legal = engine.legal_choices(sidecode, slot) or [('pass', slot, 0, -1, -1)]
        mon = public.opponent_active_pokemon[slot]
        known = list(mon.moves) if mon else []
        mv, tg = probs[slot] if probs[slot] is not None else (None, None)
        scored = []
        for c in legal:
            kind, _, index, target_side, target_slot = c
            p = 1.0
            if mv is not None:
                if kind == 'switch':
                    p = float(mv[4]) / max(1, sum(x[0] == 'switch' for x in legal))
                elif kind in ('move', 'mega'):
                    source = sources[side['active'][slot]]['set']['moves']
                    mid = source[index] if index < len(source) else 'struggle'
                    if mid in known:
                        j = known.index(mid)
                        p = float(mv[j])
                        if target_side >= 0:
                            target_index = target_slot if target_side != sidecode else 2
                            p *= float(tg[j][target_index])
                    else:
                        p = float(mv[:len(known)].mean()) if known else 1.0
                        if target_side >= 0:
                            p /= max(1, sum(x[0] == kind and x[2] == index for x in legal))
            scored.append((c, max(p, 1e-12)))
        options.append(scored)
    pairs = [(list((a, b)), pa*pb) for (a, pa), (b, pb) in itertools.product(*options)
             if VGCBridge.legal_joint(engine, sidecode, [a, b])]
    pairs.sort(key=lambda row: -row[1])
    if not pairs:
        raise ValueError('no legal opponent continuation pairs')
    pairs = pairs[:k]
    q = np.array([p for _, p in pairs], dtype=float)
    return [c for c, _ in pairs], q / q.sum()


def reseed(engine, seed):
    raw = json.loads(engine.to_json())
    if set(raw['rng']) != {'Splitmix'}:
        raise ValueError('second-turn RNG reseeding only validated for Splitmix')
    raw['rng'] = {'Splitmix': int(seed)}
    return type(engine).from_json(json.dumps(raw))


def shared_continuation_values(payoffs, masses):
    """One strategy across indistinguishable worlds, then back up each payoff."""
    solved = solve_hidden_world_game(payoffs, masses)
    if solved is None:
        raise ValueError('second-turn information-set LP failed')
    strategy, _ = solved
    return np.array([float((strategy @ matrix).min()) for matrix in payoffs])


@torch.no_grad()
def public_second_turn(model, opp_model, value_model, device, result, raw_packed,
                       encoders, terminal_values, world_weights, world_theirs,
                       k2, opp_k2, seeds2, rng):
    from ai_vgc.nn.rust_search import _generate_turn2_candidates, _evaluate_leaves
    from ai_vgc.nn.search import _opp_probs_batch
    started = time.perf_counter()
    stats = dict(nodes=len(result['non_terminal_indices']), expanded=0,
                 replacement_boundaries=0, unadvanced_boundaries=0,
                 information_sets=0, simulations=0, seeds=seeds2,
                 belief_update='root world weight times opponent-head root action probability')
    nodes = []
    groups = {}
    for i, (leaf, coord) in enumerate(zip(result['non_terminal_battles'], result['non_terminal_coords'])):
        w, root_action, root_reply = coord
        adapter = encoders[w]
        if leaf.needs_replacements():
            stats['replacement_boundaries'] += 1
            continue
        if leaf.turn <= adapter.root['turn']:
            stats['unadvanced_boundaries'] += 1
            continue
        public = adapter.project(leaf, {key: values[i] for key, values in raw_packed.items()})
        mask, mapping = public_action_map(public, adapter, leaf)
        obs = encode(public, adapter.rating)
        obs['mask'] = mask
        # Public history and own root action only; no sampled set/RNG identity.
        key = (root_action, hashlib.sha256(pickle.dumps(public, protocol=5) + mask.tobytes()).digest())
        if key not in groups:
            groups[key] = dict(obs=obs, nodes=[], public=public)
        node = dict(index=i, leaf=leaf, public=public, adapter=adapter, mapping=mapping,
                    mass=world_weights[w] * world_theirs[w][root_reply][1], group=key)
        groups[key]['nodes'].append(len(nodes))
        nodes.append(node)
    stats['expanded'] = len(nodes)
    stats['information_sets'] = len(groups)
    if not nodes:
        stats['seconds'] = time.perf_counter()-started
        return terminal_values, stats
    grouped = list(groups.values())
    # Public masks are in training action order; never use native indices here.
    for start in range(0, len(grouped), 64):
        chunk = grouped[start:start+64]
        packed = {key: np.stack([g['obs'][key] for g in chunk]) for key in chunk[0]['obs']}
        candidates = _generate_turn2_candidates(value_model, packed, list(range(len(chunk))), k2, device, ka=max(4,k2))
        probs = _opp_probs_batch(opp_model, [g['obs'] for g in chunk]) if opp_model is not None and opp_model.config.get('aux') else [[None,None] for _ in chunk]
        for group, actions, pred in zip(chunk, candidates, probs):
            group['actions'], group['probs'] = actions, pred
    for node in nodes:
        group = groups[node['group']]
        pairs = [[node['mapping'][slot][int(action[slot])] for slot in range(2)] for action in group['actions']]
        if not all(VGCBridge.legal_joint(node['leaf'], node['adapter'].us, p) for p in pairs):
            raise ValueError('illegal public second-turn candidate')
        node['ours'] = pairs
        node['theirs'], node['q'] = opponent_pairs(node['public'], node['adapter'], node['leaf'], group['probs'], opp_k2)
    chance = rng.integers(0, 2**63, size=seeds2).tolist()
    # Limit the number of Python public battle objects retained per batch.
    for start in range(0, len(nodes), 8):
        chunk = nodes[start:start+8]
        engines, ours, theirs, adapters = [], [], [], []
        for node in chunk:
            for seed in chance:
                leaf = reseed(node['leaf'], seed)
                engines.append(leaf); ours.append(node['ours']); theirs.append(node['theirs'])
                adapters.append(PublicLeafEncoder(node['public'], node['adapter'].state, leaf, node['adapter'].rating))
        out = vgc_engine.multi_world_rollout(engines, ours, theirs, our_side=encoders[0].us,
                                           rating=encoders[0].rating, ots=False)
        values = np.asarray(out['terminal_values'], dtype=float)
        idx = np.asarray(out['non_terminal_indices'], dtype=int)
        if len(idx):
            packed = encode_public_leaves(adapters, out, packed_policy_arrays(out['packed']))
            values[idx] = _evaluate_leaves(value_model, packed, device)
        if not np.isfinite(values).all():
            raise ValueError('nonfinite second-turn values')
        stats['simulations'] += out['total_rollouts']
        for j, node in enumerate(chunk):
            matrices = []
            for s in range(seeds2):
                offset = out['world_offsets'][j*seeds2+s]
                matrices.append(values[offset:offset+len(node['ours'])*len(node['theirs'])].reshape(len(node['ours']),len(node['theirs'])))
            node['payoff'] = np.mean(matrices, axis=0)
    for group in grouped:
        group_nodes = [nodes[j] for j in group['nodes']]
        backed_up = shared_continuation_values([n['payoff'] for n in group_nodes], [n['mass'] for n in group_nodes])
        for node, value in zip(group_nodes, backed_up):
            index = result['non_terminal_indices'][node['index']]
            terminal_values[index] = value
    stats['seconds'] = time.perf_counter()-started
    return terminal_values, stats
