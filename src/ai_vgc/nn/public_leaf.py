"""Project one Rust turn onto the existing public Showdown observation.

The simulator keeps its private sampled sets. Only observable changes are
applied to a copy of the player's battle before using the training encoder.
This is a correctness-first, one-turn adapter, not the native training encoder.
"""
import copy
import json
import math
import hashlib
import pickle

import numpy as np
from poke_env.battle import Move

from ai_vgc.nn.encode import BOOSTS, EFFECTS, FIELDS, SIDE, STATUSES, WEATHERS, encode, vocab


def clone_public(battle):
    shell = copy.copy(battle)
    shell.logger = None
    result = copy.deepcopy(shell)
    result.logger = battle.logger
    result._spd_frozen = True
    return result


def token_order(side):
    active = [i for i in side['active'][:2] if i < len(side['team'])]
    return active + [i for i in range(len(side['team'])) if i not in active]


class PublicLeafEncoder:
    def __init__(self, battle, state, engine_root, rating):
        self.battle, self.state, self.rating = battle, state, rating
        # Serialize once per world. Unpickling preserves the shared Pokemon
        # references between team and active slots, while avoiding repeated
        # Python-level deepcopy walks for every action/chance leaf.
        self.template = pickle.dumps(clone_public(battle), protocol=5)
        self.root = json.loads(engine_root.to_json())
        self.us = 0 if state['role'] == 'p1' else 1
        self.inverse = {key: {v: k for k, v in values.items()} for key, values in vocab().items()}

    def project(self, leaf, packed):
        raw = json.loads(leaf.to_json()) if not isinstance(leaf, dict) else leaf
        b = pickle.loads(self.template)
        advanced = raw['turn'] != self.root['turn']
        b._turn = raw['turn']
        for relative, side_name in enumerate(('us', 'them')):
            ours = relative == 0
            engine_side = self.us if ours else 1 - self.us
            key = 'p1' if engine_side == 0 else 'p2'
            before_side, after_side = self.root[key], raw[key]
            team = b.team if ours else b.opponent_team
            preview = [] if ours else list(b.teampreview_opponent_team)
            by_name = {m.name: m for m in [*preview, *team.values()]}
            source_mons = self.state['sides'][side_name]['mons']
            known = {m.name for m in team.values()}
            ordered = token_order(after_side)
            token_for = {idx: relative * 6 + n for n, idx in enumerate(ordered)}
            active = {}
            role = b.player_role if ours else b.opponent_role
            for idx, (before, after, source) in enumerate(zip(before_side['team'], after_side['team'], source_mons)):
                name = source['set']['name']
                mon = by_name.get(name)
                if mon is None:
                    raise ValueError(f'public projection cannot identify {name}')
                slot = next((s for s, i in enumerate(after_side['active'][:2]) if i == idx), None)
                spent = [max(0, x-y) for x, y in zip(before['pp'], after['pp'])]
                entered = slot is not None and idx not in before_side['active'][:2]
                observable = name in known or slot is not None or any(spent) or before['current_hp'] != after['current_hp']
                if not ours and not observable:
                    continue
                if not ours and name not in known:
                    team[f'{role}: {name}'] = mon
                    known.add(name)
                    mon._revealed = True
                mon._active = slot is not None
                if slot is not None:
                    active[f'{role}{"ab"[slot]}'] = mon
                if entered:
                    mon._revealed = True
                token = token_for[idx]
                numeric, categorical = packed['tok_num'][token], packed['tok_cat'][token]
                if before['species_id'] != after['species_id']:
                    species = self.inverse['species'][int(categorical[0])]
                    mon.forme_change(species)
                    # Our request updates our species; opponent formechange does not.
                    if ours:
                        mon._species = species
                if ours:
                    mon._stats = after['stats'].copy()
                    mon._max_hp, mon._current_hp = after['stats']['hp'], after['current_hp']
                elif before['current_hp'] != after['current_hp'] or name not in {m.name for m in self.battle.opponent_team.values()}:
                    mon._max_hp = 100
                    mon._current_hp = math.ceil(100 * after['current_hp'] / max(1, after['stats']['hp']))
                if after['fainted']:
                    mon._current_hp = 0
                if before['status'] != after['status'] or after['fainted']:
                    mon._status = next((s for s, v in zip(STATUSES, numeric[37:44]) if v), None)
                    mon._status_counter = 0
                mon._boosts = dict(zip(BOOSTS, after['boosts']))
                if advanced or before['volatiles'] != after['volatiles']:
                    for effect, present in zip(EFFECTS, numeric[51:71]):
                        if present:
                            mon._effects[effect] = mon._effects.get(effect, 0) + int(advanced)
                        else:
                            mon._effects.pop(effect, None)
                    mon._active_turns = 1 if numeric[71] else 2
                    mon._protect_counter = int(round(float(numeric[72]) * 3))
                    mon._must_recharge = bool(numeric[73])
                    slot_id = after.get('charging_move_slot', 255)
                    mon._preparing_move = Move(source['set']['moves'][slot_id], gen=9) if numeric[74] and slot_id < len(source['set']['moves']) else None
                # Do not reveal unexecuted choices (flinches, sleep, fainting).
                for k, amount in enumerate(spent):
                    if not amount or k >= len(source['set']['moves']):
                        continue
                    mid = source['set']['moves'][k]
                    if mid not in mon.moves:
                        mon._moves[mid] = Move(mid, gen=9)
                    mon.moves[mid]._current_pp = max(0, mon.moves[mid].current_pp - amount)
                    for move in mon.moves.values():
                        move._is_last_used = move.id == mid
                # A world guess alone never reveals an item or ability. Changes
                # such as consumption and Mega evolution are observable.
                if before['item_id'] != after['item_id']:
                    mon._item = self.inverse['items'].get(int(categorical[1]), '') or None
                if before['ability_id'] != after['ability_id']:
                    ability = self.inverse['abilities'].get(int(categorical[2]))
                    if ability:
                        mon._ability = ability
            if ours:
                b._active_pokemon = active
                b._used_mega_evolve = after_side['conditions']['mega_used']
            else:
                b._opponent_active_pokemon = active
                b._opponent_used_mega_evolve = after_side['conditions']['mega_used']

        # Recompute damage/KO/speed features through the exact training encoder
        # using the projected public HP, moves, fields and stat boosts.
        global_features = packed['glob']
        for values, enums, section in [(b.weather, WEATHERS, global_features[:8]),
                                        (b.fields, FIELDS, global_features[8:16]),
                                        (b.side_conditions, SIDE, global_features[16:24]),
                                        (b.opponent_side_conditions, SIDE, global_features[24:32])]:
            for enum, present in zip(enums, section):
                if present:
                    values.setdefault(enum, b.turn)
                else:
                    values.pop(enum, None)
        return b

    def encode(self, leaf, packed):
        return encode(self.project(leaf, packed), self.rating)


def encode_public_leaves(encoders, result, packed, *, cache=True):
    observations = []
    encoded = {}
    for i, (leaf, coords) in enumerate(zip(result['non_terminal_battles'], result['non_terminal_coords'])):
        encoder = encoders[coords[0]]
        public = encoder.project(leaf, {k: v[i] for k, v in packed.items()})
        # Cache only identical complete public objects, never hidden-world
        # equivalence heuristics. The cache is scoped to this single decision.
        key = (encoder.rating, hashlib.sha256(pickle.dumps(public, protocol=5)).digest()) if cache else None
        if not cache or key not in encoded:
            observation = encode(public, encoder.rating)
            if cache:
                encoded[key] = observation
        else:
            observation = encoded[key]
        observations.append(observation)
    return {k: np.stack([obs[k] for obs in observations]) for k in observations[0]}
