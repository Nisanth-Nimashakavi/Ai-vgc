"""Restore actual opponent announcements after a shadow-state JSON round trip.

The native serialization deliberately omits public knowledge. Showdown's parsed
opponent objects, rather than outcomes sampled by the shadow engine, own this
memory at real decision boundaries.
"""
import re

import numpy as np


def _id(value):
    return re.sub(r'[^a-z0-9]', '', str(value or '').lower())


def observed_opponent_facts(native_facts, raw, battle, side, vocab, species_matches):
    facts = np.array(native_facts, copy=True)
    rows = facts[side, 0, 1-side]
    rows[:] = 0
    key = f'p{2-side}'
    maps = {k: {_id(v): i for i,v in enumerate(vocab[k])} for k in ('moves','items','abilities')}
    for idx, native in enumerate(raw[key]['team']):
        matches = [p for p in battle.opponent_team.values()
                   if p.revealed and species_matches(p.species, vocab['species'][native['species_id']])]
        if not matches:
            continue
        mon = matches[0]
        row = rows[idx]
        row[0], row[1] = 1, native['species_id']
        item = _id(mon.item)
        known_item = item not in ('unknownitem', 'unknown')
        row[2] = known_item
        row[3] = maps['items'].get(item, 65535)
        row[4] = row[3]
        ability = maps['abilities'].get(_id(mon.ability))
        row[5], row[6] = ability is not None, ability if ability is not None else 65535
        row[7], row[8] = bool(mon.fainted), bool(mon.active)
        row[9:13] = 65535
        moves = [maps['moves'][_id(m)] for m in mon.moves if _id(m) in maps['moves']][-4:]
        row[9:9+len(moves)] = moves
        row[19] = next((i+1 for i,p in enumerate(battle.opponent_active_pokemon)
                        if p is mon), 0)
    return facts


def merge_branch_public_facts(branch, root, side, raw=None):
    """Retain real prior announcements while allowing hypothetical new reveals.

Dynamic HP/faint/active state and species stay branch-local; no opponent stats
or unrevealed moves are copied into the learner's public memory.
"""
    result = np.array(branch, copy=True)
    for idx, (dst, prior) in enumerate(zip(result[side,0,1-side], root[side,0,1-side])):
        if not prior[0]:
            continue
        if not dst[0]:
            dst[:] = prior
        dst[0] = 1
        if raw is not None:
            team = raw[f'p{2-side}']
            mon = team['team'][idx]
            dst[1], dst[7] = mon['species_id'], bool(mon['fainted'])
            dst[19] = next((i+1 for i,j in enumerate(team['active']) if j == idx),0)
            dst[8] = dst[19] != 0
        if not dst[2] and prior[2]:
            dst[2:5] = prior[2:5]
        if not dst[5] and prior[5]:
            dst[5:7] = prior[5:7]
        moves = []
        for move in (*prior[9:13], *dst[9:13]):
            if move != 65535:
                if move in moves:
                    moves.remove(move)
                moves.append(move)
        dst[9:13] = 65535
        dst[9:9+min(4,len(moves))] = moves[-4:]
    return result
