"""Battle state and the turn function (phase 1: damaging moves only, no switching, no abilities/items).

Flat actor ids: side * 2 + slot, so ids 0,1 are side 0's actives and 2,3 side 1's. Only the four
active Pokemon exist in the state; the bench comes with switching.
"""
import jax
import jax.numpy as jnp
import numpy as np

from . import tables as T
from .damage import damage

LEVEL = 50


def make_tables(t=None):
    """Move/type tables as device arrays, passed to `step` as one pytree."""
    t = t or T.load()
    chart = np.zeros_like(t.chart, np.int32)  # +1 super effective, -1 resisted, -99 immune
    chart[t.chart == 2] = 1
    chart[t.chart == 0.5] = -1
    chart[t.chart == 0] = -99
    return {k: jnp.asarray(v) for k, v in dict(
        chart=chart, mv_type=t.mv_type, mv_cat=t.mv_cat, mv_bp=t.mv_bp, mv_prio=t.mv_prio,
        mv_target=t.mv_target, sp_types=t.sp_types).items()}


def make_state(teams, t=None):
    """teams: [side0, side1], each a list of 4 dicts(species, moves, evs, nature) -> state of numpy arrays."""
    t = t or T.load()
    stats, species, moves = [], [], []
    for side in teams:
        for m in side[:2]:
            sid = t.species_ix[T.to_id(m["species"])]
            species.append(sid)
            stats.append(T.calc_stats(t.base[sid], [m["evs"][k] for k in T.STATS], m["nature"], LEVEL))
            moves.append([t.move_ix[T.to_id(x)] for x in m["moves"]])
    stats = np.array(stats, np.int32)
    return dict(stats=stats, hp=stats[:, 0].copy(), species=np.array(species, np.int32),
                moves=np.array(moves, np.int32))


def _type_exp(tab, mv_type, def_types):
    eff = jnp.where(def_types >= 0, tab["chart"][mv_type, jnp.maximum(def_types, 0)], 0)
    return jnp.where((eff == -99).any(), -99, eff.sum())


def step(tab, state, actions, rolls, crits, hits, tiebreak):
    """One turn. actions[a] = (move slot, foe slot); rolls[a, d] in 0..15; crits[a, d], hits[a] bool;
    tiebreak[a] random floats ordering equal priority+speed. Returns the new state."""
    state = jax.tree.map(jnp.asarray, state)
    mv = state["moves"][jnp.arange(4), actions[:, 0]]  # chosen move ids
    speed = state["stats"][:, 5]
    key = tab["mv_prio"][mv] * 100000 + speed * 4 + jnp.floor(tiebreak * 4)
    order = jnp.argsort(-key)
    types = tab["sp_types"][state["species"]]  # [4, 2]

    def act(i, hp):
        a = order[i]
        m = mv[a]
        side = a // 2
        cat, tgt = tab["mv_cat"][m], tab["mv_target"][m]
        ids = jnp.arange(4)
        foe = ids // 2 != side
        alive = hp > 0
        chosen = (1 - side) * 2 + actions[a, 1]
        other = (1 - side) * 2 + (1 - actions[a, 1])
        single = jnp.where(alive[chosen], chosen, other)
        mask = jnp.where(tgt == 0, ids == single, jnp.where(tgt == 1, foe, ids != a)) & alive
        spread = (tgt > 0) & (mask.sum() > 1)
        atk = state["stats"][a, jnp.where(cat == 0, 1, 3)]
        dfn = state["stats"][:, jnp.where(cat == 0, 2, 4)]
        mtype = tab["mv_type"][m]
        stab = jnp.where((types[a] == mtype).any(), 6144, 4096)
        texp = jax.vmap(lambda dt: _type_exp(tab, mtype, dt))(types)
        dmg = damage(LEVEL, tab["mv_bp"][m], atk, dfn, rolls[a], crits[a], stab, texp,
                     False, spread, 4096, 4096)
        go = alive[a] & hits[a] & (cat < 2)
        return jnp.where(mask & go, jnp.maximum(hp - dmg, 0), hp)

    hp = jax.lax.fori_loop(0, 4, act, state["hp"])
    return {**state, "hp": hp}
