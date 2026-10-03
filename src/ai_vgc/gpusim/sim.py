"""Battle state and the turn function (phase 1b: damage, stat stages, brn/par/psn/tox, secondaries;
no switching, abilities, items, sleep/freeze or volatiles yet).

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
    names = ("mv_type mv_cat mv_bp mv_prio mv_target sp_types mv_boost mv_boost_self mv_self_boost mv_status "
             "mv_sec_chance mv_sec_status mv_sec_boost mv_sec_self_boost mv_imm mv_ign_def mv_ign_off mv_powder").split()
    tab = {k: jnp.asarray(getattr(t, k)) for k in names}
    # type immunity to each status: brn Fire, par Electric, psn/tox Poison+Steel (index = status code)
    imm = np.zeros((5, len(t.types)), bool)
    for code, typs in {1: ["Fire"], 2: ["Electric"], 3: ["Poison", "Steel"], 4: ["Poison", "Steel"]}.items():
        imm[code, [t.types.index(x) for x in typs]] = True
    return {**tab, "chart": jnp.asarray(chart), "status_imm": jnp.asarray(imm)}


def make_state(teams, t=None, status=None):
    """teams: [side0, side1], each a list of dicts(species, moves, evs, nature); the first two of each
    side are active. status: optional [4] status codes."""
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
                moves=np.array(moves, np.int32), boosts=np.zeros((4, 5), np.int32),
                status=np.zeros(4, np.int32) if status is None else np.asarray(status, np.int32),
                tox=np.zeros(4, np.int32))


def stage(stat, boost):
    """Showdown calculateStat boost step: floor(stat * table[boost]) with table 2+b/2 or 2/(2-b)."""
    return jnp.where(boost >= 0, stat * (2 + boost) // 2, stat * 2 // (2 - boost))


def _type_exp(tab, mv_type, def_types):
    eff = jnp.where(def_types >= 0, tab["chart"][mv_type, jnp.maximum(def_types, 0)], 0)
    return jnp.where((eff == -99).any(), -99, eff.sum())


def _speeds(state):
    spe = stage(state["stats"][:, 5], state["boosts"][:, 4])
    return jnp.where(state["status"] == 2, spe // 2, spe)


def step(tab, state, actions, rng):
    """One turn. actions[a] = (move slot, foe slot). rng: dict with rolls[a, d] 0..15, crits[a, d] bool,
    hits[a] bool, sec[a, d] 0..99 (secondary fires if below its chance), skip[a] bool (fully paralysed), tiebreak[a] float."""
    state = jax.tree.map(jnp.asarray, state)
    mv = state["moves"][jnp.arange(4), actions[:, 0]]  # chosen move ids
    prio = tab["mv_prio"][mv] * 1000000 + jnp.floor(rng["tiebreak"] * 4)  # tiebreak in [0,1): higher moves first on equal speed
    types = tab["sp_types"][state["species"]]  # [4, 2]
    ids = jnp.arange(4)

    def set_status(st, code, who):
        """Apply status `code` to targets `who` (bool[4]) where legal; returns updated status/tox."""
        blocked = tab["status_imm"][code][jnp.maximum(types, 0)] & (types >= 0)
        ok = who & (st["status"] == 0) & ~blocked.any(axis=1) & (code > 0)
        return {**st, "status": jnp.where(ok, code, st["status"]), "tox": jnp.where(ok, 0, st["tox"])}

    def add_boost(boosts, who, delta):
        return jnp.where(who[:, None], jnp.clip(boosts + delta, -6, 6), boosts)

    def act(i, carry):
        st, done = carry
        # Showdown re-sorts the remaining actions after speed changes, so pick the next actor now
        a = jnp.argmax(jnp.where(done, -1, prio + _speeds(st) * 4))
        done = done.at[a].set(True)
        m = mv[a]
        side = a // 2
        cat, tgt = tab["mv_cat"][m], tab["mv_target"][m]
        foe = ids // 2 != side
        alive = st["hp"] > 0
        chosen = (1 - side) * 2 + actions[a, 1]
        other = (1 - side) * 2 + (1 - actions[a, 1])
        single = jnp.where(alive[chosen], chosen, other)
        aim = jnp.where(tab["mv_boost_self"][m] | (tgt == 3), a, single)
        mask = jnp.where(tgt == 0, ids == aim, jnp.where(tgt == 1, foe, ids != a)) & alive
        mask = jnp.where(tgt == 3, ids == a, mask)
        spread = (tgt > 0) & (mask.sum() > 1)
        go = alive[a] & rng["hits"][a] & ~((st["status"][a] == 2) & rng["skip"][a])
        mtype = tab["mv_type"][m]
        # type immunity (damaging moves, Thunder Wave) removes a target from the move entirely
        texp = jax.vmap(lambda dt: _type_exp(tab, mtype, dt))(types)
        grass = (types == T.load().types.index("Grass")).any(axis=1)  # powder moves can't hit Grass types
        immune = (((texp == -99) & tab["mv_imm"][m]) | (tab["mv_powder"][m] & grass)) & (tgt != 3)
        mask = mask & ~immune & go
        # damage
        phys = cat == 0
        atk_b = jnp.where(tab["mv_ign_off"][m], 0, st["boosts"][a, jnp.where(phys, 0, 2)])
        dfn_b = jnp.where(tab["mv_ign_def"][m], 0, st["boosts"][:, jnp.where(phys, 1, 3)])
        atk = stage(st["stats"][a, jnp.where(phys, 1, 3)], atk_b)
        dfn = stage(st["stats"][:, jnp.where(phys, 2, 4)], dfn_b)
        stab = jnp.where((types[a] == mtype).any(), 6144, 4096)
        dmg = damage(LEVEL, tab["mv_bp"][m], atk, dfn, rng["rolls"][a], rng["crits"][a], stab, texp,
                     (st["status"][a] == 1) & phys, spread, 4096, 4096)
        hit = mask & (cat < 2)
        hp = jnp.where(hit, jnp.maximum(st["hp"] - dmg, 0), st["hp"])
        st = {**st, "hp": hp}
        hit_alive = hit & (hp > 0)
        # secondary effects on surviving targets
        proc = hit_alive & (rng["sec"][a] < tab["mv_sec_chance"][m])
        st = set_status(st, tab["mv_sec_status"][m], proc)
        boosts = add_boost(st["boosts"], proc, tab["mv_sec_boost"][m])
        # status moves: boosts on the targets, then inflicted status
        smask = mask & (cat == 2)
        boosts = add_boost(boosts, smask, tab["mv_boost"][m])
        st = set_status({**st, "boosts": boosts}, tab["mv_status"][m], smask)
        # self-drops/boosts apply once if the move hit anything
        landed = (hit | smask).any()
        me = ids == a
        boosts = add_boost(st["boosts"], me & landed, tab["mv_self_boost"][m])
        boosts = add_boost(boosts, me & landed & (hit & (rng["sec"][a] < tab["mv_sec_chance"][m])).any(), tab["mv_sec_self_boost"][m])
        return {**st, "boosts": boosts}, done

    st, _ = jax.lax.fori_loop(0, 4, act, (state, jnp.zeros(4, bool)))

    # end-of-turn residual damage: burn 1/16, poison 1/8, toxic n/16
    mx = st["stats"][:, 0]
    alive = st["hp"] > 0
    tox = jnp.where((st["status"] == 4) & alive, jnp.minimum(st["tox"] + 1, 15), st["tox"])
    base = jnp.maximum(mx // 16, 1)
    res = jnp.where(st["status"] == 1, base, 0)
    res = jnp.where(st["status"] == 3, jnp.maximum(mx // 8, 1), res)
    res = jnp.where(st["status"] == 4, base * tox, res)
    hp = jnp.where(alive, jnp.maximum(st["hp"] - res, 0), st["hp"])
    return {**st, "hp": hp, "tox": tox}
