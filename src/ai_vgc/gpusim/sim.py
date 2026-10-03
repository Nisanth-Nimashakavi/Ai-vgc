"""Battle state and the turn function.

Each side brings 4 Pokemon: mon ids side*4 + k. `active[a]` (a = side*2 + slot) is the mon id in each of the
four field slots. Supported: damage, stat stages, brn/par/psn/tox, status moves, secondaries, switching,
fainting + forced replacement, weather, terrain, Trick Room, Tailwind, screens.
Not yet: abilities, items, protect, sleep/freeze/confusion, accuracy stages, hazards.
"""
import jax
import jax.numpy as jnp
import numpy as np

from . import effects as E
from . import tables as T
from .damage import damage, modify

LEVEL = 50
SWITCH_PRIO = 10_000_000
FIELD_KEYS = ("weather", "weather_t", "terrain", "terrain_t", "tr_t")
SIDE_KEYS = ("tw_t", "reflect_t", "ls_t", "av_t")  # [2] turns left per side
TABLE_NAMES = ("mv_type mv_cat mv_bp mv_prio mv_target mv_acc sp_types mv_boost mv_boost_self mv_self_boost "
               "mv_status mv_sec_chance mv_sec_status mv_sec_boost mv_sec_self_boost mv_imm mv_ign_def mv_ign_off "
               "mv_powder mv_field mv_halve_grassy mv_protect mv_blockable mv_fakeout mv_selfboost mv_pp").split()


def make_tables(t=None):
    """Move/type tables as device arrays, passed to `step` as one pytree."""
    t = t or T.load()
    chart = np.zeros_like(t.chart, np.int32)  # +1 super effective, -1 resisted, -99 immune
    chart[t.chart == 2] = 1
    chart[t.chart == 0.5] = -1
    chart[t.chart == 0] = -99
    tab = {k: jnp.asarray(getattr(t, k)) for k in TABLE_NAMES}
    # type immunity to each status: brn Fire, par Electric, psn/tox Poison+Steel (index = status code)
    imm = np.zeros((5, len(t.types)), bool)
    for code, typs in {1: ["Fire"], 2: ["Electric"], 3: ["Poison", "Steel"], 4: ["Poison", "Steel"]}.items():
        imm[code, [t.types.index(x) for x in typs]] = True
    ix = t.types.index
    # [weather+1, type] damage modifier, and which types sand damages / snow or sand defence-boosts
    wmod = np.full((5, len(t.types)), 4096, np.int32)
    wmod[1, ix("Fire")], wmod[1, ix("Water")] = 6144, 2048
    wmod[2, ix("Water")], wmod[2, ix("Fire")] = 6144, 2048
    sand_imm = np.zeros(len(t.types), bool)
    sand_imm[[ix("Rock"), ix("Ground"), ix("Steel")]] = True
    tmod = np.full((5, len(t.types)), 4096, np.int32)  # terrain boost to the attacker's move type
    tmod[1, ix("Electric")] = tmod[2, ix("Grass")] = tmod[4, ix("Psychic")] = 5325
    def per(index, values, default, dtype=np.int32):
        arr = np.full(len(index) + 1, default, dtype)
        for k, v in values.items():
            arr[index[k]] = v
        return jnp.asarray(arr)

    ai, ii = t.ability_ix, t.item_ix
    eff = dict(
        ab_weather=per(ai, E.AB_WEATHER, 0), ab_terrain=per(ai, E.AB_TERRAIN, 0),
        ab_intimidate=per(ai, dict.fromkeys(E.AB_INTIMIDATE, True), False, bool),
        it_wext=per(ii, E.IT_WEATHER_EXT, 0), it_spe=per(ii, E.IT_SPE_MOD, 4096), it_dmg=per(ii, E.IT_DMG_MOD, 4096),
        it_leftovers=per(ii, dict.fromkeys(E.IT_LEFTOVERS, True), False, bool),
        it_sash=per(ii, dict.fromkeys(E.IT_SASH, True), False, bool),
        it_sitrus=per(ii, dict.fromkeys(E.IT_SITRUS, True), False, bool),
        it_text=per(ii, dict.fromkeys(E.IT_TERRAIN_EXT, True), False, bool),
        it_clay=per(ii, dict.fromkeys(E.IT_CLAY, True), False, bool))
    return {**tab, **eff, "chart": jnp.asarray(chart), "status_imm": jnp.asarray(imm), "wmod": jnp.asarray(wmod),
            "sand_imm": jnp.asarray(sand_imm), "tmod": jnp.asarray(tmod), "t_grass": ix("Grass"),
            "t_flying": ix("Flying"), "t_dragon": ix("Dragon"), "t_rock": ix("Rock"), "t_ice": ix("Ice")}


def make_state(teams, t=None, status=None):
    """teams: [side0, side1], each a list of 4 dicts(species, moves, evs, nature); the first two of each
    side start active. status: optional [8] status codes."""
    t = t or T.load()
    stats, species, moves, abil, item = [], [], [], [], []
    for side in teams:
        for m in side[:4]:
            abil.append(t.ability_ix.get(T.to_id(m.get("ability", "")), 0))
            item.append(t.item_ix.get(T.to_id(m.get("item", "")), 0))
            sid = t.species_ix[T.to_id(m["species"])]
            species.append(sid)
            stats.append(T.calc_stats(t.base[sid], [m["evs"][k] for k in T.STATS], m["nature"], LEVEL))
            moves.append([t.move_ix[T.to_id(x)] for x in m["moves"]])
    stats = np.array(stats, np.int32)
    z = np.int32(0)
    return dict(pp=t.mv_pp[np.array(moves, np.int32)], stats=stats, hp=stats[:, 0].copy(), species=np.array(species, np.int32),
                moves=np.array(moves, np.int32), boosts=np.zeros((8, 5), np.int32),
                status=np.zeros(8, np.int32) if status is None else np.asarray(status, np.int32),
                tox=np.zeros(8, np.int32), active=np.array([0, 1, 4, 5], np.int32),
                age=np.ones(8, np.int32), stall=np.zeros(8, np.int32),
                ability=np.array(abil, np.int32), item=np.array(item, np.int32),
                weather=z, weather_t=z, terrain=z, terrain_t=z, tr_t=z,
                **{k: np.zeros(2, np.int32) for k in SIDE_KEYS})


def put(arr, idx, vals):
    """arr.at[idx].set(vals) as a one-hot select. jnp scatters compile very slowly under vmap on CPU, and these
    arrays are tiny. idx: scalar or [k] int (unique; out-of-range entries are dropped); vals match arr[idx]."""
    n = arr.shape[0]
    idx = jnp.asarray(idx)
    hit = jnp.arange(n) == (idx[..., None] if idx.ndim else idx)  # [n] or [k, n]
    if idx.ndim == 0:
        return jnp.where(hit.reshape((n,) + (1,) * (arr.ndim - 1)), jnp.asarray(vals, arr.dtype), arr)
    vals = jnp.asarray(vals, arr.dtype)
    sel = hit.T.reshape((n, idx.shape[0]) + (1,) * (arr.ndim - 1))  # [n, k, ...]
    picked = jnp.where(sel, vals[None], 0).sum(axis=1).astype(arr.dtype)
    return jnp.where(hit.any(axis=0).reshape((n,) + (1,) * (arr.ndim - 1)), picked, arr)


def stage(stat, boost):
    """Showdown calculateStat boost step: floor(stat * table[boost]) with table 2+b/2 or 2/(2-b)."""
    return jnp.where(boost >= 0, stat * (2 + boost) // 2, stat * 2 // (2 - boost))


def _add_boost(st, who, delta):
    """Apply a stat-stage change to the active slots in `who` (bool[4])."""
    A = st["active"]
    new = jnp.clip(st["boosts"][A] + delta, -6, 6)
    return {**st, "boosts": put(st["boosts"], A, jnp.where(who[:, None], new, st["boosts"][A]))}


def _chain(m, n):
    """Showdown chainModify: combine two 4096-based modifiers."""
    return (m * n + 2048) >> 12


def _type_exp(tab, mv_type, def_types):
    eff = jnp.where(def_types >= 0, tab["chart"][mv_type, jnp.maximum(def_types, 0)], 0)
    return jnp.where((eff == -99).any(), -99, eff.sum())


def _has(types, t):
    return (types == t).any(axis=-1)


def _speeds(tab, st):
    """Speed of each active slot: stages, then one chained modifier for Tailwind, paralysis and Choice Scarf."""
    A = st["active"]
    spe = stage(st["stats"][A, 5], st["boosts"][A, 4])
    m = jnp.full(4, 4096)
    m = jnp.where(st["tw_t"][jnp.arange(4) // 2] > 0, _chain(m, 8192), m)
    m = jnp.where(st["status"][A] == 2, _chain(m, 2048), m)
    m = _chain(m, tab["it_spe"][st["item"][A]])
    return modify(spe, m)


def _speed_order(tab, st):
    """Slots fastest first (slowest first under Trick Room), ties by slot."""
    spe = _speeds(tab, st)
    spe = jnp.where(st["tr_t"] > 0, 10000 - spe, spe)
    return jnp.argsort(-(spe * 4 - jnp.arange(4)))


def _update_items(tab, st):
    """Showdown's Update event: berries eaten once HP is at or below half (Sitrus heals 1/4)."""
    A = st["active"]
    mx, hp = st["stats"][A, 0], st["hp"][A]
    eat = tab["it_sitrus"][st["item"][A]] & (hp > 0) & (hp * 2 <= mx)
    hp = jnp.where(eat, jnp.minimum(hp + mx // 4, mx), hp)
    return {**st, "hp": put(st["hp"], A, hp), "item": put(st["item"], A, jnp.where(eat, 0, st["item"][A]))}


def _on_switch_in(tab, st, a, gate=True):
    """Ability effects of the Pokemon that just entered slot a (weather/terrain setters, Intimidate)."""
    mon = st["active"][a]
    ab, item = st["ability"][mon], st["item"][mon]
    live = gate & (st["hp"][mon] > 0)
    w = tab["ab_weather"][ab]
    okw = live & (w > 0) & (st["weather"] != w)
    t = tab["ab_terrain"][ab]
    okt = live & (t > 0) & (st["terrain"] != t)
    st = {**st, "weather": jnp.where(okw, w, st["weather"]),
          "weather_t": jnp.where(okw, jnp.where(tab["it_wext"][item] == w, 8, 5), st["weather_t"]),
          "terrain": jnp.where(okt, t, st["terrain"]),
          "terrain_t": jnp.where(okt, jnp.where(tab["it_text"][item], 8, 5), st["terrain_t"])}
    ids = jnp.arange(4)
    foes = (ids // 2 != a // 2) & (st["hp"][st["active"]] > 0)
    drop = jnp.array([-1, 0, 0, 0, 0])
    return _add_boost(st, foes & live & tab["ab_intimidate"][ab], drop)


def start(tab, state):
    """Battle start: the four leads switch in together, fastest first (ties by slot)."""
    st = jax.tree.map(jnp.asarray, state)
    order = _speed_order(tab, st)
    return jax.lax.fori_loop(0, 4, lambda i, s: _on_switch_in(tab, s, order[i]), st)


def _switch_in(st, a, mon):
    """Put mon into slot a: stages and the toxic counter reset."""
    return {**st, "active": put(st["active"], a, mon), "boosts": put(st["boosts"], mon, 0),
            "tox": put(st["tox"], mon, 0), "age": put(st["age"], mon, 0), "stall": put(st["stall"], mon, 0)}


def step(tab, state, actions, rng):
    """One turn. actions[a] = (kind, x, y): kind 0 = move slot x aimed at foe slot y; kind 1 = switch to
    teammate x (0..3 within the side). rng: rolls[a, d] 0..15, crits[a, d] bool, acc[a, d] 0..99 (hit if below
    accuracy), stall[a] bool (consecutive Protect succeeds), sec[a, d] 0..99 (secondary fires if below its chance), skip[a] bool (fully paralysed),
    tiebreak[a] in [0,1) (higher goes first on equal speed). Fainted slots are refilled by `replace`."""
    state = jax.tree.map(jnp.asarray, state)
    ids = jnp.arange(4)
    is_switch = actions[:, 0] == 1
    mv = state["moves"][state["active"], jnp.clip(actions[:, 1], 0, 3)]  # move ids (garbage for switches)
    prio = jnp.where(is_switch, SWITCH_PRIO, tab["mv_prio"][mv] * 1000000) + jnp.floor(rng["tiebreak"] * 4)

    def set_status(st, code, who):
        """Apply status `code` to slots `who` (bool[4]) where legal."""
        A = st["active"]
        types = tab["sp_types"][st["species"][A]]
        blocked = (tab["status_imm"][code][jnp.maximum(types, 0)] & (types >= 0)).any(axis=1)
        misty = (st["terrain"] == 3) & ~_has(types, tab["t_flying"])
        ok = who & (st["status"][A] == 0) & ~blocked & ~misty & (code > 0)
        mon = jnp.where(ok, A, 8)  # out-of-range index drops the update
        return {**st, "status": put(st["status"], mon, jnp.full(4, code)), "tox": put(st["tox"], mon, jnp.zeros(4, jnp.int32))}

    add_boost = _add_boost

    def field_move(st, f, a):
        """Set weather/terrain/Trick Room/side conditions (f = 0 for none). Fails if already active (Trick Room
        toggles)."""
        side = a // 2
        w = (f >= 1) & (f <= 4)
        mon = st["active"][a]
        item = st["item"][mon]
        okw = w & (st["weather"] != f)
        tr_on = st["tr_t"] > 0
        tg = f - 9
        okt = (f >= 10) & (st["terrain"] != tg)
        side_t = {6: "tw_t", 7: "reflect_t", 8: "ls_t", 9: "av_t"}
        wdur = jnp.where(tab["it_wext"][item] == f, 8, 5)
        tdur = jnp.where(tab["it_text"][item], 8, 5)
        st = {**st, "weather": jnp.where(okw, f, st["weather"]), "weather_t": jnp.where(okw, wdur, st["weather_t"]),
              "terrain": jnp.where(okt, tg, st["terrain"]), "terrain_t": jnp.where(okt, tdur, st["terrain_t"]),
              "tr_t": jnp.where(f == 5, jnp.where(tr_on, 0, 5), st["tr_t"])}
        for code, key in side_t.items():
            dur = 4 if code == 6 else jnp.where(tab["it_clay"][item], 8, 5)
            ok = (f == code) & (st[key][side] == 0)
            if code == 9:
                ok = ok & (st["weather"] == 4)
            st = {**st, key: put(st[key], side, jnp.where(ok, dur, st[key][side]))}
        return st

    def act(i, carry):
        st, done, prot, flinch = carry
        # Showdown re-sorts the remaining actions after speed changes, so pick the next actor now
        spe = _speeds(tab, st)
        spe = jnp.where(st["tr_t"] > 0, 10000 - spe, spe)
        a = jnp.argmax(jnp.where(done, -(2 ** 30), prio + spe * 4))
        done = put(done, a, True)
        side = a // 2

        def do_switch(st):
            return _on_switch_in(tab, _switch_in(st, a, side * 4 + actions[a, 1]), a), prot, flinch

        def do_move(st):
            A = st["active"]
            m = mv[a]
            ma = A[a]
            cat, tgt = tab["mv_cat"][m], tab["mv_target"][m]
            foe = ids // 2 != side
            alive = st["hp"][A] > 0
            types = tab["sp_types"][st["species"][A]]
            chosen = (1 - side) * 2 + actions[a, 2]
            other = (1 - side) * 2 + (1 - actions[a, 2])
            single = jnp.where(alive[chosen], chosen, other)
            aim = jnp.where(tab["mv_boost_self"][m], a, single)
            mask = jnp.where(tgt == 0, ids == aim, jnp.where(tgt == 1, foe, ids != a))
            mask = jnp.where(tgt == 3, ids == a, mask)
            mask = jnp.where(tgt == 4, False, mask) & alive
            spread = (tgt > 0) & (tgt < 3) & (mask.sum() > 1)
            is_prot = tab["mv_protect"][m]
            will_act = (~done & alive).any()  # Protect fails when nothing acts after it
            prot_ok = ((st["stall"][ma] == 0) | rng["stall"][a]) & will_act  # consecutive use succeeds 1/3^n of the time
            fo_ok = ~tab["mv_fakeout"][m] | (st["age"][ma] <= 1)  # Fake Out: first turn out only
            slot = actions[a, 1]
            started = alive[a] & ~((st["status"][ma] == 2) & rng["skip"][a]) & ~flinch[a]  # past BeforeMove
            has_pp = st["pp"][ma, slot] > 0
            st = {**st, "pp": put(st["pp"], ma, jnp.where(started & has_pp, st["pp"][ma] - (jnp.arange(4) == slot), st["pp"][ma]))}
            can = started & has_pp & fo_ok & (~is_prot | prot_ok)
            prot2 = put(prot, a, prot[a] | (can & is_prot))
            st = {**st, "stall": put(st["stall"], ma, jnp.where(can & is_prot, st["stall"][ma] + 1, st["stall"][ma]))}
            blocked = prot[ids] & tab["mv_blockable"][m] & (ids != a)  # protected targets shrug the move off
            mtype = tab["mv_type"][m]
            grounded = ~_has(types, tab["t_flying"])
            texp = jax.vmap(lambda dt: _type_exp(tab, mtype, dt))(types)
            powder_block = tab["mv_powder"][m] & _has(types, tab["t_grass"])
            psy_block = (st["terrain"] == 4) & grounded & (tab["mv_prio"][m] > 0) & foe
            immune = (((texp == -99) & tab["mv_imm"][m]) | powder_block | psy_block) & (tgt != 3)
            hit_roll = rng["acc"][a] < tab["mv_acc"][m]
            mask = mask & ~immune & ~blocked & hit_roll & can
            # damage
            phys = cat == 0
            crit = rng["crits"][a]
            atk_b = st["boosts"][ma, jnp.where(phys, 0, 2)]
            atk_b = jnp.where(tab["mv_ign_off"][m], 0, jnp.where(crit, jnp.maximum(atk_b, 0), atk_b))
            dfn_b = st["boosts"][A, jnp.where(phys, 1, 3)]
            dfn_b = jnp.where(tab["mv_ign_def"][m], 0, jnp.where(crit, jnp.minimum(dfn_b, 0), dfn_b))
            atk = stage(st["stats"][ma, jnp.where(phys, 1, 3)], atk_b)
            dfn = stage(st["stats"][A, jnp.where(phys, 2, 4)], dfn_b)
            w = st["weather"]
            boost_def = ((w == 4) & phys & _has(types, tab["t_ice"])) | ((w == 3) & ~phys & _has(types, tab["t_rock"]))
            dfn = jnp.where(boost_def, modify(dfn, 6144), dfn)
            stab = jnp.where(_has(types[a], mtype), 6144, 4096)
            # terrain changes base power (grounded attacker boosts its type; grounded defender in misty/grassy)
            g_att = grounded[a]
            bp = tab["mv_bp"][m]
            bp = jnp.where(g_att, modify(bp, tab["tmod"][st["terrain"], mtype]), bp)
            bp = jnp.where((st["terrain"] == 3) & (mtype == tab["t_dragon"]) & grounded, modify(bp, 2048), bp)
            bp = jnp.where((st["terrain"] == 2) & tab["mv_halve_grassy"][m] & grounded, modify(bp, 2048), bp)
            tside = ids // 2
            screen = jnp.where(phys, (st["reflect_t"][tside] > 0), (st["ls_t"][tside] > 0)) | (st["av_t"][tside] > 0)
            final = jnp.where(screen & ~crit & (ids != a), 2732, 4096)
            final = _chain(final, tab["it_dmg"][st["item"][ma]])  # Life Orb
            dmg = damage(LEVEL, bp, atk, dfn, rng["rolls"][a], crit, stab, texp,
                         (st["status"][ma] == 1) & phys, spread, tab["wmod"][w, mtype], final)
            hit = mask & (cat < 2)
            hp0 = st["hp"][A]
            # Focus Sash: a hit that would KO a full-HP holder leaves it at 1 HP and consumes the item
            sash = hit & tab["it_sash"][st["item"][A]] & (hp0 == st["stats"][A, 0]) & (dmg >= hp0)
            dmg = jnp.where(sash, hp0 - 1, dmg)
            hp = put(st["hp"], A, jnp.where(hit, jnp.maximum(hp0 - dmg, 0), hp0))
            st = {**st, "hp": hp, "item": put(st["item"], A, jnp.where(sash, 0, st["item"][A]))}
            hit_alive = hit & (hp[A] > 0)
            # secondary effects on surviving targets
            proc = hit_alive & (rng["sec"][a] < tab["mv_sec_chance"][m])
            flinch2 = flinch | (proc & tab["mv_fakeout"][m] & ~done[ids])
            st = set_status(st, tab["mv_sec_status"][m], proc)
            st = add_boost(st, proc, tab["mv_sec_boost"][m])
            # status moves: boosts on the targets, then inflicted status
            smask = mask & (cat == 2)
            st = add_boost(st, smask, tab["mv_boost"][m])
            st = set_status(st, tab["mv_status"][m], smask)
            # self-drops/boosts apply once if the move hit anything
            landed = (hit | smask).any()
            ongoing = winner(st) == -1
            me = ids == a
            st = add_boost(st, me & landed, tab["mv_self_boost"][m])
            st = add_boost(st, me & (hit & (rng["sec"][a] < tab["mv_sec_chance"][m])).any(), tab["mv_sec_self_boost"][m])
            st = add_boost(st, me & landed & ongoing, tab["mv_selfboost"][m])  # skipped when the move ends the battle
            # Life Orb recoil: any attack that hit something
            lo = tab["it_dmg"][st["item"][ma]] != 4096
            recoil = jnp.where(lo & hit.any() & (cat < 2), jnp.maximum(st["stats"][ma, 0] // 10, 1), 0)
            st = {**st, "hp": put(st["hp"], ma, jnp.maximum(st["hp"][ma] - recoil, 0))}
            # field-wide moves
            f = tab["mv_field"][m]
            return field_move(st, jnp.where(can & (tgt == 4), f, 0), a), prot2, flinch2

        over = winner(st) != -1  # the battle ends the moment a side is wiped out
        sw = is_switch[a] & (st["hp"][st["active"][a]] > 0)
        switched = do_switch(st)
        moved = do_move(st)
        pick = lambda cond, x, y: jax.tree.map(lambda p, q: jnp.where(cond, p, q), x, y)
        st, prot, flinch = pick(over, (st, prot, flinch), pick(sw, switched, pick(is_switch[a], (st, prot, flinch), moved)))
        st = _update_items(tab, st)
        return st, done, prot, flinch

    st, _, prot_final, _ = jax.lax.fori_loop(0, 4, act, (state, jnp.zeros(4, bool), jnp.zeros(4, bool), jnp.zeros(4, bool)))
    # the stall counter only survives a turn in which the Pokemon actually used Protect again
    st = {**st, "age": put(st["age"], st["active"], st["age"][st["active"]] + 1),
          "stall": put(st["stall"], st["active"], jnp.where(prot_final, st["stall"][st["active"]], 0))}
    res = _residual(tab, st, rng["tiebreak"])
    return jax.tree.map(lambda p, q: jnp.where(winner(st) != -1, p, q), st, res)


def _residual(tab, st, tie):
    """End of turn. Showdown runs each residual effect (weather, terrain heal, status) over the active slots in
    speed order and ends the battle the moment a side is wiped out. Timers tick down afterwards; a timer
    that reaches 0 ends its effect (weather/terrain are checked before they act)."""
    A = st["active"]
    spe = _speeds(tab, st)
    spe = jnp.where(st["tr_t"] > 0, 10000 - spe, spe)
    order = jnp.argsort(-(spe * 4 + jnp.floor(tie * 4)))
    wt = jnp.maximum(st["weather_t"] - 1, 0)
    w = jnp.where((st["weather_t"] > 0) & (wt == 0), 0, st["weather"])

    def phase(kind):
        def body(i, s):
            # a win is only checked between residual effects, so every Pokemon takes this effect's damage
            # weather/terrain heal are one handler over everyone; Leftovers and statuses are per Pokemon
            over0 = over_at_start[0] if kind < 2 else (winner(s) != -1)
            a = order[i]
            mon = s["active"][a]
            mx = s["stats"][mon, 0]
            hp = s["hp"][mon]
            types = tab["sp_types"][s["species"][mon]]
            live = (hp > 0) & ~over0
            if kind == 0:  # sandstorm chip
                sand = (w == 3) & ~(tab["sand_imm"][jnp.maximum(types, 0)] & (types >= 0)).any()
                new = jnp.where(sand, jnp.maximum(hp - jnp.maximum(mx // 16, 1), 0), hp)
            elif kind == 1:  # grassy terrain heal (runs before the terrain's own end handler)
                heal = (s["terrain"] == 2) & ~_has(types, tab["t_flying"])
                new = jnp.where(heal, jnp.minimum(hp + jnp.maximum(mx // 16, 1), mx), hp)
            elif kind == 2:  # Leftovers (order 5, after the terrain heal)
                new = jnp.where(tab["it_leftovers"][s["item"][mon]], jnp.minimum(hp + jnp.maximum(mx // 16, 1), mx), hp)
            else:  # kind 3: poison/toxic (residual order 9), kind 4: burn (order 10)
                status = s["status"][mon]
                tox = jnp.where((status == 4) & (kind == 3), jnp.minimum(s["tox"][mon] + 1, 15), s["tox"][mon])
                base = jnp.maximum(mx // 16, 1)
                res = jnp.where((status == 1) & (kind == 4), base, 0)
                res = jnp.where((status == 3) & (kind == 3), jnp.maximum(mx // 8, 1), res)
                res = jnp.where((status == 4) & (kind == 3), base * tox, res)
                new = jnp.maximum(hp - res, 0)
                s = {**s, "tox": put(s["tox"], mon, jnp.where(live, tox, s["tox"][mon]))}
            return {**s, "hp": put(s["hp"], mon, jnp.where(live, new, hp))}
        return body

    for kind in range(5):
        over_at_start = [winner(st) != -1]
        st = _update_items(tab, jax.lax.fori_loop(0, 4, phase(kind), st))
    over = winner(st) != -1  # a battle that ended mid-residual never ticks terrain/side/Trick Room timers
    tick = lambda x: jnp.where(over, x, jnp.maximum(x - 1, 0))
    tt = tick(st["terrain_t"])
    terr = jnp.where((st["terrain_t"] > 0) & (tt == 0) & ~over, 0, st["terrain"])
    return {**st, "weather": w, "weather_t": wt, "terrain": terr, "terrain_t": tt, "tr_t": tick(st["tr_t"]),
            "tw_t": tick(st["tw_t"]), "reflect_t": tick(st["reflect_t"]), "ls_t": tick(st["ls_t"]),
            "av_t": tick(st["av_t"])}


def replace(tab, state, choice):
    """Refill fainted slots after a turn. choice[a] = teammate (0..3) to send into slot a; ignored unless
    that slot's Pokemon fainted and the choice is a living bench Pokemon; -1 means no choice (battle over)."""
    st = jax.tree.map(jnp.asarray, state)
    placed = []
    for a in range(4):
        mon = (a // 2) * 4 + jnp.maximum(choice[a], 0)
        ok = (choice[a] >= 0) & (st["hp"][st["active"][a]] == 0) & (st["hp"][mon] > 0) & ~(st["active"] == mon).any()
        sw = _switch_in(st, a, mon)
        sw = {**sw, "age": put(sw["age"], mon, 1)}  # activeTurns ticks to 1 before its first move
        st = jax.tree.map(lambda p, q: jnp.where(ok, p, q), sw, st)
        placed.append(ok)
    placed = jnp.stack(placed)
    # replacements' abilities activate together in speed order (reversed under Trick Room)
    order = _speed_order(tab, st)
    st = jax.lax.fori_loop(0, 4, lambda i, s: _on_switch_in(tab, s, order[i], placed[order[i]]), st)
    return _update_items(tab, st)


def winner(state):
    """-1 while both sides have Pokemon left, else the winning side (-2 if both are out)."""
    alive = (jnp.asarray(state["hp"]).reshape(2, 4) > 0).any(axis=1)
    return jnp.where(alive.all(), -1, jnp.where(alive[0], 0, jnp.where(alive[1], 1, -2)))
