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
               "mv_powder mv_field mv_halve_grassy mv_protect mv_blockable mv_fakeout mv_selfboost mv_pp mv_recoil "
               "mv_has_sec mv_toxic mv_acc_true mv_ignore_ability mv_crit mv_willcrit mv_tracks").split()


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
        it_wext=per(ii, E.IT_WEATHER_EXT, 0), it_spe=per(ii, E.IT_SPE_MOD, 4096),
        it_leftovers=per(ii, dict.fromkeys(E.IT_LEFTOVERS, True), False, bool),
        it_sash=per(ii, dict.fromkeys(E.IT_SASH, True), False, bool),
        it_sitrus=per(ii, dict.fromkeys(E.IT_SITRUS, True), False, bool),
        it_text=per(ii, dict.fromkeys(E.IT_TERRAIN_EXT, True), False, bool),
        it_clay=per(ii, dict.fromkeys(E.IT_CLAY, True), False, bool))
    # rule tables: [n+1, 2] priority / cond1 / cond2 / multiplier for each (kind, event)
    cix = {n: i for i, n in enumerate(E.COND_NAMES)}
    n_ab, n_it = len(ai) + 1, len(ii) + 1
    rules = {}
    for kind, evs in E.RULES.items():
        index, n = (ai, n_ab) if kind in ("ab", "dab", "aab", "any") else (ii, n_it)
        for ev, byid in evs.items():
            pr = np.zeros((n, 2), np.int32)
            c1 = np.zeros((n, 2), np.int32)
            c2 = np.zeros((n, 2), np.int32)
            mu = np.full((n, 2), 4096, np.int32)
            for name, lst in byid.items():
                for r, (prio, (a_, b_), mult) in enumerate(lst):
                    pr[index[name], r], c1[index[name], r], c2[index[name], r], mu[index[name], r] = prio, cix[a_], cix[b_], mult
            rules[f"{kind}_{ev}"] = dict(prio=jnp.asarray(pr), c1=jnp.asarray(c1), c2=jnp.asarray(c2), mult=jnp.asarray(mu))
    rt_kind = np.zeros(n_ab, np.int32)
    rt_type = np.zeros(n_ab, np.int32)
    for name, (kind_, typ) in E.RETYPE.items():
        rt_kind[ai[name]], rt_type[ai[name]] = kind_, t.types.index(typ)
    flagn = lambda names: jnp.asarray(np.isin(np.arange(n_ab), [ai[x] for x in names]))
    berry_t = np.full(n_it, -1, np.int32)
    for name, typ in E.RESIST_BERRIES.items():
        berry_t[ii[name]] = t.types.index(typ)
    pika = np.array([sp["baseSpecies"] == "Pikachu" if sp.get("baseSpecies") else sp["name"] == "Pikachu"
                     for sp in t.raw["species"]], bool)
    def mat(table_by_ab, width=5):
        arr = np.zeros((n_ab, width), bool)
        for name, codes in table_by_ab.items():
            for c_ in codes:
                arr[ai[name], c_] = True
        return jnp.asarray(arr)

    def frac(d):
        arr = np.zeros(n_ab, np.int32)
        for name, n_ in d.items():
            arr[ai[name]] = n_
        return jnp.asarray(arr)

    FLAGS = ["", "wind", "bullet", "sound", "status", "ally", "notse"]
    abs_type = np.full(n_ab, -1, np.int32)
    abs_flag = np.zeros(n_ab, np.int32)
    abs_kind = np.zeros(n_ab, np.int32)  # 0 none, 1 heal, 2 boost, 3 flash fire, 4 plain immunity
    abs_stat = np.zeros(n_ab, np.int32)
    abs_n = np.zeros(n_ab, np.int32)
    for name, (match, eff_) in E.ABSORB.items():
        i_ = ai[name]
        if match in FLAGS:
            abs_flag[i_] = FLAGS.index(match)
        else:
            abs_type[i_] = t.types.index(match)
        abs_kind[i_] = {"heal": 1, "boost": 2, "flashfire": 3, "immune": 4}[eff_[0]]
        if eff_[0] == "boost":
            abs_stat[i_], abs_n[i_] = ["atk", "def", "spa", "spd", "spe"].index(eff_[1]), eff_[2]
    redirect = np.full(n_ab, -1, np.int32)
    for name, typ in E.REDIRECT.items():
        redirect[ai[name]] = t.types.index(typ)
    sp_when = np.zeros(n_ab, np.int32)
    sp_mult = np.full(n_ab, 4096, np.int32)
    for name, (when, mult) in E.SPEED.items():
        sp_when[ai[name]], sp_mult[ai[name]] = E.SPEED_WHEN.index(when), mult
    return {**tab, **eff, "rules": rules, "ab_sp_when": jnp.asarray(sp_when), "ab_sp_mult": jnp.asarray(sp_mult),
            "ab_quickfeet": flagn(["quickfeet"]), "ab_noredirect": flagn(E.IGNORE_REDIRECT), "ab_critplus": per(ai, {"superluck": 1}, 0), "ab_merciless": flagn(["merciless"]),
            "it_critplus": per(ii, E.CRIT_PLUS if False else {"scopelens": 1}, 0),
            "it_leek": per(ii, dict.fromkeys(E.CRIT_LEEK, True), False, bool),
            "sp_leek": jnp.asarray(np.array([sp["id"] in ("farfetchd", "farfetchdgalar", "sirfetchd") for sp in t.raw["species"]], bool)), "ab_speedboost": flagn(["speedboost"]),
            "abs_type": jnp.asarray(abs_type), "abs_flag": jnp.asarray(abs_flag), "abs_kind": jnp.asarray(abs_kind),
            "abs_stat": jnp.asarray(abs_stat), "abs_n": jnp.asarray(abs_n), "redirect": jnp.asarray(redirect),
            "ab_ungrounded": flagn(E.UNGROUNDED), "ab_windrider": flagn(["windrider"]), "ab_breakable": jnp.asarray(t.ab_breakable), "ab_moldbreaker": flagn(E.MOLD_BREAKER),
            "ab_status_block": mat(E.STATUS_BLOCK), "ab_sun_block": flagn(E.SUN_BLOCK),
            "ab_ally_status_block": mat(E.ALLY_STATUS_BLOCK), "ab_sand_imm": flagn(E.SAND_IMMUNE),
            "ab_sun_dmg": frac(E.SUN_DAMAGE), "ab_rain_heal": frac(E.RAIN_HEAL), "ab_snow_heal": frac(E.SNOW_HEAL),
            "ab_magicguard": flagn(E.MAGIC_GUARD), "ab_heatproof": flagn(E.HEATPROOF),
            "ab_powder_imm": flagn(E.POWDER_IMMUNE), "ab_noguard": flagn(["noguard"]), "ab_victorystar": flagn(["victorystar"]), "ab_wonderskin": flagn(["wonderskin"]), "rt_kind": jnp.asarray(rt_kind), "rt_type": jnp.asarray(rt_type),
            "ab_adapt": flagn(["adaptability"]), "ab_darkaura": flagn(["darkaura"]), "ab_fairyaura": flagn(["fairyaura"]),
            "ab_aurabreak": flagn(["aurabreak"]), "ab_friendguard": flagn(["friendguard"]),
            "ab_plusminus": flagn(["plus", "minus"]), "ab_sheerforce": flagn(["sheerforce"]), "ab_guts": flagn(["guts"]),
            "it_berry_type": jnp.asarray(berry_t), "sp_pikachu": jnp.asarray(pika),
            "mv_f": {k: jnp.asarray(v) for k, v in t.mv_f.items()},
            "t_normal": ix("Normal"), "t_poison": ix("Poison"), "t_dark": ix("Dark"), "t_fairy": ix("Fairy"), "t_steel": ix("Steel"),
            "t_ground": ix("Ground"), "t_rock2": ix("Rock"),
            "it_lifeorb": jnp.asarray(np.isin(np.arange(n_it), [ii[x] for x in E.IT_LIFEORB])),
            "chart": jnp.asarray(chart), "status_imm": jnp.asarray(imm), "wmod": jnp.asarray(wmod),
            "sand_imm": jnp.asarray(sand_imm), "tmod": jnp.asarray(tmod), "t_grass": ix("Grass"),
            "t_flying": ix("Flying"), "t_dragon": ix("Dragon"), "t_rock": ix("Rock"), "t_ice": ix("Ice")}


def make_state(teams, t=None, status=None, genders=None):
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
                age=np.ones(8, np.int32), stall=np.zeros(8, np.int32), fresh=np.zeros(8, np.int32), lost=np.zeros(8, np.int32), ff=np.zeros(8, np.int32),
                gender=np.array([{'M': 1, 'F': 2}.get(g, 0) for g in (genders or ['N'] * 8)], np.int32),
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


def _grounded(tab, st, mons):
    """Not Flying and not levitating. mons: indices into the 8 Pokemon."""
    types = tab["sp_types"][st["species"][mons]]
    return ~_has(types, tab["t_flying"]) & ~tab["ab_ungrounded"][st["ability"][mons]]


def _has(types, t):
    return (types == t).any(axis=-1)


def _speeds(tab, st):
    """Speed of each active slot: stages, then one chained modifier for Tailwind, paralysis, abilities and items."""
    A = st["active"]
    ab = st["ability"][A]
    spe = stage(st["stats"][A, 5], st["boosts"][A, 4])
    w, tr = st["weather"], st["terrain"]
    when = tab["ab_sp_when"][ab]
    cond = jnp.select([when == 1, when == 2, when == 3, when == 4, when == 5, when == 6, when == 7],
                      [w == 1, w == 2, w == 3, w == 4, tr == 1, st["status"][A] > 0, st["lost"][A] > 0], when == 0)
    m = jnp.full(4, 4096)
    m = jnp.where(st["tw_t"][jnp.arange(4) // 2] > 0, _chain(m, 8192), m)
    m = jnp.where(cond & (when > 0), _chain(m, tab["ab_sp_mult"][ab]), m)
    m = _chain(m, tab["it_spe"][st["item"][A]])
    spe = modify(spe, m)
    # paralysis comes after every other modifier has been applied (Quick Feet ignores it)
    para = (st["status"][A] == 2) & ~tab["ab_quickfeet"][ab]
    spe = jnp.where(para, spe * 50 // 100, spe)
    return jnp.minimum(spe, 10000)


def action_speeds(tab, st):
    """getActionSpeed of each active slot as the Champions mod reports it (negated under Trick Room): what turn order is sorted by."""
    st = jax.tree.map(jnp.asarray, st)
    spe = _speeds(tab, st)
    return jnp.where(st["tr_t"] > 0, -spe, spe)


def _speed_order(tab, st):
    """Slots fastest first (slowest first under Trick Room), ties by slot."""
    spe = _speeds(tab, st)
    spe = jnp.where(st["tr_t"] > 0, -spe, spe)
    return jnp.argsort(-(spe * 4 - jnp.arange(4)))


def _update_items(tab, st):
    """Showdown's Update event: berries eaten once HP is at or below half (Sitrus heals 1/4)."""
    A = st["active"]
    mx, hp = st["stats"][A, 0], st["hp"][A]
    eat = tab["it_sitrus"][st["item"][A]] & (hp > 0) & (hp * 2 <= mx)
    hp = jnp.where(eat, jnp.minimum(hp + mx // 4, mx), hp)
    return {**st, "hp": put(st["hp"], A, hp), "item": put(st["item"], A, jnp.where(eat, 0, st["item"][A])),
            "lost": put(st["lost"], A, jnp.where(eat, 1, st["lost"][A]))}


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
    st = _add_boost(st, foes & live & tab["ab_intimidate"][ab], drop)
    wr = (ids == a) & live & tab["ab_windrider"][ab] & (st["tw_t"][a // 2] > 0)  # Wind Rider entering under Tailwind
    return _add_boost(st, wr, jnp.array([1, 0, 0, 0, 0]))


def start(tab, state):
    """Battle start: the four leads switch in together, fastest first (ties by slot)."""
    st = jax.tree.map(jnp.asarray, state)
    order = _speed_order(tab, st)
    return jax.lax.fori_loop(0, 4, lambda i, s: _on_switch_in(tab, s, order[i]), st)


def _switch_in(st, a, mon):
    """Put mon into slot a: stages and the toxic counter reset."""
    return {**st, "active": put(st["active"], a, mon), "boosts": put(st["boosts"], mon, 0),
            "tox": put(st["tox"], mon, 0), "age": put(st["age"], mon, 0), "stall": put(st["stall"], mon, 0),
            "fresh": put(st["fresh"], mon, 1), "ff": put(st["ff"], mon, 0), "lost": put(st["lost"], mon, 0)}


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
        misty = (st["terrain"] == 3) & _grounded(tab, st, A)
        ab_t = st["ability"][A]
        ally = A[ids ^ 1]
        ability_block = (tab["ab_status_block"][ab_t, code] | (tab["ab_sun_block"][ab_t] & (st["weather"] == 1))
                         | (tab["ab_ally_status_block"][st["ability"][ally], code] & (st["hp"][ally] > 0)))
        ok = who & (st["status"][A] == 0) & ~blocked & ~misty & ~ability_block & (code > 0)
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
            if code == 6:  # Wind Rider holders on the side gain +1 Atk when Tailwind starts
                holders = ok & (ids // 2 == side) & tab["ab_windrider"][st["ability"][st["active"]]] & (st["hp"][st["active"]] > 0)
                st = _add_boost(st, holders, jnp.array([1, 0, 0, 0, 0]))
        return st

    def act(i, carry):
        st, done, prot, flinch = carry
        # Showdown re-sorts the remaining actions after speed changes, so pick the next actor now
        spe = _speeds(tab, st)
        spe = jnp.where(st["tr_t"] > 0, -spe, spe)
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
            # Lightning Rod / Storm Drain pull single-target moves of their type (first holder by speed)
            ab_all0 = st["ability"][A]
            mt0 = tab["mv_type"][m]
            rk0 = tab["rt_kind"][st["ability"][ma]]
            mt_ = jnp.where(((rk0 == 1) & (mt0 == tab["t_normal"])) | (rk0 == 2) | ((rk0 == 3) & tab["mv_f"]["sound"][m]),
                            tab["rt_type"][st["ability"][ma]], mt0)
            brk0 = tab["ab_moldbreaker"][st["ability"][ma]] | tab["mv_ignore_ability"][m]
            ab_vis = jnp.where(brk0 & tab["ab_breakable"][ab_all0], 0, ab_all0)  # a mold-breaker ignores breakable abilities
            holders = alive & (ids != a) & (tab["redirect"][ab_vis] == mt_) & (tgt == 0) & ~tab["mv_boost_self"][m] & ~tab["mv_tracks"][m] & ~tab["ab_noredirect"][st["ability"][ma]]
            spe_o = _speeds(tab, st)
            spe_o = jnp.where(st["tr_t"] > 0, -spe_o, spe_o)
            best = jnp.argmax(jnp.where(holders, spe_o * 4 - ids, -(2 ** 30)))
            single = jnp.where(holders.any(), best, single)
            aim = jnp.where(tab["mv_boost_self"][m], a, single)
            mask = jnp.where(tgt == 0, ids == aim, jnp.where(tgt == 1, foe, ids != a))
            mask = jnp.where(tgt == 3, ids == a, mask)
            mask = jnp.where(tgt == 4, False, mask) & alive
            spread = (tgt > 0) & (tgt < 3) & (mask.sum() > 1)
            is_prot = tab["mv_protect"][m]
            will_act = (~done & alive).any()  # Protect fails when nothing acts after it
            n = jnp.clip(st["stall"][ma], 0, 6)  # the n-th consecutive use succeeds with chance 1/3^n
            stall_roll = jnp.stack([jnp.array(True)] + [rng["chance"][f"1_{3 ** k}"] for k in range(1, 7)])[n]
            prot_ok = stall_roll & will_act
            fo_ok = ~tab["mv_fakeout"][m] | (st["age"][ma] <= 1)  # Fake Out: first turn out only
            slot = actions[a, 1]
            started = alive[a] & ~((st["status"][ma] == 2) & rng["chance"]["1_8"]) & ~flinch[a]  # past BeforeMove
            has_pp = st["pp"][ma, slot] > 0
            st = {**st, "pp": put(st["pp"], ma, jnp.where(started & has_pp, st["pp"][ma] - (jnp.arange(4) == slot), st["pp"][ma]))}
            can = started & has_pp & fo_ok & (~is_prot | prot_ok)
            prot2 = put(prot, a, prot[a] | (can & is_prot))
            st = {**st, "stall": put(st["stall"], ma, jnp.where(can & is_prot, st["stall"][ma] + 1, st["stall"][ma]))}
            blocked = prot[ids] & tab["mv_blockable"][m] & (ids != a)  # protected targets shrug the move off
            ab_a, it_a = st["ability"][ma], st["item"][ma]
            # ModifyType: -ate abilities, Normalize, Liquid Voice
            mtype0 = tab["mv_type"][m]
            rk = tab["rt_kind"][ab_a]
            changed = ((rk == 1) & (mtype0 == tab["t_normal"])) | (rk == 2) | ((rk == 3) & tab["mv_f"]["sound"][m])
            mtype = jnp.where(changed, tab["rt_type"][ab_a], mtype0)
            retyped = changed & (rk != 3)
            grounded = _grounded(tab, st, A)
            texp = jax.vmap(lambda dt: _type_exp(tab, mtype, dt))(types)
            powder_block = tab["mv_powder"][m] & (_has(types, tab["t_grass"]) | tab["ab_powder_imm"][st["ability"][A]])
            psy_block = (st["terrain"] == 4) & grounded & (tab["mv_prio"][m] > 0) & foe
            immune = (((texp == -99) & tab["mv_imm"][m]) | powder_block) & (tgt != 3)
            mask = mask & ~blocked & ~(psy_block & (tgt != 3)) & can  # Protect and Psychic Terrain act before abilities
            ab_a0 = st["ability"][ma]
            breaker = (tab["ab_moldbreaker"][ab_a0] | tab["mv_ignore_ability"][m])
            ab_d = jnp.where(breaker & tab["ab_breakable"][st["ability"][A]], 0, st["ability"][A])
            # TryHit absorption (before accuracy): Water Absorb, Lightning Rod, Levitate, Wonder Guard, ...
            kind_d = tab["abs_kind"][ab_d]
            flag_d = tab["abs_flag"][ab_d]
            m_flag = jnp.select([flag_d == 1, flag_d == 2, flag_d == 3, flag_d == 4, flag_d == 5, flag_d == 6],
                                [tab["mv_f"]["wind"][m], tab["mv_f"]["bullet"][m], tab["mv_f"]["sound"][m], cat == 2,
                                 (ids // 2 == side) & (cat < 2), (cat < 2) & (texp <= 0)], False)
            absorbed = mask & (ids != a) & (kind_d > 0) & ((tab["abs_type"][ab_d] == mtype) | m_flag) & (tgt != 3)
            heal_amt = jnp.maximum(st["stats"][A, 0] // 4, 1)
            full = st["hp"][A] >= st["stats"][A, 0]
            hpn = jnp.where(absorbed & (kind_d == 1) & ~full, jnp.minimum(st["hp"][A] + heal_amt, st["stats"][A, 0]), st["hp"][A])
            st = {**st, "hp": put(st["hp"], A, hpn)}
            bdelta = jnp.where(jnp.arange(5)[None, :] == tab["abs_stat"][ab_d][:, None], tab["abs_n"][ab_d][:, None], 0)
            st = add_boost(st, absorbed & (kind_d == 2), bdelta)
            st = {**st, "ff": put(st["ff"], A, jnp.where(absorbed & (kind_d == 3), 1, st["ff"][A]))}
            mask = mask & ~absorbed & ~immune  # TryHit (absorb) comes before the type-immunity check
            # damage ---------------------------------------------------------------------------------
            phys = cat == 0
            # crit ratio stage: move + Super Luck/Scope Lens/Leek; Merciless always crits a poisoned target;
            # the chance is 1/24, 1/8, 1/2, then certain; willCrit moves always crit
            stage_ = tab["mv_crit"][m] + tab["ab_critplus"][ab_a] + tab["it_critplus"][it_a] + \
                jnp.where(tab["it_leek"][it_a] & tab["sp_leek"][st["species"][ma]], 2, 0)
            stage_ = jnp.where(tab["ab_merciless"][ab_a] & ((st["status"][A] == 3) | (st["status"][A] == 4)), 5, stage_)
            stage_ = jnp.clip(stage_, 0, 4)
            roll_c = jnp.stack([jnp.array(False), rng["chance"]["1_24"], rng["chance"]["1_8"], rng["chance"]["1_2"], jnp.array(True)])
            crit = (roll_c[stage_] | tab["mv_willcrit"][m]) if stage_.ndim == 0 else (roll_c[stage_] | tab["mv_willcrit"][m])
            hp_a, mx_a = st["hp"][ma], st["stats"][ma, 0]
            sheer = tab["ab_sheerforce"][ab_a] & tab["mv_has_sec"][m]
            it_d = st["item"][A]  # defenders' items, per slot
            ally_slot = a ^ 1
            cond = {
                "always": jnp.ones(4, bool), "bp<=60": jnp.broadcast_to(tab["mv_bp"][m] <= 60, (4,)),
                "punch": tab["mv_f"]["punch"][m], "bite": tab["mv_f"]["bite"][m], "contact": tab["mv_f"]["contact"][m],
                "slicing": tab["mv_f"]["slicing"][m], "pulse": tab["mv_f"]["pulse"][m], "sound": tab["mv_f"]["sound"][m],
                "recoil": tab["mv_recoil"][m], "sheerforce": sheer,
                "analytic": ~((~done & alive & (ids != a)).any()),
                "rivalry_same": (st["gender"][ma] > 0) & (st["gender"][ma] == st["gender"][A]),
                "rivalry_diff": (st["gender"][ma] > 0) & (st["gender"][A] > 0) & (st["gender"][ma] != st["gender"][A]),
                "sand_rgs": (st["weather"] == 3) & ((mtype == tab["t_rock2"]) | (mtype == tab["t_ground"]) | (mtype == tab["t_steel"])),
                "spec_burn": (st["status"][ma] == 1) & ~phys, "phys_psn": ((st["status"][ma] == 3) | (st["status"][ma] == 4)) & phys,
                "status": st["status"][ma] > 0, "dstatus": st["status"][A] > 0,
                "hp_third": hp_a * 3 <= mx_a, "hp_half": hp_a * 2 <= mx_a,
                "phys": phys, "spec": ~phys, "full_hp": st["hp"][A] == st["stats"][A, 0],
                "se": texp > 0, "resisted": (texp < 0) & (texp > -99), "crit": crit,
                "fresh_target": st["fresh"][A] > 0, "sun": st["weather"] == 1,
                "electric_terrain": st["terrain"] == 1, "grassy_terrain": st["terrain"] == 2,
                "pikachu": tab["sp_pikachu"][st["species"][ma]], "retyped": retyped, "not_self": jnp.ones(4, bool),
                "ally_plus_minus": tab["ab_plusminus"][st["ability"][A[ally_slot]]] & alive[ally_slot],
                "normal_hit": jnp.ones(4, bool), "target_moved": done | ~alive, "sandstorm": st["weather"] == 3,
                "snow": st["weather"] == 4, "confused": jnp.zeros(4, bool),
                "ff_fire": jnp.broadcast_to((st["ff"][ma] > 0) & (mtype == T.load().types.index("Fire")), (4,))}
            for i_, tname in enumerate(E.TYPES):
                cond["mt:" + tname] = mtype == T.load().types.index(tname)
            cvec = jnp.stack([jnp.broadcast_to(cond[n], (4,)) for n in E.COND_NAMES])  # [n_cond, 4 defenders]
            d4 = jnp.arange(4)[:, None]

            def contribs(key, idx, cv=cvec):
                """(priority, multiplier) of each of a table's 2 rules for holders idx[4]; rules whose conditions
                fail contribute 4096."""
                r = tab["rules"].get(key)
                if r is None:
                    return jnp.zeros((4, 2), jnp.int32), jnp.full((4, 2), 4096, jnp.int32), jnp.zeros((4, 2), bool)
                ok = cv[r["c1"][idx], d4] & cv[r["c2"][idx], d4]
                return r["prio"][idx], jnp.where(ok, r["mult"][idx], 4096), ok

            def fold(parts):
                pr = jnp.concatenate([x[0] for x in parts], axis=1)
                mu = jnp.concatenate([x[1] for x in parts], axis=1)
                order = jnp.argsort(-pr, axis=1, stable=True)
                mu = jnp.take_along_axis(mu, order, axis=1)
                acc = jnp.full(4, 4096)
                for k_ in range(mu.shape[1]):
                    acc = _chain(acc, mu[:, k_])
                return acc

            b4 = lambda x: jnp.full(4, x)  # holder index broadcast over the 4 defenders
            atk_side, atk_item = b4(ab_a), b4(it_a)
            # accuracy: base, Wonder Skin, the event chain, then the roll (No Guard / always-hit / Poison Toxic skip it)
            n_vs = (tab["ab_victorystar"][st["ability"][A[a]]] & alive[a]).astype(jnp.int32) + \
                   (tab["ab_victorystar"][st["ability"][A[ally_slot]]] & alive[ally_slot]).astype(jnp.int32)  # one per holder
            acc0 = jnp.where(tab["ab_wonderskin"][ab_d] & (cat == 2), 50, tab["mv_acc"][m])
            acc_f = modify(acc0, fold([contribs("ab_acc", atk_side), contribs("it_acc", atk_item), contribs("dab_acc", ab_d),
                                       contribs("dit_acc", it_d), (jnp.full((4, 1), -1), jnp.where(n_vs == 2, _chain(4506, 4506), jnp.where(n_vs == 1, 4506, 4096)) * jnp.ones((4, 1), jnp.int32), None)]))
            noguard = tab["ab_noguard"][ab_a] | tab["ab_noguard"][ab_d]
            flashed = (absorbed & (kind_d == 3)).any()  # Flash Fire sets move.accuracy = true for the rest
            always = tab["mv_acc_true"][m] | noguard | flashed | (tab["mv_toxic"][m] & _has(types[a], tab["t_poison"]))
            hit_roll = always | (acc_f > rng["acc"][a])
            mask = mask & hit_roll
            # BasePower: attacker ability/item, defender ability, ally abilities, auras, terrain
            ally_parts = []
            for h in (a, ally_slot):  # Battery/Power Spot/Steely Spirit: the attacker and its ally
                cv_h = cvec.at[E.COND_NAMES.index("not_self")].set(jnp.broadcast_to(h != a, (4,)))
                pr_, mu_, ok_ = contribs("aab_bp", b4(st["ability"][A[h]]), cv_h)
                ally_parts.append((pr_, jnp.where(alive[h], mu_, 4096), ok_))
            ab_all = st["ability"][A]
            aura_d = ((tab["ab_darkaura"][ab_all] & alive).any() & (mtype == tab["t_dark"])) | \
                     ((tab["ab_fairyaura"][ab_all] & alive).any() & (mtype == tab["t_fairy"]))
            aura_m = jnp.where(aura_d, jnp.where((tab["ab_aurabreak"][ab_all] & alive).any(), 3072, 5448), 4096)
            gdef = grounded
            terr_m = jnp.where(grounded[a] & (tab["tmod"][st["terrain"], mtype] != 4096), tab["tmod"][st["terrain"], mtype], 4096)
            terr_m = jnp.where((st["terrain"] == 3) & (mtype == tab["t_dragon"]) & gdef, 2048, terr_m)
            terr_m = jnp.where((st["terrain"] == 2) & tab["mv_halve_grassy"][m] & gdef, 2048, terr_m)
            bp_chain = fold([contribs("ab_bp", atk_side), contribs("it_bp", atk_item), contribs("dab_bp", ab_d)] + ally_parts +
                            [(jnp.full((4, 1), 20), jnp.broadcast_to(aura_m, (4,))[:, None], None),
                             (jnp.full((4, 1), 6), terr_m[:, None], None)])
            bp = modify(tab["mv_bp"][m], bp_chain)
            bp = jnp.maximum(bp, 1)
            # attack / defense stats (stat stages first, then the event chain)
            atk_b = st["boosts"][ma, jnp.where(phys, 0, 2)]
            atk_b = jnp.where(tab["mv_ign_off"][m], 0, jnp.where(crit, jnp.maximum(atk_b, 0), atk_b))
            dfn_b = st["boosts"][A, jnp.where(phys, 1, 3)]
            dfn_b = jnp.where(tab["mv_ign_def"][m], 0, jnp.where(crit, jnp.minimum(dfn_b, 0), dfn_b))
            atk = stage(st["stats"][ma, jnp.where(phys, 1, 3)], atk_b)
            dfn = stage(st["stats"][A, jnp.where(phys, 2, 4)], dfn_b)
            ev_a = jnp.where(phys, 0, 1)

            def stat_parts(kind, idx):
                pa, ma_, _ = contribs(f"{kind}_atk", idx)
                ps, ms, _ = contribs(f"{kind}_spa", idx)
                return (jnp.where(phys, pa, ps), jnp.where(phys, ma_, ms), None)
            atk_chain = fold([stat_parts("ab", atk_side), stat_parts("it", atk_item), stat_parts("dab", ab_d)])
            atk = modify(atk, atk_chain)
            def def_parts(idx):
                pa, ma_, _ = contribs("ab_def", idx)
                ps, ms, _ = contribs("ab_spd", idx)
                return (jnp.where(phys, pa, ps), jnp.where(phys, ma_, ms), None)
            dfn = modify(dfn, fold([def_parts(ab_d)]))
            w = st["weather"]
            boost_def = ((w == 4) & phys & _has(types, tab["t_ice"])) | ((w == 3) & ~phys & _has(types, tab["t_rock"]))
            dfn = jnp.where(boost_def, modify(dfn, 6144), dfn)
            stab = jnp.where(_has(types[a], mtype), jnp.where(tab["ab_adapt"][ab_a], 8192, 6144), 4096)
            tside = ids // 2
            screen = jnp.where(phys, (st["reflect_t"][tside] > 0), (st["ls_t"][tside] > 0)) | (st["av_t"][tside] > 0)
            screen_m = jnp.where(screen & ~crit & (ids != a), 2732, 4096)
            fg = (tab["ab_friendguard"][ab_all] & alive)  # Friend Guard on the target's ally
            fg_m = jnp.where(fg[ids ^ 1], 3072, 4096)
            dmg_parts = [contribs("ab_dmg", atk_side), contribs("it_dmg", atk_item), contribs("dab_dmg", ab_d),
                         contribs("dit_dmg", it_d), (jnp.zeros((4, 1), jnp.int32), screen_m[:, None], None),
                         (jnp.zeros((4, 1), jnp.int32), fg_m[:, None], None)]
            final = fold(dmg_parts)
            burn_halves = (st["status"][ma] == 1) & phys & ~tab["ab_guts"][ab_a]
            dmg = damage(LEVEL, bp, atk, dfn, rng["rolls"][a], crit, stab, texp, burn_halves, spread, tab["wmod"][w, mtype], final)
            hit = mask & (cat < 2)
            hp0 = st["hp"][A]
            # resist berries: eaten when their rule fired on a hit that connected
            _, bm, bok = contribs("dit_dmg", it_d)
            ate_berry = hit & (bm != 4096).any(axis=1) & (tab["it_berry_type"][it_d] >= 0)
            # Focus Sash: a hit that would KO a full-HP holder leaves it at 1 HP and consumes the item
            sash = hit & tab["it_sash"][it_d] & (hp0 == st["stats"][A, 0]) & (dmg >= hp0)
            dmg = jnp.where(sash, hp0 - 1, dmg)
            hp = put(st["hp"], A, jnp.where(hit, jnp.maximum(hp0 - dmg, 0), hp0))
            gone = sash | ate_berry
            st = {**st, "hp": hp, "item": put(st["item"], A, jnp.where(gone, 0, st["item"][A])),
                  "lost": put(st["lost"], A, jnp.where(gone, 1, st["lost"][A]))}
            hit_alive = hit & (hp[A] > 0)
            # secondary effects on surviving targets (Sheer Force removes them)
            sec_on = ~sheer
            proc = hit_alive & (rng["sec"][a] < tab["mv_sec_chance"][m]) & sec_on
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
            st = add_boost(st, me & (hit & (rng["sec"][a] < tab["mv_sec_chance"][m])).any() & sec_on, tab["mv_sec_self_boost"][m])
            st = add_boost(st, me & landed & ongoing, tab["mv_selfboost"][m])  # skipped when the move ends the battle
            # Life Orb recoil: any attack that hit something (not with Sheer Force)
            recoil = jnp.where(tab["it_lifeorb"][it_a] & hit.any() & (cat < 2) & ~sheer & ~tab["ab_magicguard"][ab_a], jnp.maximum(st["stats"][ma, 0] // 10, 1), 0)
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
    res = _residual(tab, st, rng["tiebreak"])
    # bookkeeping after the residuals: the stall counter only survives a turn in which the Pokemon used Protect
    # again, activeTurns ticks, and nobody is "freshly switched in" any more
    res = {**res, "fresh": put(res["fresh"], res["active"], jnp.zeros(4, jnp.int32)),
           "age": put(res["age"], res["active"], res["age"][res["active"]] + 1),
           "stall": put(res["stall"], res["active"], jnp.where(prot_final, res["stall"][res["active"]], 0))}
    return jax.tree.map(lambda p, q: jnp.where(winner(st) != -1, p, q), st, res)


def _residual(tab, st, tie):
    """End of turn. Showdown runs each residual effect (weather, terrain heal, status) over the active slots in
    speed order and ends the battle the moment a side is wiped out. Timers tick down afterwards; a timer
    that reaches 0 ends its effect (weather/terrain are checked before they act)."""
    A = st["active"]
    spe = _speeds(tab, st)
    spe = jnp.where(st["tr_t"] > 0, -spe, spe)
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
            ab = s["ability"][mon]
            guard = tab["ab_magicguard"][ab]  # Magic Guard takes no indirect damage
            if kind == 0:  # weather: sandstorm chip, then the holder's own weather ability (same handler pass)
                sand = (w == 3) & ~(tab["sand_imm"][jnp.maximum(types, 0)] & (types >= 0)).any() & ~tab["ab_sand_imm"][ab]
                new = jnp.where(sand & ~guard, jnp.maximum(hp - jnp.maximum(mx // 16, 1), 0), hp)
                sun_n, rain_n, snow_n = tab["ab_sun_dmg"][ab], tab["ab_rain_heal"][ab], tab["ab_snow_heal"][ab]
                new = jnp.where((w == 1) & (sun_n > 0) & ~guard, jnp.maximum(new - jnp.maximum(mx // jnp.maximum(sun_n, 1), 1), 0), new)
                new = jnp.where((w == 2) & (rain_n > 0), jnp.minimum(new + jnp.maximum(mx // jnp.maximum(rain_n, 1), 1), mx), new)
                new = jnp.where((w == 4) & (snow_n > 0), jnp.minimum(new + jnp.maximum(mx // jnp.maximum(snow_n, 1), 1), mx), new)
            elif kind == 1:  # grassy terrain heal (runs before the terrain's own end handler)
                heal = (s["terrain"] == 2) & _grounded(tab, s, mon)
                new = jnp.where(heal, jnp.minimum(hp + jnp.maximum(mx // 16, 1), mx), hp)
            elif kind == 2:  # Leftovers (order 5, after the terrain heal)
                new = jnp.where(tab["it_leftovers"][s["item"][mon]], jnp.minimum(hp + jnp.maximum(mx // 16, 1), mx), hp)
            elif kind == 5:  # Speed Boost (order 28): +1 Spe if it has been out for a turn
                sb = tab["ab_speedboost"][ab] & (s["fresh"][mon] == 0) & live
                newb = jnp.where(jnp.arange(5) == 4, jnp.minimum(s["boosts"][mon] + 1, 6), s["boosts"][mon])
                s = {**s, "boosts": put(s["boosts"], mon, jnp.where(sb, newb, s["boosts"][mon]))}
                new = hp
            else:  # kind 3: poison/toxic (residual order 9), kind 4: burn (order 10)
                status = s["status"][mon]
                tox = jnp.where((status == 4) & (kind == 3), jnp.minimum(s["tox"][mon] + 1, 15), s["tox"][mon])
                base = jnp.maximum(mx // 16, 1)
                burn = jnp.where(tab["ab_heatproof"][ab], jnp.maximum(base // 2, 1), base)
                res = jnp.where((status == 1) & (kind == 4), burn, 0)
                res = jnp.where((status == 3) & (kind == 3), jnp.maximum(mx // 8, 1), res)
                res = jnp.where((status == 4) & (kind == 3), base * tox, res)
                res = jnp.where(guard, 0, res)
                new = jnp.maximum(hp - res, 0)
                s = {**s, "tox": put(s["tox"], mon, jnp.where(live, tox, s["tox"][mon]))}
            return {**s, "hp": put(s["hp"], mon, jnp.where(live, new, hp))}
        return body

    for kind in range(6):
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
        sw = {**sw, "age": put(sw["age"], mon, 1), "fresh": put(sw["fresh"], mon, 0)}  # activeTurns ticks to 1 before its first move
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
