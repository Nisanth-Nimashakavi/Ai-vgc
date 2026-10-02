"""Turn a poke-env DoubleBattle into fixed-size arrays for the policy network.

The same function encodes replayed logs (training) and live battles (play), so
the two can't drift apart.

Layout, from the deciding player's view (`T` = 12 Pokemon tokens: our team in
`battle.team` order, then the opponent's; an empty token is all zeros):

    tok_cat  int32  [T, 3]        species, item, ability ids (0 = empty/unknown)
    tok_num  f16    [T, N_TOK]    HP, stats, types, status, boosts, flags...
    mv_cat   int32  [T, 4]        move ids; static move features come from
                                  `move_table()` inside the network
    mv_dyn   f16    [T, 4, 2]     PP left, used last turn
    dmg      f16    [4, 4, 2, N_DMG]  active slots (our a, our b, their a, their b)
                                  x their 4 moves x the 2 opposing active slots:
                                  damage range, KO, effectiveness, who's faster,
                                  and whether an ability, priority block or immunity
                                  stops the move on that target (`rules.foe_blocked`)
    glob     f16    [N_GLOB]      weather, terrain, Trick Room, screens, turn, rating
    act_tok  int8   [4]           token index of our a, our b, their a, their b; -1 = none
    ser_tok, ser_mv, ser_glob     Bo3 series context from `battle._series` (`series.py`);
                                  zeros in game 1 and Bo1
    mask     bool   [2, 107]      structurally legal actions per slot (poke-env's
                                  DoublesEnv action ids)

The damage features are the main difference from vgc-bench's encoding: the
network is told damage, KOs and turn order as inputs rather than having to learn
a damage calculator.
"""

from __future__ import annotations

import json
import math
import re
from functools import cache
from pathlib import Path

import numpy as np
from poke_env.battle import (
    DoubleBattle,
    Effect,
    Field,
    Move,
    MoveCategory,
    Pokemon,
    PokemonType,
    SideCondition,
    Status,
    Target,
    Weather,
)
from poke_env.data import GenData

from ai_vgc.calc import damage_pct, effective_speed
from ai_vgc.nn.series import N_SER_GLOB, N_SER_MV, N_SER_TOK

VOCAB_PATH = Path("data/models/nn_vocab.json")
SHOWDOWN_DATA = Path("pokemon-showdown/data")

N_ACT = 107  # poke-env DoublesEnv action ids per slot
MEGA_OFFSET = 20  # ids 27-46 are ids 7-26 plus mega evolution
T = 12

TYPES = [t for t in PokemonType if t not in (PokemonType.THREE_QUESTION_MARKS, PokemonType.STELLAR)]
STATUSES = list(Status)
STATS = ["hp", "atk", "def", "spa", "spd", "spe"]
BOOSTS = ["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"]
EFFECTS = [
    Effect.SUBSTITUTE, Effect.TAUNT, Effect.ENCORE, Effect.CONFUSION, Effect.LEECH_SEED,
    Effect.DISABLE, Effect.TORMENT, Effect.YAWN, Effect.PERISH3, Effect.PERISH2, Effect.PERISH1,
    Effect.SALT_CURE, Effect.HELPING_HAND, Effect.FOLLOW_ME, Effect.RAGE_POWDER, Effect.TRAPPED,
    Effect.HEAL_BLOCK, Effect.ATTRACT, Effect.FLASH_FIRE, Effect.CHARGE,
]
FIELDS = [Field.TRICK_ROOM, Field.ELECTRIC_TERRAIN, Field.GRASSY_TERRAIN, Field.MISTY_TERRAIN,
          Field.PSYCHIC_TERRAIN, Field.GRAVITY, Field.WONDER_ROOM, Field.MAGIC_ROOM]
WEATHERS = [Weather.SUNNYDAY, Weather.RAINDANCE, Weather.SANDSTORM, Weather.SNOWSCAPE, Weather.HAIL,
            Weather.DESOLATELAND, Weather.PRIMORDIALSEA, Weather.DELTASTREAM]
SIDE = [SideCondition.TAILWIND, SideCondition.REFLECT, SideCondition.LIGHT_SCREEN,
        SideCondition.AURORA_VEIL, SideCondition.SAFEGUARD, SideCondition.MIST,
        SideCondition.QUICK_GUARD, SideCondition.WIDE_GUARD]
CATEGORIES = list(MoveCategory)
TARGETS = list(Target)
SINGLE_TARGET = {Target.NORMAL, Target.ANY, Target.ADJACENT_FOE}
ALLY_TARGET = {Target.ADJACENT_ALLY, Target.ADJACENT_ALLY_OR_SELF}
HITS_FOES = SINGLE_TARGET | {Target.ALL_ADJACENT_FOES, Target.ALL_ADJACENT, Target.RANDOM_NORMAL}
PROTECT_IDS = {"protect", "detect", "spikyshield", "kingsshield", "banefulbunker", "silktrap",
               "burningbulwark", "obstruct", "maxguard", "wideguard", "quickguard"}

N_TOK = 1 + 2 + 4 + 6 + 6 + len(TYPES) + len(STATUSES) + len(BOOSTS) + len(EFFECTS) + 9
N_MV = 7 + len(CATEGORIES) + len(TARGETS) + len(TYPES)
N_DMG = 9  # 8 before the "blocked" flag; older checkpoints read the first 8
N_GLOB = len(WEATHERS) + len(FIELDS) + 2 * len(SIDE) + 6


# ---------------------------------------------------------------- vocabularies

def _ts_keys(name: str) -> list[str]:
    text = (SHOWDOWN_DATA / name).read_text()
    return sorted(set(re.findall(r"^\t([a-z0-9]+): \{", text, re.M)))


def build_vocab() -> dict[str, list[str]]:
    g = GenData.from_gen(9)
    return {
        "species": sorted(g.pokedex),
        "moves": sorted(g.moves),
        "items": _ts_keys("items.ts"),
        "abilities": _ts_keys("abilities.ts"),
    }


@cache
def vocab() -> dict[str, dict[str, int]]:
    """Name -> id, with 0 reserved for empty/unknown. Saved once so ids never shift."""
    if not VOCAB_PATH.exists():
        VOCAB_PATH.parent.mkdir(parents=True, exist_ok=True)
        VOCAB_PATH.write_text(json.dumps(build_vocab()))
    v = json.loads(VOCAB_PATH.read_text())
    return {k: {name: i + 1 for i, name in enumerate(names)} for k, names in v.items()}


def vocab_sizes() -> dict[str, int]:
    return {k: len(v) + 1 for k, v in vocab().items()}


@cache
def mega_stones() -> frozenset[str]:
    text = (SHOWDOWN_DATA / "items.ts").read_text()
    blocks = re.split(r"^\t(?=[a-z0-9]+: \{)", text, flags=re.M)
    return frozenset(b.split(":", 1)[0] for b in blocks if "megaStone:" in b)


def can_mega(battle: DoubleBattle, mon: Pokemon) -> bool:
    return not battle.used_mega_evolve and mon.item in mega_stones() and "mega" not in mon.species


# ---------------------------------------------------------------- features

def _team(battle: DoubleBattle) -> list[Pokemon]:
    return list(battle.team.values())[:6]


def _foe_team(battle: DoubleBattle) -> list[Pokemon]:
    team = list(battle.opponent_team.values())
    # Before the opponent reveals anything, open team sheets give the six.
    if not team and battle.teampreview_opponent_team:
        team = list(battle.teampreview_opponent_team)
    return team[:6]


def _mon_num(mon: Pokemon, ours: bool, act_a: bool, act_b: bool) -> list[float]:
    stats = mon.stats if ours and mon.stats else {}
    known = [float(stats.get(s) or 0) / 255 for s in STATS[1:]]
    effects = mon.effects
    return [
        1.0,  # present
        float(ours), float(not ours),
        mon.current_hp_fraction, float(mon.fainted), float(act_a), float(act_b),
        *[mon.base_stats.get(s, 0) / 255 for s in STATS],
        (mon.max_hp or 0) / 255 if ours else 0.0, *known,
        *[float(t in mon.types) for t in TYPES],
        *[float(mon.status == s) for s in STATUSES],
        *[mon.boosts.get(b, 0) / 6 for b in BOOSTS],
        *[float(e in effects) for e in EFFECTS],
        float(mon.first_turn), mon.protect_counter / 3, float(mon.must_recharge),
        float(mon.preparing), float(mon.revealed), float(mon.selected_in_teampreview),
        float(mon.item in mega_stones()),
        float("mega" in mon.species), min(mon.weight, 500) / 500,
    ]


def _move_static(move: Move) -> list[float]:
    return [
        move.base_power / 150, (move.accuracy if move.accuracy is not True else 1.0),
        move.priority / 5, float(move.id in PROTECT_IDS), float(move.self_switch is not False),
        float(bool(move.heal)), float(bool(move.drain)),
        *[float(move.category == c) for c in CATEGORIES],
        *[float(move.target == t) for t in TARGETS],
        *[float(move.type == t) for t in TYPES],
    ]


def move_table() -> np.ndarray:
    """[n_moves + 1, N_MV] static features by move id; row 0 (unknown) is zeros."""
    ids = vocab()["moves"]
    table = np.zeros((len(ids) + 1, N_MV), np.float32)
    for mid, i in ids.items():
        try:
            table[i] = _move_static(Move(mid, gen=9))
        except Exception:
            pass
    return table


def _moves(mon: Pokemon) -> list[Move]:
    return list(mon.moves.values())[:4]


def _dmg(battle: DoubleBattle, att: Pokemon, move: Move, dfn: Pokemon | None, att_spe: float,
         dfn_spe: dict[int, float], ally: Pokemon | None, dfn_side: list[Pokemon]) -> list[float]:
    from ai_vgc.nn.rules import foe_blocked

    if dfn is None:
        return [0.0] * N_DMG
    faster = att_spe > dfn_spe.get(id(dfn), 0)
    if Field.TRICK_ROOM in battle.fields:
        faster = not faster
    try:
        blocked = move.target in HITS_FOES and \
            foe_blocked(battle, att, ally, dfn_side, move, dfn, move.target in SINGLE_TARGET) is not None
    except KeyError:  # pseudo-moves with no data entry
        blocked = False
    row = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, float(faster), 0.0, float(blocked)]
    if move.category == MoveCategory.STATUS or not move.base_power:
        return row
    d = damage_pct(battle, att, dfn, move)
    if d is None:
        return row
    lo, hi = d
    hp = dfn.current_hp_fraction * 100
    eff = dfn.damage_multiplier(move)
    row[1:6] = [min(lo, 200) / 100, min(hi, 200) / 100, float(lo >= hp), float(hi >= hp),
                math.log2(eff) / 2 if eff > 0 else -1.5]
    row[7] = float(eff == 0)
    return row


def encode(battle: DoubleBattle, rating: float = 0.0) -> dict[str, np.ndarray]:
    v = vocab()
    ours, foes = _team(battle), _foe_team(battle)
    mons: list[Pokemon | None] = ours + [None] * (6 - len(ours)) + foes + [None] * (6 - len(foes))
    our_act, foe_act = battle.active_pokemon, battle.opponent_active_pokemon
    act_ids = [[id(m) if m else None for m in our_act], [id(m) if m else None for m in foe_act]]

    tok_cat = np.zeros((T, 3), np.int32)
    tok_num = np.zeros((T, N_TOK), np.float16)
    mv_cat = np.zeros((T, 4), np.int32)
    mv_dyn = np.zeros((T, 4, 2), np.float16)
    dmg = np.zeros((4, 4, 2, N_DMG), np.float16)
    act_tok = np.full(4, -1, np.int8)

    spe: dict[int, float] = {}
    for m in [*our_act, *foe_act]:
        if m is not None:
            spe[id(m)] = effective_speed(battle, m)[0]

    for i, mon in enumerate(mons):
        if mon is None:
            continue
        side = 0 if i < 6 else 1
        a, b = id(mon) == act_ids[side][0], id(mon) == act_ids[side][1]
        slot = 2 * side + (0 if a else 1) if (a or b) else None
        if slot is not None:
            act_tok[slot] = i
        tok_cat[i] = [v["species"].get(mon.species, 0), v["items"].get(mon.item or "", 0),
                      v["abilities"].get(mon.ability or "", 0)]
        tok_num[i] = _mon_num(mon, side == 0, a, b)
        opp = foe_act if side == 0 else our_act
        own = our_act if side == 0 else foe_act
        ally = own[1] if a else own[0]
        ally = ally if ally is not None and not ally.fainted else None
        opp_live = [m for m in opp if m is not None and not m.fainted]
        for k, move in enumerate(_moves(mon)):
            mv_cat[i, k] = v["moves"].get(move.id, 0)
            mv_dyn[i, k] = [move.current_pp / max(move.max_pp, 1), float(move.is_last_used)]
            if slot is not None:
                for j in range(2):
                    dmg[slot, k, j] = _dmg(battle, mon, move, opp[j], spe[id(mon)], spe, ally, opp_live)

    glob = np.zeros(N_GLOB, np.float16)
    turn = battle.turn
    glob[: len(WEATHERS)] = [float(w in battle.weather) for w in WEATHERS]
    o = len(WEATHERS)
    glob[o: o + len(FIELDS)] = [float(f in battle.fields) for f in FIELDS]
    o += len(FIELDS)
    for conds in (battle.side_conditions, battle.opponent_side_conditions):
        glob[o: o + len(SIDE)] = [float(c in conds) for c in SIDE]
        o += len(SIDE)
    glob[o:] = [min(turn, 20) / 20, float(battle.teampreview), rating / 2000,
                float(battle.used_mega_evolve), float(battle.opponent_used_mega_evolve),
                sum(m.fainted for m in foes) / 4 - sum(m.fainted for m in ours) / 4]

    ser_tok = np.zeros((T, N_SER_TOK), np.float16)
    ser_mv = np.zeros((T, 4, N_SER_MV), np.float16)
    ser_glob = np.zeros(N_SER_GLOB, np.float16)
    ctx = getattr(battle, "_series", None)
    if ctx is not None:
        for i, mon in enumerate(mons):
            if mon is not None:
                ser_tok[i] = ctx.tok(mon.name, i < 6)
                for k, move in enumerate(_moves(mon)):
                    ser_mv[i, k] = ctx.mv(mon.name, i < 6, move.id)
        ser_glob[:] = ctx.glob()

    return {
        "tok_cat": tok_cat, "tok_num": tok_num, "mv_cat": mv_cat, "mv_dyn": mv_dyn,
        "dmg": dmg, "glob": glob, "act_tok": act_tok, "mask": legal_mask(battle),
        "ser_tok": ser_tok, "ser_mv": ser_mv, "ser_glob": ser_glob,
    }


# ---------------------------------------------------------------- legality

def _move_targets(move: Move, pos: int, ally_present: bool) -> list[int]:
    """Showdown target numbers a move can take: 1/2 foes, -1/-2 our slots, 0 none."""
    if move.target in SINGLE_TARGET:
        return [1, 2] + ([-2 if pos == 0 else -1] if ally_present else [])
    return [0]


def legal_mask(battle: DoubleBattle) -> np.ndarray:
    """Actions legal from the battle state alone, without a Showdown request.

    Logs carry no requests, so training uses this; live play ANDs it with
    poke-env's request-based mask.
    """
    mask = np.zeros((2, N_ACT), bool)
    team = _team(battle)
    role = battle.player_role
    if battle.teampreview:
        for k, mon in enumerate(team, 1):
            if not mon.selected_in_teampreview:
                mask[:, k] = True
        return mask
    raw = [battle._active_pokemon.get(f"{role}{s}") for s in "ab"]
    active = battle.active_pokemon
    active_ids = {id(m) for m in active if m is not None}
    bench = [k for k, m in enumerate(team, 1) if not m.fainted and id(m) not in active_ids
             and m.selected_in_teampreview]
    # After a faint, Showdown asks only for replacements (when there is a bench).
    replacing = bool(bench) and any(r is not None and r.fainted for r in raw)
    for pos in range(2):
        mon = active[pos]
        if mon is None:
            # Empty or fainted slot: send in a replacement if we have one, else pass.
            if raw[pos] is not None and raw[pos].fainted and bench:
                mask[pos, bench] = True
            else:
                mask[pos, 0] = True
            continue
        if replacing:
            mask[pos, 0] = True
            continue
        if mon.must_recharge:
            mask[pos, 9] = True  # "recharge" is the only option: move 1, no target
            continue
        mask[pos, bench] = True
        mega = can_mega(battle, mon)
        for k, move in enumerate(_moves(mon)):
            for t in _move_targets(move, pos, active[1 - pos] is not None):
                idx = 7 + 5 * k + t + 2
                mask[pos, idx] = True
                if mega:
                    mask[pos, idx + MEGA_OFFSET] = True
    return mask


def joint_mask(mask_b: np.ndarray, action_a: int) -> np.ndarray:
    """Slot b's mask given slot a's action: no double switch-in or double mega."""
    m = mask_b.copy()
    if 1 <= action_a <= 6:
        m[action_a] = False
    if 27 <= action_a <= 46:
        m[27:47] = False
    return m
