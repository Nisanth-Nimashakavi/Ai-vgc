"""Hard rules for moves that are certain to fail, applied to the action mask at play time.

The network never sees these interactions (abilities are only an ID embedding, damage
features assume every move lands), so it sometimes clicks Fake Out into Armor Tail or
Close Combat into a Ghost. `ai_vgc.nn.failures` measured which of these happen; this
removes them before the network picks. Only cases that fail whatever the opponent does
are blocked, apart from the target switching out or Terastallizing. A slot always keeps
at least one legal action.
"""

from __future__ import annotations

import numpy as np
from poke_env.battle import (
    DoubleBattle,
    Field,
    Move,
    MoveCategory,
    Pokemon,
    PokemonType,
    SideCondition,
    Status,
    Weather,
)
from poke_env.battle.effect import Effect

# Types that can't get a status condition.
STATUS_IMMUNE_TYPES = {
    Status.PSN: {PokemonType.POISON, PokemonType.STEEL},
    Status.TOX: {PokemonType.POISON, PokemonType.STEEL},
    Status.BRN: {PokemonType.FIRE},
    Status.PAR: {PokemonType.ELECTRIC},
}
# Abilities that pull single-target moves of a type onto their holder.
REDIRECT_ABILITIES = {"lightningrod": PokemonType.ELECTRIC, "stormdrain": PokemonType.WATER}


def _status_fails(attacker: Pokemon | None, target: Pokemon, move: Move) -> bool:
    """Whether a status move is known to fail on this foe."""
    if target.ability == "goodasgold":
        return True
    if target.ability == "magicbounce" and "reflectable" in move.flags:
        return True
    if "powder" in move.flags and (
        PokemonType.GRASS in target.types
        or target.ability == "overcoat"
        or target.item == "safetygoggles"
    ):
        return True
    if move.id == "thunderwave" and PokemonType.GROUND in target.types:
        return True
    if move.status is not None:
        if target.status is not None:
            return True
        immune = STATUS_IMMUNE_TYPES.get(move.status, set())
        corrosion = attacker is not None and attacker.ability == "corrosion"
        if immune & set(target.types) and not (corrosion and move.status in (Status.PSN, Status.TOX)):
            return True
    return False


PRIORITY_BLOCKERS = {"armortail", "dazzling", "queenlymajesty"}
MOLD_BREAKERS = {"moldbreaker", "teravolt", "turboblaze"}
# Abilities that make the holder immune to a move type (Lightning Rod / Storm Drain included).
TYPE_ABSORB = {
    "flashfire": PokemonType.FIRE, "wellbakedbody": PokemonType.FIRE,
    "waterabsorb": PokemonType.WATER, "stormdrain": PokemonType.WATER, "dryskin": PokemonType.WATER,
    "voltabsorb": PokemonType.ELECTRIC, "lightningrod": PokemonType.ELECTRIC, "motordrive": PokemonType.ELECTRIC,
    "sapsipper": PokemonType.GRASS, "levitate": PokemonType.GROUND, "eartheater": PokemonType.GROUND,
}
FLAG_IMMUNE = {"bulletproof": "bullet", "soundproof": "sound", "windrider": "wind"}
# Moves or abilities whose type isn't the listed one, so type immunities can't be judged.
VARIABLE_TYPE = {"terablast", "terastarstorm", "weatherball", "judgment", "multiattack", "ivycudgel",
                 "ragingbull", "revelationdance", "naturalgift", "hiddenpower", "technoblast", "aurawheel"}
TYPE_CHANGING = {"pixilate", "aerilate", "refrigerate", "galvanize", "normalize", "liquidvoice"}
IGNORES_IMMUNITY = {"thousandarrows", "struggle"}
FIRST_TURN_ONLY = {"fakeout", "firstimpression", "matblock"}
SCREENS = {"tailwind": SideCondition.TAILWIND, "reflect": SideCondition.REFLECT,
           "lightscreen": SideCondition.LIGHT_SCREEN, "auroraveil": SideCondition.AURORA_VEIL}
WEATHERS = {"raindance": Weather.RAINDANCE, "sunnyday": Weather.SUNNYDAY, "sandstorm": Weather.SANDSTORM,
            "snowscape": Weather.SNOWSCAPE}
TERRAINS = {"electricterrain": Field.ELECTRIC_TERRAIN, "grassyterrain": Field.GRASSY_TERRAIN,
            "mistyterrain": Field.MISTY_TERRAIN, "psychicterrain": Field.PSYCHIC_TERRAIN}
SELF_HEALS = {"recover", "roost", "slackoff", "softboiled", "milkdrink", "shoreup", "synthesis", "moonlight",
              "morningsun", "healorder", "junglehealing", "lunarblessing"}


def effective_priority(battle: DoubleBattle, user: Pokemon, move: Move) -> int:
    p = move.priority
    if user.ability == "prankster" and move.category == MoveCategory.STATUS:
        p += 1
    elif user.ability == "galewings" and move.type == PokemonType.FLYING and user.current_hp_fraction == 1:
        p += 1
    elif user.ability == "triage" and "heal" in move.flags:
        p += 3
    if move.id == "grassyglide" and Field.GRASSY_TERRAIN in battle.fields and battle.is_grounded(user):
        p += 1
    return p


def _ignores_abilities(user: Pokemon) -> bool:
    return user.ability in MOLD_BREAKERS


def foe_blocked(battle: DoubleBattle, user: Pokemon, ally: Pokemon | None, foes: list[Pokemon], move: Move,
                tgt: Pokemon, single: bool) -> str | None:
    """Why `move` from `user` can't affect the opposing `tgt`, else None. Works for either side
    (the encoder asks it for the opponent's moves into us too). `foes` are tgt's side, `ally`
    is user's partner; `single` = a single-target move, which Lightning Rod / Storm Drain pull away."""
    status = move.category == MoveCategory.STATUS
    breaker = _ignores_abilities(user)
    prio = effective_priority(battle, user, move)
    if prio > 0:
        if not breaker and any(f.ability in PRIORITY_BLOCKERS for f in foes):
            return "priority into Armor Tail / Dazzling / Queenly Majesty"
        if Field.PSYCHIC_TERRAIN in battle.fields and battle.is_grounded(tgt):
            return "priority into Psychic Terrain"
        if status and user.ability == "prankster" and PokemonType.DARK in tgt.types:
            return "Prankster into a Dark type"
    if not status and move.id not in VARIABLE_TYPE and user.ability not in TYPE_CHANGING \
            and move.id not in IGNORES_IMMUNITY and tgt.damage_multiplier(move) == 0 \
            and not (user.ability in ("scrappy", "mindseye")
                     and move.type in (PokemonType.NORMAL, PokemonType.FIGHTING)) \
            and tgt.item != "ringtarget":
        return "type immunity"
    if not breaker and move.id not in VARIABLE_TYPE and user.ability not in TYPE_CHANGING:
        if TYPE_ABSORB.get(tgt.ability or "") == move.type:
            if not (move.type == PokemonType.GROUND and status):
                return f"ability immunity ({tgt.ability})"
        if FLAG_IMMUNE.get(tgt.ability or "") in move.flags:
            return f"ability immunity ({tgt.ability})"
        # A single-target Water/Electric move is pulled into any other Storm Drain / Lightning Rod.
        others = [m for m in [ally, *foes] if m is not None and m is not tgt]
        if single and move.id != "snipeshot" and user.ability not in ("stalwart", "propellertail") \
                and any(REDIRECT_ABILITIES.get(m.ability or "") == move.type for m in others):
            return "redirected by Lightning Rod / Storm Drain"
    if not status and move.type == PokemonType.GROUND and tgt.item == "airballoon" \
            and move.id not in IGNORES_IMMUNITY:
        return "Air Balloon"
    if status and not breaker and _status_fails(user, tgt, move):
        return "status move into an immunity"
    return None


def move_fails(battle: DoubleBattle, pos: int, move: Move, target: int) -> str | None:
    """Why `move` from active slot `pos` into showdown target `target` is sure to fail, else None."""
    user = battle.active_pokemon[pos]
    if user is None:
        return None
    foes = [m for m in battle.opponent_active_pokemon if m is not None and not m.fainted]
    ally = battle.active_pokemon[1 - pos]
    ally = ally if ally is not None and not ally.fainted else None
    hits_foe = target in (battle.OPPONENT_1_POSITION, battle.OPPONENT_2_POSITION)
    tgt = None
    if hits_foe:
        tgt = battle.opponent_active_pokemon[target - 1]
        tgt = tgt if tgt is not None and not tgt.fainted else None
    elif target in (battle.POKEMON_1_POSITION, battle.POKEMON_2_POSITION) and -target - 1 != pos:
        tgt = ally
    status = move.category == MoveCategory.STATUS

    if move.id in FIRST_TURN_ONLY and not user.first_turn:
        return "not the first turn out"
    if hits_foe and tgt is not None:
        why = foe_blocked(battle, user, ally, foes, move, tgt, single=True)
        if why:
            return why
    if tgt is ally and ally is not None and status and ally.ability == "goodasgold":
        return "Good as Gold ally"

    # Setup that can't work.
    if move.id in SCREENS and SCREENS[move.id] in battle.side_conditions:
        return f"{move.id} already up"
    if move.id == "auroraveil" and not ({Weather.SNOWSCAPE, Weather.HAIL} & set(battle.weather)):
        return "Aurora Veil without snow"
    if move.id in WEATHERS and WEATHERS[move.id] in battle.weather:
        return "weather already up"
    if move.id in TERRAINS and TERRAINS[move.id] in battle.fields:
        return "terrain already up"
    if move.id == "stockpile" and Effect.STOCKPILE3 in user.effects:
        return "Stockpile at 3"
    if move.id in SELF_HEALS and user.current_hp_fraction == 1:
        return "heal at full HP"
    if move.id == "lifedew" and user.current_hp_fraction == 1 and (ally is None or ally.current_hp_fraction == 1):
        return "Life Dew at full HP"
    if move.id in ("healpulse", "floralhealing") and tgt is not None and tgt.current_hp_fraction == 1:
        return "heal at full HP"
    return None


def slot_moves(battle: DoubleBattle, pos: int) -> list[Move]:
    """The move list that DoublesEnv's action ids index into."""
    mon = battle.active_pokemon[pos]
    avail = battle.available_moves[pos]
    known = list(mon.moves.values())[:4] if mon is not None else []
    return avail if len(avail) == 1 and avail[0].id not in {m.id for m in known} else known


def apply_rules(battle: DoubleBattle, mask: np.ndarray) -> np.ndarray:
    """Copy of the [2, 107] action mask without actions that are sure to fail."""
    out = mask.copy()
    for pos in range(2):
        if battle.active_pokemon[pos] is None:
            continue
        mvs = slot_moves(battle, pos)
        for a in np.flatnonzero(mask[pos]):
            if a < 7:
                continue
            i, target = (a - 7) % 20 // 5, (a - 7) % 5 - 2
            try:
                fails = i < len(mvs) and move_fails(battle, pos, mvs[i], target)
            except KeyError:  # pseudo-moves like "recharge" have no data entry
                fails = False
            if fails:
                out[pos, a] = False
        if not out[pos].any():
            out[pos] = mask[pos]
    return out
