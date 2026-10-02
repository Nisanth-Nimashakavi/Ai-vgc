"""Hard rules for moves that are certain to fail, applied to the action mask at play time.

Abilities covered (every one on pokemondb.net/ability that can make a move fail or miss its
target): redirection (Lightning Rod, Storm Drain; Stalwart / Propeller Tail / Snipe Shot ignore
it), absorption (Flash Fire, Well-Baked Body, Water Absorb, Storm Drain, Dry Skin, Volt Absorb,
Lightning Rod, Motor Drive, Sap Sipper, Levitate, Earth Eater), move flags (Soundproof,
Bulletproof, Wind Rider), priority (Armor Tail, Dazzling, Queenly Majesty, Prankster into Dark),
status (Good as Gold, Magic Bounce, Overcoat, Insomnia, Vital Spirit, Sweet Veil, Immunity,
Pastel Veil, Water Veil, Water Bubble, Thermal Exchange, Limber, Magma Armor, Comatose, Purifying
Salt, Leaf Guard in sun, Own Tempo, Oblivious, Aroma Veil, Clear Body, White Smoke, Full Metal
Body, Mirror Armor, Hyper Cutter, Big Pecks, Guard Dog, Suction Cups, Sticky Hold), Wonder
Guard, Damp, Neutralizing Gas, and Mold Breaker / Teravolt / Turboblaze ignoring them. A Mega
Evolution changes the ability (see `ability`).

The network never sees these interactions (abilities are only an ID embedding, damage
features assume every move lands), so it sometimes clicks Fake Out into Armor Tail or
Close Combat into a Ghost. `ai_vgc.nn.failures` measured which of these happen; this
removes them before the network picks. Only cases that fail whatever the opponent does
are blocked, apart from the target switching out or Terastallizing. A slot always keeps
at least one legal action.
"""

from __future__ import annotations

import contextvars
from functools import cache
from pathlib import Path

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
from poke_env.data import GenData, to_id_str

# Types that can't get a status condition.
STATUS_IMMUNE_TYPES = {
    Status.PSN: {PokemonType.POISON, PokemonType.STEEL},
    Status.TOX: {PokemonType.POISON, PokemonType.STEEL},
    Status.BRN: {PokemonType.FIRE},
    Status.PAR: {PokemonType.ELECTRIC},
}
# Abilities that pull single-target moves of a type onto their holder.
REDIRECT_ABILITIES = {"lightningrod": PokemonType.ELECTRIC, "stormdrain": PokemonType.WATER}


# Set while `apply_rules` runs: only the action filter guesses abilities. The encoder's "blocked"
# feature calls the same checks and must stay as the network was trained (unknown = no ability).
_GUESS: contextvars.ContextVar[bool] = contextvars.ContextVar("guess_abilities", default=False)


# Mega forme a user's own Mega Evolution this turn gives it (set per action by `apply_rules`).
_MEGA_NOW: contextvars.ContextVar[Pokemon | None] = contextvars.ContextVar("mega_now", default=None)


@cache
def mega_stones() -> dict[str, dict[str, str]]:
    """Mega Stone id -> {base species id: Mega forme id}, from the Showdown checkout's item data
    (Champions adds stones poke-env doesn't know)."""
    import re

    out: dict[str, dict[str, str]] = {}
    for root in ("pokemon-showdown-mc", "pokemon-showdown"):
        path = Path(__file__).resolve().parents[3] / root / "data" / "items.ts"
        if not path.exists():
            continue
        for m in re.finditer(r"\n\t(\w+): \{\n\t\tname: \"[^\"]+\",(.*?)\n\t\},", path.read_text(), re.S):
            if ms := re.search(r"megaStone: \{([^}]*)\}", m.group(2)):
                for base, forme in re.findall(r'"([^"]+)": "([^"]+)"', ms.group(1)):
                    out.setdefault(m.group(1), {})[to_id_str(base)] = to_id_str(forme)
    return out


def mega_ability(mon: Pokemon, item: str | None = None) -> str | None:
    """The ability `mon` has after Mega Evolving with `item` (default: what it holds)."""
    forme = mega_stones().get(to_id_str(item or mon.item or ""), {}).get(to_id_str(mon.base_species))
    entry = GenData.from_gen(9).pokedex.get(forme or "", {})
    return to_id_str(entry["abilities"]["0"]) if entry else None


def is_mega(mon: Pokemon) -> bool:
    return "mega" in mon.species or mon.forme_change_ability is not None


def may_mega(battle: DoubleBattle, mon: Pokemon) -> str | None:
    """For an opposing Pokemon that can still Mega Evolve this turn, the ability it would get
    (Mega Evolution happens before moves, so its current ability may be gone by the time our
    move lands), else None. It can when its side hasn't Mega Evolved yet and it holds, or by
    MunchStats usage more often than not runs, a stone for its species."""
    if battle.opponent_used_mega_evolve or is_mega(mon):
        return None
    stones = {k: v for k, v in mega_stones().items() if to_id_str(mon.base_species) in v}
    if not stones:
        return None
    if mon.item and mon.item != "unknown_item":
        return mega_ability(mon, mon.item) if mon.item in stones else None
    from ai_vgc.munchstats import cached, to_id

    items = cached().get(to_id(mon.species), {}).get("items") or []
    held = [(to_id(i), p) for i, p in items if to_id(i) in stones]
    if held and sum(p for _, p in held) >= 50:
        return mega_ability(mon, max(held, key=lambda x: x[1])[0])
    return None


def ability(mon: Pokemon | None, battle: DoubleBattle | None = None) -> str:
    """The Pokemon's ability as far as the rules go. Outside `apply_rules` (the encoder's
    "blocked" feature, which the network was trained with): the one poke-env knows (shown, or a
    Mega forme's), else "". Inside `apply_rules`:
      - our Pokemon Mega Evolving this turn: its Mega forme's ability;
      - an opposing Pokemon that may Mega Evolve first this turn with an ability that differs
        from its current one: "" (can't be relied on either way);
      - Neutralizing Gas on the field: "" for everyone else;
      - an opposing Pokemon that hasn't shown its ability (closed team sheets): the one its
        species runs at least 80% of the time by MunchStats usage (Raichu: Lightning Rod,
        Kommo-o: Soundproof). A 3% Static Raichu then has a move blocked by mistake; far more
        often it would have been pulled away or absorbed."""
    if mon is None:
        return ""
    if not _GUESS.get():
        return mon.ability or ""
    if battle is not None and any(m is not None and m is not mon and m.ability == "neutralizinggas"
                                  for m in [*battle.active_pokemon, *battle.opponent_active_pokemon]):
        return ""  # Neutralizing Gas on the field: no other ability works
    if _MEGA_NOW.get() is mon:
        return mega_ability(mon) or mon.ability or ""
    current = mon.ability
    if not current and battle is not None and mon in battle.opponent_team.values():
        from ai_vgc.munchstats import likely_ability

        current = likely_ability(mon.species)
    if battle is not None and mon in battle.opponent_team.values():
        mega = may_mega(battle, mon)
        if mega is not None and mega != current:
            return ""
    return current or ""


def _status_fails(attacker: Pokemon | None, target: Pokemon, move: Move,
                  battle: DoubleBattle | None = None, partner: Pokemon | None = None) -> bool:
    """Whether a status move is known to fail on this foe."""
    if ability(target, battle) == "goodasgold":
        return True
    if ability(target, battle) == "magicbounce" and "reflectable" in move.flags:
        return True
    if "powder" in move.flags and (
        PokemonType.GRASS in target.types
        or ability(target, battle) == "overcoat"
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
        if ability(target, battle) in STATUS_BLOCK.get(move.status, ()) | {"comatose", "purifyingsalt"}:
            return True
        if ability(partner, battle) in ALLY_STATUS_BLOCK.get(move.status, ()):
            return True
        if ability(target, battle) == "leafguard" and battle is not None and Weather.SUNNYDAY in battle.weather:
            return True
    vol = getattr(move.volatile_status, "name", "").lower()
    if vol:
        if ability(target, battle) in VOLATILE_BLOCK.get(vol, ()):
            return True
        if vol in ("attract", "taunt", "encore", "torment", "disable", "healblock") and \
                "aromaveil" in (ability(target, battle), ability(partner, battle)):
            return True
        if vol == "yawn" and (target.status is not None or ability(target, battle) in STATUS_BLOCK[Status.SLP]
                              or ability(partner, battle) == "sweetveil"):
            return True
    if move.boosts and all(v < 0 for v in move.boosts.values()) and not move.volatile_status:
        blockers = {"clearbody", "whitesmoke", "fullmetalbody"}
        if ability(target, battle) in blockers or ability(target, battle) == "mirrorarmor":
            return True
        stats = set(move.boosts)
        if stats == {"atk"} and ability(target, battle) == "hypercutter":
            return True
        if stats == {"def"} and ability(target, battle) == "bigpecks":
            return True
    if move.id in ("roar", "whirlwind") and ability(target, battle) in ("guarddog", "suctioncups"):
        return True
    if move.id in ("trick", "switcheroo") and ability(target, battle) == "stickyhold":
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
# Abilities that stop a status condition on their holder (Comatose and Purifying Salt stop all).
STATUS_BLOCK = {
    Status.SLP: {"insomnia", "vitalspirit", "sweetveil"},
    Status.PSN: {"immunity", "pastelveil"}, Status.TOX: {"immunity", "pastelveil"},
    Status.BRN: {"waterveil", "waterbubble", "thermalexchange"},
    Status.PAR: {"limber"}, Status.FRZ: {"magmaarmor"},
}
ALLY_STATUS_BLOCK = {Status.SLP: {"sweetveil"}, Status.PSN: {"pastelveil"}, Status.TOX: {"pastelveil"}}
VOLATILE_BLOCK = {"confusion": {"owntempo"}, "attract": {"oblivious"}, "taunt": {"oblivious"}}
SELF_KO = {"explosion", "selfdestruct", "mistyexplosion", "mindblown"}
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
    # Mold Breaker (e.g. Mega Ampharos, Mega Gyarados) also counts when the user Mega Evolves now.
    return ability(user) in MOLD_BREAKERS


def foe_blocked(battle: DoubleBattle, user: Pokemon, ally: Pokemon | None, foes: list[Pokemon], move: Move,
                tgt: Pokemon, single: bool) -> str | None:
    """Why `move` from `user` can't affect the opposing `tgt`, else None. Works for either side
    (the encoder asks it for the opponent's moves into us too). `foes` are tgt's side, `ally`
    is user's partner; `single` = a single-target move, which Lightning Rod / Storm Drain pull away."""
    status = move.category == MoveCategory.STATUS
    breaker = _ignores_abilities(user)
    prio = effective_priority(battle, user, move)
    if prio > 0:
        if not breaker and any(ability(f, battle) in PRIORITY_BLOCKERS for f in foes):
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
        if TYPE_ABSORB.get(ability(tgt, battle)) == move.type:
            if not (move.type == PokemonType.GROUND and status):
                return f"ability immunity ({ability(tgt, battle)})"
        if FLAG_IMMUNE.get(ability(tgt, battle)) in move.flags:
            return f"ability immunity ({ability(tgt, battle)})"
        # A single-target Water/Electric move is pulled into any other Storm Drain / Lightning Rod.
        others = [m for m in [ally, *foes] if m is not None and m is not tgt]
        if single and move.id != "snipeshot" and user.ability not in ("stalwart", "propellertail") \
                and any(REDIRECT_ABILITIES.get(ability(m, battle)) == move.type for m in others):
            return "redirected by Lightning Rod / Storm Drain"
    if not status and move.type == PokemonType.GROUND and tgt.item == "airballoon" \
            and move.id not in IGNORES_IMMUNITY:
        return "Air Balloon"
    if not status and not breaker and move.id not in VARIABLE_TYPE and ability(tgt, battle) == "wonderguard" \
            and tgt.damage_multiplier(move) <= 1:
        return "Wonder Guard"
    partner = next((f for f in foes if f is not tgt), None)
    if status and not breaker and _status_fails(user, tgt, move, battle, partner):
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
    if move.id in SELF_KO and not _ignores_abilities(user) and \
            any(ability(m, battle) == "damp" for m in [ally, *foes] if m is not None):
        return "self-KO move with Damp on the field"
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
    token = _GUESS.set(True)
    try:
        return _apply_rules(battle, mask)
    finally:
        _GUESS.reset(token)


def _apply_rules(battle: DoubleBattle, mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    for pos in range(2):
        if battle.active_pokemon[pos] is None:
            continue
        mvs = slot_moves(battle, pos)
        for a in np.flatnonzero(mask[pos]):
            if a < 7:
                continue
            i, target = (a - 7) % 20 // 5, (a - 7) % 5 - 2
            mega = (a - 7) // 20 == 1  # this action Mega Evolves first
            token = _MEGA_NOW.set(battle.active_pokemon[pos] if mega else None)
            try:
                fails = i < len(mvs) and move_fails(battle, pos, mvs[i], target)
            except KeyError:  # pseudo-moves like "recharge" have no data entry
                fails = False
            finally:
                _MEGA_NOW.reset(token)
            if fails:
                out[pos, a] = False
        if not out[pos].any():
            out[pos] = mask[pos]
    return out
