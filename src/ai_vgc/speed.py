"""Speed inference: narrow each opposing Pokemon's Speed stat from the order moves happen in.

Within a turn, moves of the same priority go fastest first (slowest first under Trick Room), so
every time one of ours and one of theirs act in the same priority bracket, their effective Speed
is bounded by ours, which we know exactly. Dividing out what we can see (stat stages, Tailwind,
paralysis, weather abilities, a known Choice Scarf, Unburden) bounds their Speed stat.

Each battle keeps `battle._spd`: {opposing Pokemon name: Bounds}. `estimate` turns the bounds
into the Speed the damage/speed features use (`calc.effective_speed`) when the battle's
`_spd_on` is set (NNPlayer.speed_inference); otherwise nothing changes, so older models see
the inputs they were trained on.

Champions Reg M-C stats at level 50: Speed = trunc((base + points + 20) * nature), points 0-32,
nature 0.9 / 1 / 1.1. Turns where the order could come from something hidden are skipped:
- a move whose priority depends on an ability they may have but haven't shown (Prankster,
  Gale Wings, Triage, Quick Draw, Mycelium Might, Stall);
- an item that changes order (a Quick Claw or Custap message), After You, Quash, Instruct, or
  moves called by other moves or abilities ("[from]");
- anything after a mid-turn Speed change (stat stages, Tailwind, Trick Room, paralysis,
  weather, items, switches, Mega Evolution), since Gen 9 re-sorts the turn order at once.
A Choice Scarf they may hold only loosens the bound it would explain (their moving first);
their moving later bounds the stat either way.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from poke_env.battle import (
    AbstractBattle,
    Field,
    Move,
    MoveCategory,
    Pokemon,
    SideCondition,
    Status,
    Weather,
)
from poke_env.data import GenData, to_id_str

BOOST = {-6: 2 / 8, -5: 2 / 7, -4: 2 / 6, -3: 2 / 5, -2: 2 / 4, -1: 2 / 3, 0: 1,
         1: 3 / 2, 2: 4 / 2, 3: 5 / 2, 4: 6 / 2, 5: 7 / 2, 6: 8 / 2}
WEATHER_SPEED = {
    "chlorophyll": {Weather.SUNNYDAY, Weather.DESOLATELAND},
    "swiftswim": {Weather.RAINDANCE, Weather.PRIMORDIALSEA},
    "sandrush": {Weather.SANDSTORM},
    "slushrush": {Weather.SNOWSCAPE, Weather.HAIL},
}
PRIORITY_ABILITIES = {"prankster", "galewings", "triage", "quickdraw", "myceliummight", "stall"}
SKIP_MOVES = {"afteryou", "quash", "instruct"}
# Events after which the rest of the turn's order can't be read (Gen 9 re-sorts by Speed at once).
CHANGES = {"-boost", "-unboost", "-setboost", "-clearallboost", "-clearboost", "-clearnegativeboost",
           "-invertboost", "-swapboost", "-copyboost", "-sidestart", "-sideend", "-fieldstart", "-fieldend",
           "-status", "-curestatus", "-weather", "-item", "-enditem", "-ability", "-endability", "switch",
           "drag", "replace", "detailschange", "-mega", "-transform", "-formechange", "faint"}


@dataclass
class Bounds:
    """Speed stat bounds for one opposing Pokemon (for the base Speed `base`)."""
    base: int
    lo: float = 0.0  # lower bound if they hold no Choice Scarf
    lo_scarf: float = 0.0  # lower bound allowing for a Scarf they may hold
    hi: float = math.inf
    observations: int = 0


def stat_range(base: int) -> tuple[int, int]:
    """(lowest, highest) Speed stat a Pokemon with this base Speed can have at level 50."""
    return math.floor((base + 20) * 0.9), math.floor((base + 52) * 1.1)


def _dex() -> dict:
    return GenData.from_gen(9).pokedex


def _possible_abilities(mon: Pokemon) -> set[str]:
    if mon.ability:
        return {to_id_str(mon.ability)}
    entry = _dex().get(mon.species, {})
    return {to_id_str(a) for a in entry.get("abilities", {}).values()}


def _item(mon: Pokemon) -> str | None:
    """Held item id, "" for none, None when not yet seen."""
    return None if mon.item == "unknown_item" else (mon.item or "")


def priority(battle: AbstractBattle, mon: Pokemon, move_id: str) -> int | None:
    """The move's priority bracket for this user, or None when it depends on a hidden ability."""
    try:
        move = Move(move_id, gen=9)
        p = move.priority
    except Exception:
        return None
    abilities = _possible_abilities(mon)
    known = len(abilities) == 1 and bool(mon.ability)
    if move.id == "grassyglide" and Field.GRASSY_TERRAIN in battle.fields:
        p += 1
    status = move.category == MoveCategory.STATUS
    flying = move.type is not None and move.type.name == "FLYING"
    healing = bool(move.heal) or bool(move.drain)
    for ability, applies, bonus in (("prankster", status, 1), ("galewings", flying, 1), ("triage", healing, 3)):
        if ability in abilities and applies:
            if not known:
                return None
            if ability == "galewings" and mon.current_hp_fraction < 1:
                continue
            p += bonus
    if not known and abilities & {"quickdraw", "myceliummight", "stall"}:
        return None
    if known and abilities & {"myceliummight", "stall"} and (status or "stall" in abilities):
        return None
    return p


def multiplier(battle: AbstractBattle, mon: Pokemon, ours: bool) -> tuple[float, bool] | None:
    """(what their Speed stat is multiplied by right now, whether a Scarf they may hold could
    multiply it by 1.5 more), or None when a hidden ability could change it."""
    m = BOOST[mon.boosts.get("spe", 0)]
    conditions = battle.side_conditions if ours else battle.opponent_side_conditions
    if SideCondition.TAILWIND in conditions:
        m *= 2
    abilities = _possible_abilities(mon)
    known = bool(mon.ability) or ours
    if mon.status == Status.PAR and "quickfeet" not in abilities:
        m *= 0.5
    weather = set(battle.weather or ())
    for ability, w in WEATHER_SPEED.items():
        if ability in abilities and weather & w:
            if not known:
                return None
            m *= 2
    if "surgesurfer" in abilities and Field.ELECTRIC_TERRAIN in battle.fields:
        if not known:
            return None
        m *= 2
    item = _item(mon)
    if "unburden" in abilities and mon.name in battle.__dict__.get("_spd_lost", ()):
        if not known:
            return None
        m *= 2
    scarf_possible = False
    if item == "choicescarf":
        m *= 1.5
    elif item is None and not ours:
        scarf_possible = True
    return m, scarf_possible


def _bounds(battle: AbstractBattle, mon: Pokemon) -> Bounds | None:
    base = mon.base_stats.get("spe") if mon.base_stats else None
    if not base:
        return None
    store = battle.__dict__.setdefault("_spd", {})
    b = store.get(mon.name)
    if b is None or b.base != base:  # first seen, or a new forme (Mega Evolution) changed the base
        b = store[mon.name] = Bounds(base)
    return b


def _process(battle: AbstractBattle) -> None:
    """Read the finished turn's order into the bounds."""
    turn = battle.__dict__.get("_spd_turn")
    battle._spd_turn = None
    if not turn or not turn["moves"]:
        return
    snap, tr = turn["snap"], turn["trickroom"]
    acts = [a for a in turn["moves"] if a["ident"] in snap and a["prio"] is not None]
    for i, a in enumerate(acts):
        for b in acts[i + 1:]:
            if a["ours"] == b["ours"] or a["prio"] != b["prio"]:
                continue
            first, second = (a, b)
            mine, theirs = (first, second) if first["ours"] else (second, first)
            our_eff = snap[mine["ident"]]["eff"]
            t = snap[theirs["ident"]]
            mult, scarf = t["mult"], t["scarf"]
            # They moved first: faster (slower under Trick Room), ties going either way.
            faster = (theirs is first) != tr
            bounds = t["bounds"]
            if faster:
                bounds.lo = max(bounds.lo, our_eff / mult)
                bounds.lo_scarf = max(bounds.lo_scarf, our_eff / (mult * (1.5 if scarf else 1.0)))
            else:
                bounds.hi = min(bounds.hi, our_eff / mult)
            bounds.observations += 1


def _snapshot(battle: AbstractBattle) -> dict:
    """Effective Speeds and multipliers of everything active, taken when the first move of the
    turn goes (after switches and Mega Evolution, which come first)."""
    from ai_vgc.calc import effective_speed

    snap = {}
    for ours, side in ((True, battle.active_pokemon), (False, battle.opponent_active_pokemon)):
        for mon in side:
            if mon is None or mon.fainted:
                continue
            role = battle.player_role if ours else battle.opponent_role
            ident = mon.identifier(role)
            if ours:
                mm = multiplier(battle, mon, True)
                if mm is None or not mon.stats or not mon.stats.get("spe"):
                    continue
                snap[ident] = {"eff": mon.stats["spe"] * mm[0]}
            else:
                mm = multiplier(battle, mon, False)
                b = _bounds(battle, mon)
                if mm is None or b is None:
                    continue
                snap[ident] = {"mult": mm[0], "scarf": mm[1], "bounds": b}
    return snap


def observe(battle: AbstractBattle, split: list[str]) -> None:
    """Feed one protocol message (before poke-env parses it) to the inference."""
    if battle.__dict__.get("_spd_frozen") or len(split) < 2:
        return
    kind = split[1]
    if kind in ("turn", "upkeep", "win", "tie"):
        _process(battle)
        battle._spd_turn = {"moves": [], "snap": None, "trickroom": False, "dirty": False}
        return
    if kind == "-enditem" and len(split) > 2:  # Unburden: the item is gone for good (Grassy Seed goes before turn 1)
        try:  # (poke-env's Pokemon has __slots__, so the record lives on the battle)
            battle.__dict__.setdefault("_spd_lost", set()).add(battle.get_pokemon(split[2]).name)
        except Exception:
            pass
    turn = battle.__dict__.get("_spd_turn")
    if turn is None or turn["dirty"]:
        return
    if kind == "move":
        ident = split[2][:3] + split[2][4:] if len(split[2]) > 3 and split[2][3] in "ab" else split[2]
        extra = split[5:] if len(split) > 5 else []
        move_id = to_id_str(split[3])
        if any(e.startswith("[from]") for e in extra) or move_id in SKIP_MOVES:
            turn["dirty"] = True
            return
        if turn["snap"] is None:
            turn["snap"] = _snapshot(battle)
            turn["trickroom"] = Field.TRICK_ROOM in battle.fields
        try:
            mon = battle.get_pokemon(split[2])
        except Exception:
            turn["dirty"] = True
            return
        ours = split[2].startswith(battle.player_role or "?")
        role = battle.player_role if ours else battle.opponent_role
        turn["moves"].append({"ident": mon.identifier(role), "ours": ours,
                              "prio": priority(battle, mon, move_id)})
        return
    if kind == "-activate" and len(split) > 3 and any(x in split[3] for x in ("Quick Claw", "Custap", "Quick Draw")):
        turn["dirty"] = True
        return
    if kind in CHANGES and turn["moves"]:
        if kind in ("-boost", "-unboost", "-setboost") and len(split) > 3 and split[3] != "spe":
            return
        turn["dirty"] = True


def estimate(battle: AbstractBattle, mon: Pokemon) -> tuple[float, float] | None:
    """(Speed stat estimate, extra multiplier for an inferred Choice Scarf) for an opposing
    Pokemon: the max-invested neutral guess the features always used, moved inside the
    inferred bounds. None without any bounds."""
    b = battle.__dict__.get("_spd", {}).get(mon.name)
    if b is None or not b.observations or not mon.base_stats or b.base != mon.base_stats.get("spe"):
        return None
    low, high = stat_range(b.base)
    guess = b.base + 52
    scarf = 1.0
    if b.lo > high + 0.5:
        # Faster than any spread could be. A Pokemon that can Mega Evolve may have done so without
        # poke-env picking up the new forme's base Speed (the Champions Mega-Z formes): trust the
        # bound. Otherwise, with no item seen, a Choice Scarf.
        others = _dex().get(mon.species, {}).get("otherFormes", [])
        if any("-Mega" in f for f in others):
            return b.lo, 1.0
        if _item(mon) is not None:
            return None
        scarf, lo = 1.5, b.lo_scarf
    else:
        lo = b.lo
    hi = b.hi
    if lo > hi + 0.5:  # contradictory (something we didn't model): don't trust them
        return None
    return min(max(guess, lo, low), hi, high), scarf


def install() -> None:
    """Patch poke-env's message parsing so every battle feeds `observe`."""
    parse = AbstractBattle.parse_message
    if getattr(parse, "_speed_patch", False):
        return

    def parse_message(self, split_message):
        try:
            observe(self, split_message)
        except Exception:
            self.__dict__["_spd_turn"] = None
        return parse(self, split_message)

    parse_message._speed_patch = True
    AbstractBattle.parse_message = parse_message


install()
