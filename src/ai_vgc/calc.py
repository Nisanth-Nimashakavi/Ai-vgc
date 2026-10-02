"""Damage and speed estimates for the network's input features (`nn/encode.py`)."""

from __future__ import annotations

from poke_env.battle import (
    DoubleBattle,
    Effect,
    Field,
    Move,
    Pokemon,
    SideCondition,
    Status,
    Weather,
)
from poke_env.calc.damage_calc_gen9 import calculate_damage

from ai_vgc import bulk, speed

BOOST = {-6: 2 / 8, -5: 2 / 7, -4: 2 / 6, -3: 2 / 5, -2: 2 / 4, -1: 2 / 3, 0: 1,
         1: 3 / 2, 2: 4 / 2, 3: 5 / 2, 4: 6 / 2, 5: 7 / 2, 6: 8 / 2}
WEATHER_SPEED = {
    "chlorophyll": {Weather.SUNNYDAY, Weather.DESOLATELAND},
    "swiftswim": {Weather.RAINDANCE, Weather.PRIMORDIALSEA},
    "sandrush": {Weather.SANDSTORM},
    "slushrush": {Weather.SNOWSCAPE, Weather.HAIL},
}


def is_ours(battle: DoubleBattle, mon: Pokemon) -> bool:
    return any(mon is m for m in battle.team.values())


def _ident(battle: DoubleBattle, mon: Pokemon) -> str:
    return mon.identifier(battle.player_role if is_ours(battle, mon) else battle.opponent_role)


def damage_pct(
    battle: DoubleBattle, attacker: Pokemon, defender: Pokemon, move: Move
) -> tuple[float, float] | None:
    """Damage range as percent of the defender's max HP, or None if unknown."""
    try:
        # Closed sheets with damage inference on (`bulk.py`): the opponent's stats are an estimate.
        with bulk.filled(battle, attacker, defender):
            lo, hi = calculate_damage(_ident(battle, attacker), _ident(battle, defender), move, battle)
            max_hp = defender.stats["hp"]
    except Exception:
        return None
    if not max_hp:
        return None
    return 100 * lo / max_hp, 100 * hi / max_hp


def effective_speed(battle: DoubleBattle, mon: Pokemon) -> tuple[float, bool]:
    """(speed, exact). Opponent speed falls back to a max-invested estimate."""
    exact = True
    spe = mon.stats.get("spe") if mon.stats else None
    scarf = 1.0
    if not spe:
        # Level 50, 31 IVs, 252 EVs, neutral nature; or, with speed inference on for this battle
        # (`speed.py`), that guess moved inside the bounds the turn order has shown.
        spe, exact = mon.base_stats["spe"] + 52, False
        est = speed.estimate(battle, mon) if battle.__dict__.get("_spd_on") else None
        if est is not None:
            spe, scarf = est
    spe *= BOOST[mon.boosts.get("spe", 0)]
    conditions = battle.side_conditions if is_ours(battle, mon) else battle.opponent_side_conditions
    if SideCondition.TAILWIND in conditions:
        spe *= 2
    if mon.item == "choicescarf":
        spe *= 1.5
    else:
        spe *= scarf
    if mon.status == Status.PAR and mon.ability != "quickfeet":
        spe *= 0.5
    if battle.weather and set(battle.weather) & WEATHER_SPEED.get(mon.ability or "", set()):
        spe *= 2
    if mon.ability == "surgesurfer" and Field.ELECTRIC_TERRAIN in battle.fields:
        spe *= 2
    if Effect.SLOW_START in mon.effects:
        spe *= 0.5
    return spe, exact
