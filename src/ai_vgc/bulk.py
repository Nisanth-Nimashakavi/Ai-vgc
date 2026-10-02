"""Opposing stats for the damage features under closed team sheets, narrowed by the damage seen.

poke-env's damage calculator needs every stat of both Pokemon, and without a team sheet it never
learns the opponent's, so `calc.damage_pct` returned None for every move into or from them: the
damage, KO and effectiveness features were all zeros in closed-sheet play. With `battle._dmg_on`
set (NNPlayer.damage_inference), an opposing Pokemon without stats gets an estimate for the calc:

- candidates: (stat points, nature) spreads from MunchStats usage (`search.spread_prior`), the
  set-guess world's spread first when there is one, plus a few plain archetypes, each with a weight;
- each direct hit between one of ours and one of theirs narrows them: a candidate whose damage
  range (from the same calculator) doesn't contain the HP that was actually lost is made very
  unlikely. Our side's HP is exact; theirs is a percentage, so the check allows rounding. Crits,
  multi-hit and spread moves, anything with "[from]", knockouts, and hits followed by an item or
  ability message are skipped, and a hit that rules every candidate out is ignored (an item or
  ability we can't see explains it).

`estimate` gives the candidate with the most weight. Champions Reg M-C stats at level 50:
HP = base + points + 75, others trunc((base + points + 20) x nature), points 0-32.
"""

from __future__ import annotations

import math
from contextlib import contextmanager

from poke_env.battle import AbstractBattle, Pokemon
from poke_env.data import GenData, to_id_str

STATS = ["hp", "atk", "def", "spa", "spd", "spe"]
ARCHETYPES = [  # (points, nature) for when usage data runs out
    ((32, 0, 0, 0, 0, 32), "Hardy"), ((32, 32, 0, 0, 0, 2), "Adamant"), ((32, 0, 0, 32, 0, 2), "Modest"),
    ((2, 32, 0, 0, 0, 32), "Jolly"), ((2, 0, 0, 32, 0, 32), "Timid"), ((32, 0, 16, 0, 16, 0), "Bold"),
    ((32, 0, 16, 0, 16, 0), "Careful"), ((32, 0, 32, 0, 2, 0), "Impish"), ((32, 0, 0, 0, 32, 2), "Calm"),
]


def _nature_mult(nature: str, stat: str) -> float:
    entry = GenData.from_gen(9).natures.get(to_id_str(nature) or "hardy", {})
    plus, minus = to_id_str(entry.get("plus") or ""), to_id_str(entry.get("minus") or "")
    return 1.1 if plus == stat else 0.9 if minus == stat else 1.0


def stats_for(base: dict[str, int], points: tuple[int, ...], nature: str) -> dict[str, int]:
    out = {"hp": base["hp"] + points[0] + 75}
    for i, s in enumerate(STATS[1:], 1):
        out[s] = math.floor((base[s] + points[i] + 20) * _nature_mult(nature, s))
    return out


def _candidates(battle: AbstractBattle, mon: Pokemon) -> list[list]:
    from ai_vgc.nn.search import _base, _species, spread_prior

    store = battle.__dict__.setdefault("_blk", {})
    base = mon.base_stats
    key = (mon.name, tuple(sorted(base.items())))
    if key in store:
        return store[key]
    cands: dict[tuple, float] = {}
    if battle.__dict__.get("_guess"):
        from ai_vgc.nn.encode import _guesses

        world = _guesses(battle).get(mon.name)
        if world and world[3]:
            cands[(tuple(world[3][0]), world[3][1])] = 1.0
    for (points, nature), p in spread_prior(_base(_species(mon)), 24):
        cands[(tuple(points), nature)] = cands.get((tuple(points), nature), 0.0) + p
    for spread in ARCHETYPES:
        cands.setdefault(spread, 0.01)
    store[key] = [[stats_for(base, pts, nat), w] for (pts, nat), w in cands.items()]
    return store[key]


def estimate(battle: AbstractBattle, mon: Pokemon) -> dict[str, int] | None:
    """Stats to calc with for an opposing Pokemon with none known, or None when inference is off."""
    if not battle.__dict__.get("_dmg_on") or not mon.base_stats:
        return None
    if mon.stats and all(isinstance(v, (int, float)) for v in mon.stats.values()):
        return None
    cands = _candidates(battle, mon)
    return max(cands, key=lambda c: c[1])[0] if cands else None


@contextmanager
def filled(battle: AbstractBattle, *mons: Pokemon):
    """Temporarily give opposing Pokemon their estimated stats (for one damage calc)."""
    saved = []
    try:
        for mon in mons:
            est = estimate(battle, mon) if mon is not None else None
            if est is not None:
                saved.append((mon, mon._stats))
                mon._stats = dict(est)
        yield
    finally:
        for mon, stats in saved:
            mon._stats = stats


def _hp_lost(mon: Pokemon, split: list[str]) -> float | None:
    """HP lost in this -damage line, as a fraction of max HP (ours exact, theirs from a %)."""
    hp = split[3].split()[0] if len(split) > 3 else ""
    if hp == "0":
        return None
    try:
        cur, mx = (float(x) for x in hp.split("/"))
    except ValueError:
        return None
    return mon.current_hp_fraction - cur / mx


def observe(battle: AbstractBattle, split: list[str]) -> None:
    """Feed one protocol message (before poke-env parses it)."""
    if battle.__dict__.get("_spd_frozen") or len(split) < 2:
        return
    kind = split[1]
    if kind == "move":
        extra = split[5:] if len(split) > 5 else []
        battle._blk_hit = None
        if len(split) < 5 or not split[4] or any(e.startswith("[from]") or e == "[spread]" for e in extra):
            return
        battle._blk_hit = {"user": split[2], "move": to_id_str(split[3]), "target": split[4], "clean": True}
        return
    hit = battle.__dict__.get("_blk_hit")
    if hit is None:
        return
    if kind == "turn":
        battle._blk_hit = None
        return
    if kind in ("-crit", "-hitcount", "-activate", "-enditem", "-item", "-ability"):
        hit["clean"] = False
        return
    if kind != "-damage" or len(split) != 4 or split[2] != hit["target"] or not hit["clean"]:
        return
    battle._blk_hit = None
    _infer(battle, hit, split)


def _infer(battle: AbstractBattle, hit: dict, split: list[str]) -> None:
    from poke_env.battle import Move

    from ai_vgc.calc import damage_pct

    role = battle.player_role or "?"
    ours_user = hit["user"].startswith(role)
    if ours_user == hit["target"].startswith(role):
        return  # an ally hit or self-damage
    try:
        user, target = battle.get_pokemon(hit["user"]), battle.get_pokemon(hit["target"])
        move = Move(hit["move"], gen=9)
    except Exception:
        return
    if move.n_hit != (1, 1) or not move.base_power:
        return
    lost = _hp_lost(target, split)
    if lost is None or lost <= 0:
        return
    them = target if ours_user else user
    if them.stats and all(isinstance(v, (int, float)) for v in them.stats.values()):
        return  # their stats are known (open sheets)
    cands = _candidates(battle, them)
    tol = 0.012 if ours_user else 0.003  # their HP comes as a whole percentage
    keep = []
    for cand in cands:
        saved = them._stats
        them._stats = dict(cand[0])
        try:
            d = damage_pct(battle, user, target, move)
        finally:
            them._stats = saved
        if d is None:
            return
        keep.append(d[0] / 100 - tol <= lost <= d[1] / 100 + tol)
    if any(keep) and not all(keep):
        for cand, ok in zip(cands, keep):
            if not ok:
                cand[1] *= 1e-3  # unlikely, not impossible: an unseen item or ability could explain it


def install() -> None:
    parse = AbstractBattle.parse_message
    if getattr(parse, "_bulk_patch", False):
        return

    def parse_message(self, split_message):
        try:
            observe(self, split_message)
        except Exception:
            self.__dict__["_blk_hit"] = None
        return parse(self, split_message)

    parse_message._bulk_patch = True
    AbstractBattle.parse_message = parse_message


install()
