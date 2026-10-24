"""One-turn search at play time (idea 7): try our best joint actions against the opponent's
likely ones in the Showdown simulator and score the results with the value head.

    uv run python -m ai_vgc.nn.player --model data/models/archive/mb-bo3-do-v1.pt --search --n 100
    uv run python -m ai_vgc.nn.player --model data/models/archive/mb-bo3-do-v1.pt --search \\
        --opp-model data/models/all-bo3-bc-v2.pt --n 100

Each decision:

1. Rebuild the position in the simulator from what our bot can see (`position`): both
   teams from the open team sheets (with closed sheets, the opponent's unseen items, abilities
   and moves are guessed from the sets people run, see `worlds`), the opponent's stat points
   and nature from MunchStats usage (`ai_vgc.munchstats`) or the team pool. `--search-worlds
   N` scores each play against N guesses at all of this, weighted by likelihood, then current HP, status, boosts, weather, terrain, Trick
   Room, screens and Tailwind with turns left, Protect counters, first-turn flags (Fake Out)
   and items used up. The opponent's unrevealed back Pokemon are filled in from team preview.
2. Our candidates are the policy's `k` most likely joint actions; the opponent's are the
   `opp_k` most likely by the opponent-action head (a model trained with `--aux`, from
   `opp_model`), or every move/target pair equally likely when there is no such head.
3. `sim_bridge.js` plays each pairing for `seeds` random seeds (the same seeds for every
   pairing, so luck mostly cancels out between them) and returns the turn's log as our client
   would see it. The log is parsed into a copy of the live battle, which is encoded and scored
   by the value head exactly as in real play, so M[i, j] = our mean win chance.
4. We solve the matrix game M using Double Oracle to get equilibrium strategies for both
   players (Nessie-style), instead of just best-responding to the opponent's fixed distribution.
   This eliminates exploitability in simultaneous-move situations (team preview, leads).
   The policy's log-probability is added as a prior to break near-ties toward the policy.

Search only runs on ordinary move turns; forced switches and team preview use the policy.
"""

from __future__ import annotations

import asyncio
import copy
import json
import random
import select
import subprocess
import threading
import time
from itertools import combinations
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from functools import cache
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from poke_env.battle import (
    AbstractBattle,
    DoubleBattle,
    Effect,
    Field,
    Move,
    Pokemon,
    SideCondition,
    Target,
    Weather,
)
from poke_env.data import GenData, to_id_str
from poke_env.environment import DoublesEnv
from poke_env.player import BattleOrder
from poke_env.teambuilder import Teambuilder

from ai_vgc import munchstats
from ai_vgc.nn.encode import encode
from ai_vgc.nn.model import Policy, masked, slot_b_mask
from ai_vgc.nn.player import NNPlayer, action_mask, load_policy, safe_action_to_order
from ai_vgc.nn.rules import apply_rules
from ai_vgc.nn.double_oracle import equilibrium_strategy_from_search
from ai_vgc.nn.vgc_bridge import VGCBridge
from ai_vgc.nn.chance_branching import (
    enumerate_damage_rolls, enumerate_crit, enumerate_secondary_effects,
    enumerate_accuracy, enumerate_protect, ChanceOutcome
)

BRIDGE = Path(__file__).with_name("sim_bridge.js")
STATS = ["hp", "atk", "def", "spa", "spd", "spe"]
VOLATILES = {
    Effect.SUBSTITUTE: "substitute", Effect.TAUNT: "taunt", Effect.CONFUSION: "confusion",
    Effect.LEECH_SEED: "leechseed", Effect.YAWN: "yawn", Effect.PERISH1: "perish1",
    Effect.PERISH2: "perish2", Effect.PERISH3: "perish3", Effect.SALT_CURE: "saltcure",
    Effect.HEAL_BLOCK: "healblock", Effect.CHARGE: "charge",
}
# Turns a condition lasts when set (the item/ability extensions aren't visible, so ignored).
SIDE_TURNS = {SideCondition.TAILWIND: 4, SideCondition.REFLECT: 5, SideCondition.LIGHT_SCREEN: 5,
              SideCondition.AURORA_VEIL: 5, SideCondition.SAFEGUARD: 5, SideCondition.MIST: 5}
PSEUDO = {Field.TRICK_ROOM: "trickroom", Field.GRAVITY: "gravity", Field.WONDER_ROOM: "wonderroom",
          Field.MAGIC_ROOM: "magicroom"}
TERRAINS = {Field.ELECTRIC_TERRAIN, Field.GRASSY_TERRAIN, Field.MISTY_TERRAIN, Field.PSYCHIC_TERRAIN}
TARGETED = {Target.NORMAL, Target.ANY, Target.ADJACENT_FOE, Target.ADJACENT_ALLY,
            Target.ADJACENT_ALLY_OR_SELF}
IGNORE = {"", "t:", "request", "win", "tie", "error", "bigerror", "inactive", "inactiveoff"}


def _pokedex() -> dict:
    return GenData.from_gen(9).pokedex


def _base(species: str) -> str:
    """Species id without a Mega forme (the team pool lists the base species)."""
    entry = _pokedex().get(species, {})
    if str(entry.get("forme", "")).startswith("Mega"):
        return to_id_str(entry["baseSpecies"])
    return species


def _species(mon: Pokemon) -> str:
    """The current forme. poke-env keeps an opposing Mega's species as the base one and only
    swaps its base stats, so the forme is found by those."""
    dex = _pokedex()
    if dex.get(mon.species, {}).get("baseStats") == mon.base_stats:
        return mon.species
    for forme in dex.get(mon.species, {}).get("otherFormes", []):
        if dex.get(to_id_str(forme), {}).get("baseStats") == mon.base_stats:
            return to_id_str(forme)
    return mon.species


TEAM_MASS = 0.8  # share of the guesses given to matching whole teams (0: per-species guesses only)
USAGE_DIR = Path("data/teams")  # set_usage_<format>.json, built by `python -m ai_vgc.nn.search --usage`


@cache
def set_counts(teams_dir: str, fmt: str = "") -> dict[str, Counter]:
    """How often each (item, ability, moves) set is run, per base species: every set in the team
    pool counts once, plus the usage table from open team sheets in human logs when there is one
    (`USAGE_DIR/set_usage_<fmt without bo3>.json`)."""
    out: dict[str, Counter] = {}
    for f in sorted(Path(teams_dir).glob("*.txt")):
        for m in Teambuilder.parse_showdown_team(f.read_text()):
            out.setdefault(to_id_str(m.species or m.nickname), Counter())[
                (to_id_str(m.item or ""), to_id_str(m.ability or ""), frozenset(to_id_str(x) for x in m.moves))] += 1
    usage = USAGE_DIR / f"set_usage_{fmt.removesuffix('bo3')}.json"
    if fmt and usage.exists():
        for base, rows in json.loads(usage.read_text()).items():
            for item, ability, moves, n in rows:
                out.setdefault(base, Counter())[(item, ability, frozenset(moves))] += n
    return out


def set_posterior(base: str, item: str, ability: str, moves: list[str], teams_dir: str,
                  fmt: str = "") -> list[tuple[tuple[str, str, list[str]], float]]:
    """Closed team sheets: the sets this Pokemon may run, given what it has shown (revealed moves
    are a subset, a shown item or ability matches), weighted by how often each is run
    (`weighted_sets`: whole sets from team sheets, matched to MunchStats' usage). Unshown
    parts are filled in; revealed moves keep their order and come first. When nothing in the
    usage agrees, the closest set (most revealed moves in common) stands in."""
    if item and ability and len(moves) >= 4:
        return [((item, ability, moves), 1.0)]
    counts = weighted_sets(base, teams_dir, fmt)
    if not counts:
        return [((item, ability, moves), 1.0)]
    seen = set(moves)
    fits = {st: n for st, n in counts.items()
            if seen <= st[2] and (not item or st[0] == item) and (not ability or st[1] == ability)}
    if not fits:
        best = max(counts, key=lambda st: (len(st[2] & seen), (not item) or st[0] == item,
                                           (not ability) or st[1] == ability, counts[st]))
        fits = {best: 1}
    merged: Counter = Counter()
    for st, n in fits.items():
        merged[(item or st[0], ability or st[1], tuple(moves + sorted(st[2] - seen))[:4])] += n
    total = sum(merged.values())
    return [((i, a, list(m)), n / total) for (i, a, m), n in merged.most_common()]


_WEIGHTED: dict[tuple, Counter] = {}


def weighted_sets(base: str, teams_dir: str, fmt: str = "", rounds: int = 20) -> Counter:
    """Set weights for a species: whole sets (item, ability, moves) from the team pool and the
    open-sheet logs, which show what goes together, rescaled so that each item's, ability's and
    move's share matches MunchStats' current usage (iterative proportional fitting; e.g.
    Basculegion ends up Adaptability 94% / Swift Swim 6% whatever the logs said). Sets built from
    MunchStats alone (`munch_sets`) are mixed in with a little weight, so usage the logs never
    showed can get its share. Without MunchStats data for the species, the logs' counts as they are."""
    munch = _munch().get(base)
    key = (base, teams_dir, fmt, _MUNCH[0])
    if key in _WEIGHTED:
        return _WEIGHTED[key]
    logged = set_counts(teams_dir, fmt).get(base, Counter())
    extra = munch_sets(base)
    if not munch or not munch.get("moves"):
        out = Counter(logged) or extra
        _WEIGHTED[key] = out
        return out
    w: Counter = Counter()
    total = sum(logged.values())
    for st, n in logged.items():
        w[st] += n / total if total else 0
    etotal = sum(extra.values())
    for st, n in extra.items():  # 10% of the mass (all of it when the logs have nothing)
        w[st] += (0.1 if total else 1.0) * n / etotal
    items = {to_id_str(i): p / 100 for i, p in munch.get("items", [])}
    abilities = {to_id_str(a): p / 100 for a, p in munch.get("abilities", [])}
    moves = {to_id_str(m): p / 100 for m, p in munch.get("moves", [])}

    def fit(target: dict[str, float], part) -> None:
        # Scale each group of sets (by item, or ability) to its target share. Values the site
        # doesn't list (its top 10 miss rare ones) keep their share of what is left over.
        cur: Counter = Counter()
        for st, n in w.items():
            cur[part(st)] += n
        mass = sum(cur.values())
        rest = max(0.0, 1 - sum(target.values()))
        other = sum(n for v, n in cur.items() if v not in target) / mass
        for st in w:
            v = part(st)
            share = cur[v] / mass
            want = target[v] if v in target else (rest * share / other if other else 0)
            w[st] *= want / share if share else 0

    blend = dict(w)  # the logged+MunchStats mixture, before marginal fitting
    for _ in range(rounds):
        if items:
            fit(items, lambda st: st[0])
        if abilities:
            fit(abilities, lambda st: st[1])
        for m, p in list(moves.items()):  # a move's share = sets that carry it
            mass = sum(w.values())
            cur = sum(n for st, n in w.items() if m in st[2]) / mass
            if 0 < cur < 1:
                for st in w:
                    w[st] *= p / cur if m in st[2] else (1 - p) / (1 - cur)
    out = Counter({st: n for st, n in w.items() if n > 1e-6})
    if not out:  # iterative fitting underflowed every set; keep the unfitted blend
        out = Counter({st: n for st, n in blend.items() if n > 1e-6})
    _WEIGHTED[key] = out
    return out


_MUNCH: tuple[float, dict] = (0.0, {})


def _munch() -> dict:
    """MunchStats cache (`ai_vgc.munchstats`), re-read whenever the file changes, since a
    running bot's refresher keeps adding to it."""
    global _MUNCH
    try:
        mtime = munchstats.CACHE.stat().st_mtime
    except OSError:
        return {}
    if mtime != _MUNCH[0]:
        _MUNCH = (mtime, munchstats.load())
    return _MUNCH[1]


def spread_prior(base: str, top: int = 12) -> list[tuple[tuple[tuple[int, ...], str], float]]:
    """The `top` most likely (stat points, nature) for a species from MunchStats' usage, taking
    spread and nature as independent (the site lists them separately); [] without data."""
    entry = _munch().get(base)
    if not entry or not entry.get("spreads"):
        return []
    natures = entry.get("natures") or [["Hardy", 100.0]]
    combos = [((tuple(int(x) for x in sp.split("/")), nat), ps * pn)
              for sp, ps in entry["spreads"] for nat, pn in natures]
    combos = sorted(combos, key=lambda c: -c[1])[:top]
    total = sum(p for _, p in combos)
    return [(c, p / total) for c, p in combos]


def munch_sets(base: str, top: int = 24) -> Counter:
    """Sets built from MunchStats' separate move, item and ability usage, for a species no team
    or log shows whole: the likeliest items and abilities with the four likeliest moves, or one of
    those swapped for any lower-ranked move the site lists. Weights are products of the usages.
    Every move MunchStats reports (its top ~10) can therefore land in some set; otherwise moves
    ranked past the swap window are unrepresentable in any sampled world no matter the width."""
    entry = _munch().get(base)
    if not entry or not entry.get("moves"):
        return Counter()
    moves = [(to_id_str(m), p / 100) for m, p in entry["moves"]]
    items = [(to_id_str(i), p) for i, p in entry.get("items", [])[:4]] or [("", 1.0)]
    abilities = [(to_id_str(a), p) for a, p in entry.get("abilities", [])[:2]] or [("", 1.0)]
    core, extra = moves[:4], moves[4:]
    movesets = [(frozenset(m for m, _ in core), float(np.prod([p for _, p in core])))]
    for i in range(len(core)):
        for m, p in extra:
            swapped = core[:i] + core[i + 1:] + [(m, p)]
            movesets.append((frozenset(x for x, _ in swapped),
                             float(np.prod([q for _, q in swapped])) * (1 - core[i][1]) / max(core[i][1], 1e-3)))
    out = Counter({(i, a, ms): pi * pa * pm for i, pi in items for a, pa in abilities for ms, pm in movesets})
    return Counter(dict(out.most_common(top)))


COSMETIC = ("Four", "Three", "Masterpiece", "Artisan", "Antique")  # formes that play the same


def _core(species: str) -> str:
    """Species id for matching teams: a Mega counts as the forme it Mega Evolves from (Floette-Mega
    as Floette-Eternal) and cosmetic formes as the base species, since sheets and team files write
    them either way."""
    entry = _pokedex().get(species, {})
    forme = str(entry.get("forme", ""))
    if forme.startswith("Mega"):
        return to_id_str(entry.get("battleOnly") if isinstance(entry.get("battleOnly"), str) else entry["baseSpecies"])
    if forme in COSMETIC:
        return to_id_str(entry["baseSpecies"])
    return species


@cache
def team_library(teams_dir: str, fmt: str = "") -> dict[frozenset, list[tuple[dict, float]]]:
    """Whole teams, indexed by their six species and by every five of them: {key: [(team, weight)]},
    a team being {species: (item, ability, moves, spread)}. Team files in the pool count once and
    carry their stat points; teams from human open sheets (`USAGE_DIR/team_library_<fmt>.json`,
    built by scripts/team_library.py) count as often as they were seen, with their nature but no
    stat points (spread None, or MunchStats' likeliest spread for that nature)."""
    teams: list[tuple[dict, float]] = []
    core, spread_for = cache(_core), {}
    for f in sorted(Path(teams_dir).glob("*.txt")):
        mons = Teambuilder.parse_showdown_team(f.read_text())
        team = {core(to_id_str(m.species or m.nickname)):
                (to_id_str(m.item or ""), to_id_str(m.ability or ""), frozenset(to_id_str(x) for x in m.moves),
                 (tuple(m.evs or [0] * 6), m.nature or "Hardy")) for m in mons}
        if len(team) == 6:
            teams.append((team, 1.0))
    path = USAGE_DIR / f"team_library_{fmt.removesuffix('bo3')}.json"
    if fmt and path.exists():
        for rows, n in json.loads(path.read_text()):
            team = {}
            for sp, item, ability, moves, nature in rows:
                base = core(sp)
                if (base, nature) not in spread_for:
                    spread_for[base, nature] = next((c for c, _ in spread_prior(base, 50) if c[1] == nature),
                                                    None) if nature else None
                team[base] = (item, ability, frozenset(moves), spread_for[base, nature])
            if len(team) == 6:
                teams.append((team, float(n)))
    index: dict[frozenset, list[tuple[dict, float]]] = {}
    for team, w in teams:
        key = frozenset(team)
        index.setdefault(key, []).append((team, w))
        for sp in team:
            index.setdefault(key - {sp}, []).append((team, w))
    return index


def search_opponent_roster(battle: DoubleBattle) -> list[Pokemon]:
    """Merge public preview species with subsequently revealed live objects."""
    team = list(battle.opponent_team.values())
    seen_species = {_core(_species(m)) for m in team}
    return team + [m for m in battle.teampreview_opponent_team if _core(_species(m)) not in seen_species]


def team_worlds(battle: DoubleBattle, teams_dir: str, n: int, fmt: str = "", include_preview=False) -> list[tuple[dict, float]]:
    """Guesses at the opponent's hidden sets from whole known teams: teams with the same six
    species as their team preview (or five of the six, at a quarter of the weight) whose sets
    agree with everything revealed so far (moves, item, ability). Each guess is {name: (item,
    ability, moves, spread)} for the Pokemon the team covers, with its share of the matching
    weight; up to `n`, most likely first. [] when no known team fits."""
    preview = {_core(to_id_str(m.species)) for m in battle.teampreview_opponent_team}
    if len(preview) != 6:
        return []
    lib = team_library(teams_dir, fmt)
    revealed = {}
    for mon in search_opponent_roster(battle) if include_preview else battle.opponent_team.values():
        species = _species(mon)
        abilities = _pokedex()[species]["abilities"]
        ability = to_id_str(abilities["0"]) if len(abilities) == 1 else mon.ability or ""
        revealed[_core(species)] = (mon.name, _item(mon), ability, list(mon.moves)[:4])
    cands: dict[tuple, float] = {}
    keys = [(frozenset(preview), 1.0)] + [(frozenset(preview - {sp}), 0.25) for sp in preview]
    for key, scale in keys:
        for team, w in lib.get(key, []):
            if set(team) != preview and len(set(team) & preview) < 5:
                continue
            ok = True
            for sp, (_, item, ability, moves) in revealed.items():
                if sp not in team:
                    continue
                t_item, t_ability, t_moves, _ = team[sp]
                if (item and item != t_item) or (ability and ability != t_ability) or not set(moves) <= t_moves:
                    ok = False
                    break
            if ok:
                sig = tuple(sorted((sp, *team[sp][:2], tuple(sorted(team[sp][2])), team[sp][3]) for sp in team))
                cands[sig] = cands.get(sig, 0.0) + w * (1.0 if set(team) == preview else scale)
    if not cands:
        return []
    out = []
    total = sum(cands.values())
    for sig, w in sorted(cands.items(), key=lambda c: -c[1])[:n]:
        world = {}
        for sp, item, ability, moves, spread in sig:
            if sp in revealed:
                name, _, _, seen = revealed[sp]
                world[name] = (item, ability, (seen + sorted(set(moves) - set(seen)))[:4], spread)
        out.append((world, w / total))
    return out


def worlds(battle: DoubleBattle, teams_dir: str, n: int, fmt: str = "", *, include_preview=False, rng=None) -> list[tuple[dict, float]]:
    """Up to `n` guesses at the opponent's hidden sets, {name: (item, ability, moves, spread)},
    each with its probability (the product over their Pokemon, renormalised over the ones kept).
    Sets come from `set_posterior`; spreads (stat points and nature, which no team sheet shows)
    from `spread_prior`, or None to use the team pool's. The most likely of both for every
    Pokemon comes first; the rest are sampled. When whole known teams match their team preview
    (`team_worlds`), those guesses take `TEAM_MASS` of the probability."""
    known = team_worlds(battle, teams_dir, n, fmt, include_preview) if TEAM_MASS else []
    if known:
        mass, kt = TEAM_MASS, sum(p for _, p in known)
        rest = worlds_by_species(battle, teams_dir, max(1, n - len(known)), fmt, include_preview)
        # Pokemon a team guess doesn't cover (5-of-6 matches) take the species guess's set.
        filler = rest[0][0]
        merged = [({**filler, **w}, mass * p / kt) for w, p in known] + [(w, (1 - mass) * p) for w, p in rest]
        merged = sorted(merged, key=lambda c: -c[1])[:n]
        total = sum(p for _, p in merged)
        sampled = [(w, p / total) for w, p in merged]
    else:
        sampled = worlds_by_species(battle, teams_dir, n, fmt, include_preview)
    return bring_four_worlds(battle, sampled, n, rng) if include_preview else sampled


def bring_four_worlds(battle, sampled, n, rng=None):
    """Sample the hidden bring-four too, always retaining revealed Pokémon.

    No bring-selection model is available: use a uniform prior over legal unseen
    subsets. Monte Carlo samples have equal weight (not posterior weight twice).
    """
    roster = search_opponent_roster(battle)
    seen = [_core(_species(m)) for m in roster if m.revealed or m.active or m.fainted]
    unseen = [_core(_species(m)) for m in roster if _core(_species(m)) not in seen]
    subsets = list(combinations(unseen, min(len(unseen), max(0, 4-len(seen)))))
    if len(subsets) == 1:
        return [(dict(w, __bring__=seen + list(subsets[0])), p) for w, p in sampled]
    rng = rng if rng is not None else np.random.default_rng()
    probs = np.asarray([p for _, p in sampled], dtype=float)
    probs /= probs.sum()
    return [(dict(sampled[int(rng.choice(len(sampled), p=probs))][0],
                  __bring__=seen + list(subsets[int(rng.integers(len(subsets)))])), 1/n)
            for _ in range(n)]


def worlds_by_species(battle: DoubleBattle, teams_dir: str, n: int, fmt: str = "", include_preview=False) -> list[tuple[dict, float]]:
    """`worlds` from each Pokemon's own set posterior (no team-level match)."""
    post = {}
    for mon in search_opponent_roster(battle) if include_preview else battle.opponent_team.values():
        species = _species(mon)
        abilities = _pokedex()[species]["abilities"]
        ability = to_id_str(abilities["0"]) if len(abilities) == 1 else mon.ability or ""
        sets = set_posterior(_base(species), _item(mon), ability, list(mon.moves)[:4], teams_dir, fmt)
        spreads = spread_prior(_base(species)) or [(None, 1.0)]
        post[mon.name] = [((*st, sp), ps * pp) for st, ps in sets for sp, pp in spreads]
        post[mon.name].sort(key=lambda c: -c[1])
    if not post:
        return [({}, 1.0)]
    names = list(post)
    # A world is one index into each Pokemon's posterior; index 0 is its most likely set.
    found = {(0,) * len(names): None}
    rng = np.random.default_rng()
    for _ in range(8 * n):
        if len(found) >= n:
            break
        found.setdefault(tuple(int(rng.choice(len(post[nm]), p=[p for _, p in post[nm]])) for nm in names), None)
    picks = list(found)[:n]
    ps = np.array([np.prod([post[nm][i][1] for nm, i in zip(names, pick)]) for pick in picks])
    return [({nm: post[nm][i][0] for nm, i in zip(names, pick)}, p) for pick, p in zip(picks, ps / ps.sum())]


@cache
def team_pool(teams_dir: str) -> tuple[dict, dict]:
    """(exact sets keyed by (species, item, ability, moves), most common spread per species)
    from every team file in the pool."""
    exact, spreads = {}, {}
    for f in sorted(Path(teams_dir).glob("*.txt")):
        for m in Teambuilder.parse_showdown_team(f.read_text()):
            sp = to_id_str(m.species or m.nickname)
            spread = (tuple(m.evs or [0] * 6), m.nature or "Hardy", tuple(m.ivs or [31] * 6))
            key = (sp, to_id_str(m.item or ""), to_id_str(m.ability or ""),
                   frozenset(to_id_str(x) for x in m.moves))
            exact.setdefault(key, spread)
            spreads.setdefault(sp, Counter())[spread] += 1
    return exact, {sp: c.most_common(1)[0][0] for sp, c in spreads.items()}


def _item(mon: Pokemon) -> str:
    return "" if mon.item in (None, "unknown_item") else mon.item


def _set(mon: Pokemon, teams_dir: str, world: dict | None = None) -> dict:
    """Showdown set for a Pokemon as the battle and open team sheets show it. Spread: the world's,
    else the team pool's when the set matches a pool team exactly, else MunchStats' most used,
    else the pool's most common for the species. With closed
    sheets the unseen item, ability and moves come from `world` (see `worlds`), or else the most
    likely set (`set_posterior`)."""
    species = _species(mon)
    base = _base(species)
    abilities = _pokedex()[species]["abilities"]
    ability = to_id_str(abilities["0"]) if len(abilities) == 1 else mon.ability or ""
    moves = list(mon.moves)[:4]
    item = _item(mon)
    spread = None
    if world is not None and mon.name in world:
        item, ability, moves, spread = world[mon.name]
    elif len(moves) < 4 or not item or not ability:  # closed sheets: not all of it seen yet
        item, ability, moves = set_posterior(base, item, ability, moves, teams_dir)[0][0]
    exact, spreads = team_pool(teams_dir)
    if spread is None and (base, item, ability, frozenset(moves)) not in exact and spread_prior(base):
        spread = spread_prior(base)[0][0]  # no exact team match: the most used spread
    if spread is not None:
        evs, nature, ivs = spread[0], spread[1], (31,) * 6
    else:
        evs, nature, ivs = exact.get((base, item, ability, frozenset(moves)),
                                     spreads.get(base, ((0,) * 6, "Hardy", (31,) * 6)))
    return {"name": mon.name, "species": species, "item": item, "ability": ability,
            "moves": moves, "nature": nature, "level": mon.level or 50,
            "evs": dict(zip(STATS, evs)), "ivs": dict(zip(STATS, ivs))}


def _mon_state(mon: Pokemon, ours: bool, teams_dir: str, world: dict | None = None) -> dict:
    s = {"set": _set(mon, teams_dir, None if ours else world), "fainted": mon.fainted, "firstTurn": mon.first_turn,
         "protect": mon.protect_counter, "boosts": dict(mon.boosts), "item": _item(mon),
         "status": mon.status.name.lower() if mon.status and not mon.fainted else "",
         "statusTurns": mon.status_counter,
         "volatiles": [v for e, v in VOLATILES.items() if e in mon.effects]}
    # UNKNOWN_ITEM is different from None/empty: the latter is publicly gone.
    s["itemKnown"] = mon.item != "unknown_item"
    s["item"] = _item(mon) if s["itemKnown"] else s["set"].get("item", "")
    s["volatileElapsed"] = {v: int(mon.effects[e]) for e, v in VOLATILES.items() if e in mon.effects}
    s["pp"] = {mid: int(move.current_pp) for mid, move in mon.moves.items()}
    s["lastMove"] = mon.last_move.id if mon.last_move else None
    s["mustRecharge"] = mon.must_recharge
    s["chargingMove"] = mon.preparing_move.id if mon.preparing_move else None
    if ours:
        s["hp"], s["maxhp"] = mon.current_hp, mon.max_hp
        s["stats"] = {k: v for k, v in (mon.stats or {}).items() if v}
    else:
        # Not seen yet (max HP 0): full HP.
        s["hpFrac"] = mon.current_hp_fraction if mon.max_hp else 1.0
    return s


def _side(battle: DoubleBattle, ours: bool, teams_dir: str, world: dict | None = None) -> dict | None:
    """The side's brought Pokemon, actives first in slot order. None when a slot can't be
    filled (it is empty and nothing has fainted to stand in for it)."""
    team = battle.team if ours else battle.opponent_team
    if not ours and world is not None and "__bring__" in world:
        team = {m.name: m for m in search_opponent_roster(battle)
                if _core(_species(m)) in world["__bring__"]}
    active = battle.active_pokemon if ours else battle.opponent_active_pokemon
    # With open team sheets poke-env holds all six opposing Pokemon from the start, so the
    # ones not yet seen fill the opponent's four in team order.
    if ours:
        brought = [m for m in team.values() if m.selected_in_teampreview or m.active]
    else:
        seen = [m for m in team.values() if m.revealed or m.active or m.fainted]
        brought = seen + [m for m in team.values() if m not in seen][:max(0, 4 - len(seen))]
    fainted = [m for m in brought if m.fainted and m not in active]
    order = []
    for m in active:
        if m is None:
            if not fainted:
                return None
            m = fainted.pop()
        order.append(m)
    order += [m for m in brought if m not in order]
    conds = battle.side_conditions if ours else battle.opponent_side_conditions
    return {
        "mons": [_mon_state(m, ours, teams_dir, world) for m in order],
        "conditions": {c.name.lower().replace("_", ""): max(1, SIDE_TURNS[c] - (battle.turn - t))
                       if c in SIDE_TURNS else 0 for c, t in conds.items()},
        "megaUsed": battle.used_mega_evolve if ours else battle.opponent_used_mega_evolve,
    }


def position(battle: DoubleBattle, fmt: str, teams_dir: str, world: dict | None = None) -> dict | None:
    """The bridge's `state` for this battle (the opponent's hidden sets from `world`), or None when
    it can't be rebuilt."""
    us, them = _side(battle, True, teams_dir), _side(battle, False, teams_dir, world)
    if us is None or them is None:
        return None
    st = {"format": fmt, "turn": battle.turn, "role": battle.player_role,
          "sides": {"us": us, "them": them}, "pseudo": {}}
    for w, t in battle.weather.items():
        if w not in (Weather.UNKNOWN,):
            st["weather"], st["weatherTurns"] = w.name.lower(), max(1, 5 - (battle.turn - t))
    for f, t in battle.fields.items():
        left = max(1, 5 - (battle.turn - t))
        if f in TERRAINS:
            st["terrain"], st["terrainTurns"] = f.name.lower().replace("_", ""), left
        elif f in PSEUDO:
            st["pseudo"][PSEUDO[f]] = left
    return st


class _Worker:
    """One `sim_bridge.js` process. Not thread-safe: the caller holds the Bridge's lock."""

    def __init__(self, showdown: str | Path, timeout: float):
        self.showdown, self.timeout = showdown, timeout
        self._start()

    def _start(self) -> None:
        self.proc = subprocess.Popen(["node", str(BRIDGE), str(self.showdown)], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, text=True, bufsize=1)

    def ask(self, req: dict) -> dict:
        # A bridge that died (e.g. killed on a busy node) is restarted rather than failing
        # every later search in the run.
        if self.proc.poll() is not None:
            self._start()
        assert self.proc.stdin and self.proc.stdout
        self.proc.stdin.write(json.dumps(req) + "\n")
        # The read blocks the event loop and every battle on it, so a simulation that never
        # returns (it happens on ilab) must not wait forever: kill it and start a fresh one.
        if not select.select([self.proc.stdout], [], [], self.timeout)[0]:
            self.proc.kill()
            self.proc.wait()
            raise RuntimeError(f"sim bridge took over {self.timeout:.0f}s; restarted")
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError(f"sim bridge exited with code {self.proc.wait()}")
        out = json.loads(line)
        if "error" in out:
            raise RuntimeError(out["error"])
        return out


class Bridge:
    """`sim_bridge.js` processes for one Showdown checkout. Turn-1 playouts without `keep` are split
    across `procs` processes (each simulation is independent, so more seeds cost wall time only
    until the cores run out); everything else, including kept positions, uses the first one."""

    def __init__(self, showdown: str | Path = "pokemon-showdown", timeout: float = 60.0, procs: int = 1):
        self.showdown = showdown
        self.timeout = timeout
        # Searches run in worker threads (SearchPlayer), several battles at once: without this, two
        # requests interleave on the one pipe, replies get mixed up (JSONDecodeError) and a timeout
        # in one thread kills the process under the other.
        self.lock = threading.RLock()  # re-entrant: a two-turn search holds it across its requests
        self.workers = [_Worker(showdown, timeout) for _ in range(max(1, procs))]
        self._threads = ThreadPoolExecutor(len(self.workers)) if len(self.workers) > 1 else None

    @property
    def proc(self) -> subprocess.Popen:
        return self.workers[0].proc

    def kill(self) -> None:
        """Kill every process; each restarts on its next request."""
        with self.lock:
            for w in self.workers:
                w.proc.kill()
                w.proc.wait()

    def _ask(self, req: dict) -> dict:
        with self.lock:
            return self.workers[0].ask(req)

    def run(self, state: dict, pairs: list[tuple[str, str]], seeds: list[int], keep: bool = False) -> list[dict]:
        """Play each pair once per seed. With `keep`, results that end on an ordinary move turn
        carry "next" ({"id", "request"}) and the bridge keeps that position for `cont`."""
        if keep or self._threads is None or len(pairs) < 2:
            return self._ask({"state": state, "pairs": pairs, "seeds": seeds, "keep": keep})["results"]
        # Results are pair-major, so contiguous chunks of pairs concatenate back in order.
        n = min(len(self.workers), len(pairs))
        size, extra = divmod(len(pairs), n)
        chunks, i = [], 0
        for j in range(n):
            chunks.append(pairs[i:i + size + (j < extra)])
            i += size + (j < extra)
        with self.lock:
            outs = list(self._threads.map(
                lambda wc: wc[0].ask({"state": state, "pairs": wc[1], "seeds": seeds, "keep": False}),
                zip(self.workers, chunks)))
        return [r for out in outs for r in out["results"]]

    def cont(self, conts: list[dict]) -> list[list[dict]]:
        """Turn 2 from positions kept by the last `run(..., keep=True)`: each item is
        {"id", "pairs", "seeds"}. Hold `self.lock` across both calls, or another search's
        `run` can replace the kept positions in between."""
        return self._ask({"cont": conts})["conts"]

    def dump(self, state: dict) -> list:
        """The position as the simulator rebuilt it, for checking against poke-env's."""
        return self._ask({"state": state, "dump": True})["dump"]

    def close(self) -> None:
        for w in self.workers:
            w.proc.terminate()


def after(battle: DoubleBattle, lines: list[str]) -> DoubleBattle:
    """A copy of the battle with a simulated turn's log parsed into it."""
    logger, battle.logger = battle.logger, None
    try:
        b = copy.deepcopy(battle)
    finally:
        battle.logger = logger
    b.logger = logger
    b._spd_frozen = True  # a simulated turn says nothing about their real Speed
    for line in lines:
        split = line.split("|")
        if len(split) > 1 and split[1] not in IGNORE:
            b.parse_message(split)
    return b


@torch.no_grad()
def our_candidates(model: Policy, obs: dict[str, np.ndarray], mask: np.ndarray, k: int,
                   ka: int = 4) -> list[tuple[np.ndarray, float]]:
    """The policy's k most likely joint actions (slot a from its top `ka`), with log-probs."""
    return our_candidates_batch(model, [obs], [mask], k, ka)[0]


@torch.no_grad()
def our_candidates_batch(model: Policy, obs_list: list[dict[str, np.ndarray]], mask_list: list[np.ndarray],
                         k: int, ka: int = 4) -> list[list[tuple[np.ndarray, float]]]:
    """`our_candidates` for N states in one batched encode/slot-0 call instead of N separate ones
    (the transformer encode is the expensive part; slot b's small head is still run per a0 candidate,
    exactly as in `our_candidates`, since that's cheap and a0 differs per state)."""
    n = len(obs_list)
    if n == 0:
        return []
    b = {key: torch.from_numpy(np.stack([np.ascontiguousarray(o[key]) for o in obs_list])) for key in obs_list[0]}
    m = torch.from_numpy(np.stack(mask_list))
    enc = model.encode(b)
    lp_a_all = F.log_softmax(masked(model.slot_logits(enc, b, 0), m[:, 0]), -1)
    cls, toks, mv = enc
    out: list[list[tuple[np.ndarray, float]]] = []
    for i in range(n):
        enc_i = (cls[i:i + 1], toks[i:i + 1], mv[i:i + 1])
        b_i = {key: v[i:i + 1] for key, v in b.items()}
        lp_a = lp_a_all[i]
        cands = []
        for a0 in lp_a.topk(min(ka, int(mask_list[i][0].sum()))).indices.tolist():
            prev = torch.tensor([a0])
            lp_b = F.log_softmax(masked(model.slot_logits(enc_i, b_i, 1, prev),
                                        slot_b_mask(m[i:i + 1, 1], prev)), -1)[0]
            for a1 in lp_b.topk(min(k, int(slot_b_mask(m[i:i + 1, 1], prev).sum()))).indices.tolist():
                cands.append((np.array([a0, a1]), (lp_a[a0] + lp_b[a1]).item()))
        out.append(sorted(cands, key=lambda c: -c[1])[:k])
    return out


def _slot_options(battle: DoubleBattle, state: dict, slot: int,
                  probs: tuple[np.ndarray, np.ndarray] | None) -> list[tuple[str, float]]:
    """(choice, probability) for one opposing slot, from the opponent head's (move-or-switch,
    target) probabilities, or every move and target equally likely without them."""
    mon = battle.opponent_active_pokemon[slot]
    if mon is None or mon.fainted:
        return [("pass", 1.0)]
    # The moves of this world's set: the ones seen (in the order the head scores them) first,
    # then the guessed ones, which the head never saw; they get the seen moves' mean chance.
    known = len(mon.moves)
    moves = [Move(m, gen=9) for m in state["sides"]["them"]["mons"][slot]["set"]["moves"]][:4]
    if probs is not None and len(moves) > known:
        mv_p, tg_p = probs[0].copy(), probs[1].copy()
        guess = mv_p[:known].mean() if known else 1.0
        for i in range(known, len(moves)):
            ally = moves[i].target in (Target.ADJACENT_ALLY, Target.ADJACENT_ALLY_OR_SELF)
            mv_p[i], tg_p[i] = guess, np.array([0.0, 0.0, 1.0] if ally else [0.5, 0.5, 0.0])
        probs = (mv_p, tg_p)
    # Their view: our a / our b are foe slots 1 / 2; the ally is -1 (slot a) or -2 (slot b).
    targets = [1, 2, -2 if slot == 0 else -1]
    out: dict[str, float] = {}
    for i, mv in enumerate(moves):
        p_move = probs[0][i] if probs is not None else 1.0
        if mv.target in TARGETED:
            tgt = range(3) if probs is not None else range(2)
            for t in tgt:
                p = p_move * (probs[1][i][t] if probs is not None else 1.0)
                out[f"move {mv.id} {targets[t]}"] = out.get(f"move {mv.id} {targets[t]}", 0) + p
        else:
            out[f"move {mv.id}"] = out.get(f"move {mv.id}", 0) + p_move
    bench = [m["set"]["name"] for m in state["sides"]["them"]["mons"][2:] if not m["fainted"]]
    if bench:
        # The head predicts switching, not a particular bench destination.
        # Preserve that total mass while exposing every healthy destination.
        switch_mass = float(probs[0][4]) if probs is not None else 1.0
        for name in bench:
            out[f"switch {name}"] = switch_mass / len(bench)
    total = sum(out.values()) or 1.0
    return [(c, p / total) for c, p in out.items()]


@torch.no_grad()
def opp_candidates(opp_model: Policy | None, battle: DoubleBattle, obs: dict[str, np.ndarray],
                   state: dict, k: int = 6, *, ensure_switches: bool = True) -> list[tuple[str, float]]:
    """The opponent's k most likely joint choices (in their side's choice syntax), with
    probabilities renormalised over the candidates. Guarantees top-6 minimum and full bench
    switch recall when ensure_switches is True."""
    probs = [None, None]
    if opp_model is not None and opp_model.config.get("aux"):
        b = {key: torch.from_numpy(np.ascontiguousarray(v))[None] for key, v in obs.items()}
        mv, tg = opp_model.opp_logits(opp_model.encode(b), b)
        mv, tg = F.softmax(mv[0], -1).numpy(), F.softmax(tg[0], -1).numpy()
        probs = [(mv[0], tg[0]), (mv[1], tg[1])]
    probs = _opp_probs_batch(opp_model, [obs]) if opp_model is not None and opp_model.config.get("aux") else [probs]
    return _opp_candidates_from_probs(battle, obs, state, k, probs[0], ensure_switches=ensure_switches)


def _opp_probs_batch(opp_model: Policy, obs_list: list[dict[str, np.ndarray]]) -> list[list[tuple | None]]:
    """Batched `opp_model.encode`+`opp_logits` call across N states (the expensive part of
    `opp_candidates`); returns the same `probs` list each state's candidate construction expects."""
    b = {key: torch.from_numpy(np.stack([np.ascontiguousarray(o[key]) for o in obs_list])) for key in obs_list[0]}
    mv, tg = opp_model.opp_logits(opp_model.encode(b), b)
    mv, tg = F.softmax(mv, -1).numpy(), F.softmax(tg, -1).numpy()
    return [[(mv[i, 0], tg[i, 0]), (mv[i, 1], tg[i, 1])] for i in range(len(obs_list))]


def _opp_candidates_from_probs(battle: DoubleBattle, obs: dict[str, np.ndarray], state: dict, k: int | None,
                               probs: list[tuple | None], *, mega_slots=None,
                               ensure_switches: bool = False) -> list[tuple[str, float]]:
    # None preserves full support for callers that filter legality before top-k.
    per_slot = [_slot_options(battle, state, s, probs[s]) for s in (0, 1)]
    mega = -1
    if mega_slots is not None:
        mega = min(mega_slots, default=-1)
    elif not state["sides"]["them"]["megaUsed"]:
        for s in (0, 1):
            mon = battle.opponent_active_pokemon[s]
            if mon and not mon.fainted and _species(mon) == mon.species and \
                    any(to_id_str(f) in _pokedex() and str(_pokedex()[to_id_str(f)].get("forme", "")).startswith("Mega")
                        and _item(mon).endswith("ite") for f in _pokedex()[mon.species].get("otherFormes", [])):
                mega = s
                break
    joint = []
    for (ca, pa), (cb, pb) in ((a, b) for a in per_slot[0] for b in per_slot[1]):
        if ca.startswith("switch") and ca == cb:
            continue
        ca += " mega" if mega == 0 and ca.startswith("move") else ""
        cb += " mega" if mega == 1 and cb.startswith("move") else ""
        joint.append((f"{ca}, {cb}", pa * pb))
    if probs[0] is None:  # all equally likely: a random k of them
        random.shuffle(joint)
    joint = sorted(joint, key=lambda c: -c[1])
    if k is not None:
        target_k = max(6, k) if ensure_switches else k
        selected = list(joint[:target_k])
        if ensure_switches:
            selected_set = {c for c, _ in selected}
            best_slot0_switches = {}
            best_slot1_switches = {}
            best_double_switch = None
            for c, p in joint:
                parts = [part.strip() for part in c.split(",")]
                if len(parts) != 2:
                    continue
                is_sw0 = parts[0].startswith("switch ")
                is_sw1 = parts[1].startswith("switch ")
                if is_sw0 and is_sw1:
                    if best_double_switch is None:
                        best_double_switch = (c, p)
                elif is_sw0:
                    tgt = parts[0].removeprefix("switch ")
                    if tgt not in best_slot0_switches:
                        best_slot0_switches[tgt] = (c, p)
                elif is_sw1:
                    tgt = parts[1].removeprefix("switch ")
                    if tgt not in best_slot1_switches:
                        best_slot1_switches[tgt] = (c, p)

            for tgt, (c, p) in best_slot0_switches.items():
                if not any(t.split(",")[0].strip() == f"switch {tgt}" for t, _ in selected):
                    if c not in selected_set:
                        selected.append((c, p))
                        selected_set.add(c)

            for tgt, (c, p) in best_slot1_switches.items():
                if not any(len(t.split(",")) > 1 and t.split(",")[1].strip() == f"switch {tgt}" for t, _ in selected):
                    if c not in selected_set:
                        selected.append((c, p))
                        selected_set.add(c)

            if best_double_switch is not None:
                if not any(len(t.split(",")) == 2 and t.split(",")[0].strip().startswith("switch ") and t.split(",")[1].strip().startswith("switch ") for t, _ in selected):
                    if best_double_switch[0] not in selected_set:
                        selected.append(best_double_switch)
                        selected_set.add(best_double_switch[0])
        joint = selected
    total = sum(p for _, p in joint) or 1.0
    return [(c, p / total) for c, p in joint]


@torch.no_grad()
def opp_candidates_batch(opp_model: Policy | None, items: list[tuple[DoubleBattle, dict[str, np.ndarray], dict]],
                         k: int) -> list[list[tuple[str, float]]]:
    """`opp_candidates` for N (battle, obs, state) tuples, with the opp_model call batched once
    across all of them instead of once per item."""
    if not items:
        return []
    if opp_model is not None and opp_model.config.get("aux"):
        probs_list = _opp_probs_batch(opp_model, [obs for _, obs, _ in items])
    else:
        probs_list = [[None, None]] * len(items)
    return [_opp_candidates_from_probs(battle, obs, state, k, probs, ensure_switches=True)
            for (battle, obs, state), probs in zip(items, probs_list)]


@torch.no_grad()
def values(model: Policy, battles: list[DoubleBattle], rating: float) -> np.ndarray:
    """Value head's win probability for each battle, from our side."""
    obs = [encode(b, rating) for b in battles]
    b = {k: torch.from_numpy(np.stack([o[k] for o in obs])) for k in obs[0]}
    return torch.sigmoid(model.value(model.encode(b)[0])[:, 0]).numpy()


def _playout_values(model: Policy, battle: DoubleBattle, res: list[dict], rating: float) -> np.ndarray:
    """Win probability after each one-turn playout.

    Showdown results contain protocol ``lines`` and are encoded through poke-env.
    The vgc-engine backend instead returns a bounded Rust-observation material
    leaf under ``value`` until its observation-to-network adapter lands.
    """
    v = np.full(len(res), np.nan)
    live, idx = [], []
    for i, r in enumerate(res):
        if r["winner"]:
            v[i] = {"us": 1.0, "them": 0.0}.get(r["winner"], 0.5)
        elif "value" in r:
            v[i] = float(r["value"])
        else:
            live.append(after(battle, r["lines"]))
            idx.append(i)
    if live:
        v[idx] = values(model, live, rating)
    return v


def _second_turn(model: Policy, opp_model: Policy | None, bridge: Bridge, battle: DoubleBattle,
                 res: list[dict], world: dict | None, fmt: str, teams_dir: str, rating: float,
                 k2: int, opp_k2: int, seeds2: list[int]) -> np.ndarray:
    """Value of each turn-1 playout in `res` after a second turn: we pick our best reply among the
    policy's `k2` candidates, the opponent mixes over their `opp_k2` likeliest. Playouts that end
    the game, force a switch or failed keep their one-turn value (value head after turn 1)."""
    v = np.full(len(res), np.nan)
    pending = []  # (i, b, b2, mask, obs, state)
    fallback = {}  # i -> b, for items without a usable next-turn position
    for i, r in enumerate(res):
        if r["winner"]:
            v[i] = {"us": 1.0, "them": 0.0}.get(r["winner"], 0.5)
            continue
        b = after(battle, r["lines"])
        if r.get("next"):
            try:
                b2 = copy.copy(b)
                b2.parse_request(r["next"]["request"])
                mask = apply_rules(b2, action_mask(b2))
                obs = encode(b2, rating)
                state = position(b2, fmt, teams_dir, world)
                if state is not None:
                    pending.append((i, b, b2, mask, obs, state))
                    continue
            except Exception:  # a position poke-env can't take the request for: one-turn value
                pass
        fallback[i] = b

    nodes, conts = [], []
    if pending:
        ours_batch = our_candidates_batch(model, [obs | {"mask": mask} for _, _, _, mask, obs, _ in pending],
                                          [mask for _, _, _, mask, _, _ in pending], k2)
        theirs_batch = opp_candidates_batch(
            opp_model, [(b2, obs, state) for _, _, b2, _, obs, state in pending], opp_k2)
    else:
        ours_batch, theirs_batch = [], []
    for (i, b, b2, _, _, _), ours, theirs in zip(pending, ours_batch, theirs_batch):
        if ours:
            r = res[i]
            choices = [DoublesEnv.action_to_order(a, b2, strict=False).message.removeprefix("/choose ")
                       for a, _ in ours]
            nodes.append((i, b2, len(choices), [p for _, p in theirs]))
            conts.append({"id": r["next"]["id"], "pairs": [(c, t) for c in choices for t, _ in theirs],
                          "seeds": seeds2})
        else:
            fallback[i] = b
    for i, b in fallback.items():
        nodes.append((i, b, 0, []))
    _second_turn.expanded += len(conts)
    _second_turn.nodes += len(res)
    if conts:
        outs = iter(bridge.cont(conts))
    leaves, where = [], []
    for n, (i, b, k, q) in enumerate(nodes):
        if not k:
            leaves.append(b)
            where.append((n, None))
            continue
        out = next(outs)
        _second_turn.sims += len(out)
        second_errors = [r["err"] for r in out if r["err"]]
        _second_turn.errors += len(second_errors)
        vals = np.full(len(out), np.nan)
        for j, r in enumerate(out):
            if r["winner"]:
                vals[j] = {"us": 1.0, "them": 0.0}.get(r["winner"], 0.5)
            else:
                leaves.append(after(b, r["lines"]))
                where.append((n, j))
        nodes[n] = (i, b, k, q, vals)
    if leaves:
        lv = values(model, leaves, rating)
        for (n, j), x in zip(where, lv):
            if j is None:
                v[nodes[n][0]] = x
            else:
                nodes[n][4][j] = x
    for node in nodes:
        if len(node) == 5:
            i, _, k, q, vals = node
            M2 = vals.reshape(k, len(q), len(seeds2)).mean(-1)
            v[i] = float((M2 @ np.array(q)).max())
    return v


# turn-1 playouts searched a second turn / all; turn-2 simulations / those the bridge failed on
_second_turn.expanded = _second_turn.nodes = _second_turn.sims = _second_turn.errors = 0


def search(model: Policy, opp_model: Policy | None, bridge: Bridge, battle: DoubleBattle,
           mask: np.ndarray, fmt: str, teams_dir: str, rating: float, k: int = 6, opp_k: int = 6,
           seeds: int = 2, prior: float = 0.0, n_worlds: int = 1, depth: int = 1, k2: int = 3,
           opp_k2: int = 3, seeds2: int = 1, refine: int = 0,
           refine_seeds: int = 0, chance_branching: bool = False,
           double_oracle: bool = False) -> tuple[np.ndarray, dict] | None:
    """(our chosen action, details) by one-turn search, or None when the position can't be
    rebuilt. With closed team sheets the opponent's hidden sets are guessed `n_worlds` ways
    (`worlds`); each candidate is scored in every world and the scores averaged by how likely
    each world is, so a play has to hold up against the sets they commonly run, not just one.
    With `refine` > 0 (one-turn search), the `refine` best candidates are simulated `refine_seeds`
    more times each, in every world, after the first `seeds` pass over all of them.
    With `depth` 2, each turn-1 playout is scored after a second turn (`_second_turn`) instead
    of by the value head right after turn 1.
    With `chance_branching` True, seed results are weighted by the probability of the chance
    events (damage rolls, crits, secondary effects) that occurred in each seed, giving a
    better approximation of true expected value (Nessie-style).
    `M` and `q` in the details are the most likely world's."""
    import os as _os
    _dbg = _os.environ.get("NESSIE_SEARCH_TIMING") == "1"
    _t0 = time.perf_counter() if _dbg else 0.0
    obs = encode(battle, rating)
    ours = our_candidates(model, obs | {"mask": mask}, mask, k)
    if len(ours) < 2:
        return None
    choices = [DoublesEnv.action_to_order(a, battle, strict=False).message.removeprefix("/choose ")
               for a, _ in ours]
    seed_list = [int(s) for s in np.random.randint(1, 2**30, seeds)]
    # value: per-action values (weighted across worlds), used for refinement ranking
    value = np.zeros(len(ours))
    per_world, errors, sims = [], [], 0
    for world, w in worlds(battle, teams_dir, n_worlds, fmt):
        _t0 = time.perf_counter() if _dbg else 0.0
        state = position(battle, fmt, teams_dir, world)
        if state is None:
            return None
        theirs = opp_candidates(opp_model, battle, obs, state, opp_k)
        pairs = [(c, t) for c in choices for t, _ in theirs]
        _t1 = time.perf_counter() if _dbg else 0.0
        if depth >= 2:
            seeds2_list = [int(s) for s in np.random.randint(1, 2**30, seeds2)]
            with bridge.lock:  # the kept turn-1 positions must survive until the turn-2 request
                res = bridge.run(state, pairs, seed_list, keep=True)
                _t2 = time.perf_counter() if _dbg else 0.0
                v = _second_turn(model, opp_model, bridge, battle, res, world, fmt, teams_dir, rating,
                                 k2, opp_k2, seeds2_list)
        else:
            res = bridge.run(state, pairs, seed_list)
            _t2 = time.perf_counter() if _dbg else 0.0
            v = _playout_values(model, battle, res, rating)
        if _dbg:
            _t3 = time.perf_counter()
            print(f"[search-timing] setup={_t1-_t0:.3f}s bridge={_t2-_t1:.3f}s value={_t3-_t2:.3f}s "
                  f"pairs={len(pairs)} seeds={len(seed_list)}", flush=True)
        M = v.reshape(len(ours), len(theirs), seeds).mean(-1)
        q = np.array([p for _, p in theirs])
        if double_oracle:
            eq_strat, eq_value, eq_info = equilibrium_strategy_from_search(
                M, q, epsilon=1e-3
            )
            br_values = M @ q
            value += w * br_values
            per_world.append([w, state, theirs, M, q, seeds, eq_strat, eq_info, eq_value])
        else:
            value += w * (M @ q)
            per_world.append([w, state, theirs, M, q, seeds, None, None, None])
        world_errors = [r["err"] for r in res if r["err"]]
        errors += world_errors
        sims += len(res)
    logp = np.array([lp for _, lp in ours])
    if refine and depth < 2 and len(ours) > refine:
        # Stage 2: the candidates still in the running get `refine_seeds` more simulations against
        # every reply in every world, so close calls aren't decided by a handful of rolls.
        top = np.argsort(-(value + prior * logp))[:refine]
        seeds_b = [int(x) for x in np.random.randint(1, 2**30, refine_seeds)]
        value = np.zeros(len(ours))
        for pw in per_world:
            w, state, theirs, M, q, n, eq_strat, eq_info, eq_value = pw
            pairs = [(choices[i], t) for i in top for t, _ in theirs]
            res = bridge.run(state, pairs, seeds_b)
            v = _playout_values(model, battle, res, rating)
            extra = v.reshape(len(top), len(theirs), refine_seeds)
            M = M.copy()
            M[top] = (M[top] * n + extra.sum(-1)) / (n + refine_seeds)
            if double_oracle:
                eq_strat, eq_value, eq_info = equilibrium_strategy_from_search(M, q, epsilon=1e-3)
                pw[6] = eq_strat
                pw[7] = eq_info
                pw[8] = eq_value
            pw[3] = M
            refine_errors = [r["err"] for r in res if r["err"]]
            errors += refine_errors
            sims += len(res)
        for w, _, _, M, q, _, eq_strat, eq_info, eq_value in per_world:
            value += w * (M @ q)
    # Action selection
    M, q, their_choices = per_world[0][3], per_world[0][4], [t for t, _ in per_world[0][2]]
    if double_oracle and per_world[0][6] is not None:
        eq_strat = sum(pw[0] * pw[6] for pw in per_world if pw[6] is not None)
        eq_strat = np.maximum(np.asarray(eq_strat, dtype=np.float64), 0.0)
        tot = eq_strat.sum()
        if tot > 0:
            eq_strat /= tot
        else:
            eq_strat = np.ones(len(ours), dtype=np.float64) / len(ours)
        eq_value = sum(pw[0] * pw[8] for pw in per_world if pw[8] is not None)
        eq_info = per_world[0][7]
        score = eq_strat
        best = int(np.random.choice(len(ours), p=eq_strat))
    else:
        eq_strat = eq_value = eq_info = None
        score = value + prior * logp
        best = int(score.argmax())
    return ours[best][0], {"M": M, "q": q, "ours": choices, "theirs": their_choices,
                           "actions": np.stack([a for a, _ in ours]), "value": value,
                           "logp": np.array([lp for _, lp in ours]), "obs": obs | {"mask": mask},
                           "score": score, "errors": errors, "sims": sims,
                           "eq_strat": eq_strat, "eq_value": eq_value, "eq_info": eq_info}


class SearchPlayer(NNPlayer):
    """NNPlayer that picks ordinary move-turn actions by `search` (the policy otherwise)."""

    def __init__(self, *args, opp_model: Policy | str | Path | None = None, fmt: str,
                 teams_dir: str | Path, showdown: str | Path = "pokemon-showdown", k: int = 6,
                 opp_k: int = 6, seeds: int = 16, prior: float = 0.1, worlds: int = 16, depth: int = 1,
                 k2: int = 3, opp_k2: int = 3, seeds2: int = 1, procs: int = 1, refine: int = 0,
                 refine_seeds: int = 0, chance_branching: bool = False, opp_br: int = 0, lanes: int = 1,
                 backend: str = "vgc-engine", bridge: Bridge | VGCBridge | None = None,
                 double_oracle: bool = True, closed_sheets: bool = False,
                 solver: str = "regret", worst_case_weight: float = 0.5,
                 regret_iterations: int = 2048, value_device: str = "cpu",
                 ensure_switches: bool = True, **kwargs):
        super().__init__(*args, battle_format=fmt, **kwargs)
        if backend in ("vgc-engine", "rust"):
            backend = "vgc-engine"
        self.closed_sheets = closed_sheets or not kwargs.get("accept_open_team_sheet", True)
        self.opp_model = load_policy(opp_model) if isinstance(opp_model, (str, Path)) else opp_model
        self.fmt, self.teams_dir = fmt, str(teams_dir)
        # One Bridge's requests all serialize on its lock (needed within a request: a two-turn
        # search holds it across turn 1 and turn 2). With several concurrent battles, a single
        # Bridge makes every one of them queue behind the others' searches no matter how high
        # --concurrency is. `lanes` independent Bridges, picked per battle by tag hash, let that
        # many battles' searches actually run at once (each still single-flight within its lane).
        if bridge is not None:
            self.bridges = [bridge]
        elif backend in ("vgc-engine", "rust"):
            self.bridges = [VGCBridge() for _ in range(max(1, lanes))]
        elif backend == "showdown":
            self.bridges = [Bridge(showdown, procs=procs) for _ in range(max(1, lanes))]
        else:
            raise ValueError(f"unknown search backend {backend!r} (want showdown, vgc-engine, or rust)")
        self.bridge = self.bridges[0]  # back-compat: bo3.Adapter reads this for its own searches
        self.backend = backend
        self.k, self.opp_k, self.seeds, self.prior, self.worlds = k, opp_k, seeds, prior, worlds
        self.depth, self.k2, self.opp_k2, self.seeds2 = depth, k2, opp_k2, seeds2
        self.refine, self.refine_seeds = refine, refine_seeds
        self.chance_branching = chance_branching
        self.opp_br = opp_br
        self.ensure_switches = ensure_switches
        self.double_oracle = double_oracle
        self.solver, self.worst_case_weight, self.regret_iterations = solver, worst_case_weight, regret_iterations
        self.value_device = value_device
        self.search_value_model = self.model if value_device == "cpu" else copy.deepcopy(self.model).to(value_device).eval()
        self.searched = self.fallbacks = self.countered = 0
        self.sims = self.sim_errors = 0  # simulated pairings, and those the bridge failed on
        self.search_time = 0.0
        self.failed_in_a_row = 0
        self.on_search = None  # called with (battle, details) after each search (exit.py records them)
        self.refresher = None  # munchstats.Refresher: fetch unknown opposing species' usage (ladder bot)
        # Bo3: humans answer the turn 1 we played last game. With the same leads on both sides, weight
        # their best reply to our last turn-1 choice by `counter_t1` (0 = off).
        self.counter_t1 = 0.0
        self.last_t1: dict[tuple, str] = {}  # (Bo3 room, our leads, their leads) -> our turn-1 choice

    def _lead_risk(self, l1: Pokemon, l2: Pokemon, foes: list[Pokemon]) -> float:
        risk = 0.0
        for lead, partner in [(l1, l2), (l2, l1)]:
            if getattr(lead, "item", "") == "focussash":
                continue
            has_support = any(mv in getattr(partner, "moves", {}) for mv in ("fakeout", "followme", "ragepowder"))
            lead_spe = lead.base_stats.get("spe", 100)
            for foe in foes:
                foe_spe = foe.base_stats.get("spe", 100)
                if foe_spe > lead_spe:
                    for t in foe.types:
                        if t is None:
                            continue
                        eff = lead.damage_multiplier(t)
                        if eff >= 4.0:
                            risk += 3.0 if not has_support else 0.5
                        elif eff >= 2.0:
                            risk += 1.0 if not has_support else 0.2
        return risk

    async def teampreview(self, battle: AbstractBattle) -> str:
        # Opposing species with no MunchStats data yet: fetch them in the background (search
        # picks them up on a later turn; until then it uses the team pool's spreads).
        if self.refresher is not None:
            self.refresher.need([m.species for m in battle.teampreview_opponent_team])
        return await super().teampreview(battle)

    async def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        assert isinstance(battle, DoubleBattle)
        battle._spd_on = self.speed_inference
        battle._guess = self.set_guess
        battle._dmg_on = self.damage_inference
        self.attach_series(battle)
        if battle.battle_tag not in self.leads:
            self.leads[battle.battle_tag] = (
                tuple(m.species for m in battle.active_pokemon if m),
                tuple(m.species for m in battle.opponent_active_pokemon if m))
        if any(battle.force_switch) or any(m is None or m.fainted for m in battle.active_pokemon):
            return await super().choose_move(battle)
        mask = action_mask(battle)
        if self.rules:
            mask = apply_rules(battle, mask)
        t = time.time()
        bridge = self.bridges[hash(battle.battle_tag) % len(self.bridges)]
        try:
            # In a worker thread: a search can take seconds, and blocking the event loop that long
            # starves the Showdown websocket's keepalive pings until the server drops the connection.
            if self.backend in ("vgc-engine", "rust"):
                from ai_vgc.nn.rust_search import rust_search

                out = await asyncio.to_thread(
                    rust_search,
                    self.model,
                    self.opp_model,
                    battle,
                    mask,
                    self.fmt,
                    self.teams_dir,
                    self.rating,
                    self.k,
                    self.opp_k,
                    self.worlds,
                    self.depth,
                    self.k2,
                    self.opp_k2,
                    self.refine,
                    self.prior,
                    self.double_oracle,
                    not self.closed_sheets,
                    None,
                    bridge,
                    seeds=self.seeds,
                    seeds2=self.seeds2,
                    solver=self.solver,
                    worst_case_weight=self.worst_case_weight,
                    regret_iterations=self.regret_iterations,
                    value_model=self.search_value_model,
                    chance_nodes=self.chance_branching,
                    opp_br=self.opp_br,
                    ensure_switches=self.ensure_switches,
                )
            else:
                out = await asyncio.to_thread(search, self.model, self.opp_model, bridge, battle, mask,
                                              self.fmt, self.teams_dir, self.rating, self.k, self.opp_k,
                                              self.seeds, self.prior, self.worlds, self.depth, self.k2,
                                              self.opp_k2, self.seeds2, self.refine, self.refine_seeds,
                                              self.chance_branching, self.double_oracle)
        except Exception as e:  # a position the bridge can't handle: play the policy
            self.logger.warning("search failed on turn %s: %r", battle.turn, e)
            out = None
            self.failed_in_a_row += 1
            if self.failed_in_a_row >= 3:  # e.g. it loaded a half-rebuilt dist/: start a fresh one
                bridge.kill()
                self.failed_in_a_row = 0
        else:
            self.failed_in_a_row = 0
        self.search_time += time.time() - t
        if out is None:
            self.fallbacks += 1
            return await super().choose_move(battle)
        self.searched += 1
        self.sims += out[1]["sims"]
        self.sim_errors += len(out[1].get("errors", []))
        # A matrix made entirely of simulator fallbacks has no valid search signal.
        # Play policy rather than letting default actions decide the matrix game.
        if "errors" in out[1] and out[1]["sims"] > 0 and len(out[1]["errors"]) == out[1]["sims"]:
            self.logger.warning("search had no valid simulations on turn %s; using policy", battle.turn)
            self.fallbacks += 1
            return await super().choose_move(battle)
        if self.on_search:
            self.on_search(battle, out[1])
        action, d = out[0], out[1]
        series = self.series_of.get(battle.battle_tag)
        key = series and battle.turn == 1 and (series[0], *map(frozenset, self.leads[battle.battle_tag]))
        if key and self.counter_t1 and self.last_t1.get(key) in d["ours"]:
            q = (1 - self.counter_t1) * d["q"]
            q[int(d["M"][d["ours"].index(self.last_t1[key])].argmin())] += self.counter_t1
            d["score"] = d["M"] @ q + self.prior * d["logp"]
            action = d["actions"][int(d["score"].argmax())]
            self.countered += 1
        if self.vary and self.later_game(battle):
            # Near-ties by search's score, sampled by the policy's own probabilities.
            near = np.flatnonzero(d["score"] >= d["score"].max() - self.vary_margin)
            p = np.exp(d["logp"][near] - d["logp"][near].max())
            action = d["actions"][np.random.choice(near, p=p / p.sum())]
        order = safe_action_to_order(action, battle)
        if key:
            self.last_t1[key] = order.message.removeprefix("/choose ")
        return order
