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
4. We pick the row with the best expected value against the opponent's distribution,
   plus `prior` times the policy's log-probability to break near-ties toward the policy.

Search only runs on ordinary move turns; forced switches and team preview use the policy.
"""

from __future__ import annotations

import copy
import json
import random
import select
import subprocess
import time
from collections import Counter
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
from ai_vgc.nn.player import NNPlayer, action_mask, load_policy
from ai_vgc.nn.rules import apply_rules

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

    for _ in range(rounds):
        if items:
            fit(items, lambda st: st[0])
        if abilities:
            fit(abilities, lambda st: st[1])
        for m, p in list(moves.items())[:8]:  # a move's share = sets that carry it
            mass = sum(w.values())
            cur = sum(n for st, n in w.items() if m in st[2]) / mass
            if 0 < cur < 1:
                for st in w:
                    w[st] *= p / cur if m in st[2] else (1 - p) / (1 - cur)
    out = Counter({st: n for st, n in w.items() if n > 1e-6})
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
    those swapped for the next ones. Weights are products of the usages."""
    entry = _munch().get(base)
    if not entry or not entry.get("moves"):
        return Counter()
    moves = [(to_id_str(m), p / 100) for m, p in entry["moves"]]
    items = [(to_id_str(i), p) for i, p in entry.get("items", [])[:4]] or [("", 1.0)]
    abilities = [(to_id_str(a), p) for a, p in entry.get("abilities", [])[:2]] or [("", 1.0)]
    core, extra = moves[:4], moves[4:7]
    movesets = [(frozenset(m for m, _ in core), float(np.prod([p for _, p in core])))]
    for i in range(len(core)):
        for m, p in extra:
            swapped = core[:i] + core[i + 1:] + [(m, p)]
            movesets.append((frozenset(x for x, _ in swapped),
                             float(np.prod([q for _, q in swapped])) * (1 - core[i][1]) / max(core[i][1], 1e-3)))
    out = Counter({(i, a, ms): pi * pa * pm for i, pi in items for a, pa in abilities for ms, pm in movesets})
    return Counter(dict(out.most_common(top)))


def worlds(battle: DoubleBattle, teams_dir: str, n: int, fmt: str = "") -> list[tuple[dict, float]]:
    """Up to `n` guesses at the opponent's hidden sets, {name: (item, ability, moves, spread)},
    each with its probability (the product over their Pokemon, renormalised over the ones kept).
    Sets come from `set_posterior`; spreads (stat points and nature, which no team sheet shows)
    from `spread_prior`, or None to use the team pool's. The most likely of both for every
    Pokemon comes first; the rest are sampled."""
    post = {}
    for mon in battle.opponent_team.values():
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


class Bridge:
    """A `sim_bridge.js` process for one Showdown checkout."""

    def __init__(self, showdown: str | Path = "pokemon-showdown", timeout: float = 60.0):
        self.showdown = showdown
        self.timeout = timeout
        self._start()

    def _start(self) -> None:
        self.proc = subprocess.Popen(["node", str(BRIDGE), str(self.showdown)], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, text=True, bufsize=1)

    def _ask(self, req: dict) -> dict:
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

    def run(self, state: dict, pairs: list[tuple[str, str]], seeds: list[int]) -> list[dict]:
        return self._ask({"state": state, "pairs": pairs, "seeds": seeds})["results"]

    def dump(self, state: dict) -> list:
        """The position as the simulator rebuilt it, for checking against poke-env's."""
        return self._ask({"state": state, "dump": True})["dump"]

    def close(self) -> None:
        self.proc.terminate()


def after(battle: DoubleBattle, lines: list[str]) -> DoubleBattle:
    """A copy of the battle with a simulated turn's log parsed into it."""
    logger, battle.logger = battle.logger, None
    try:
        b = copy.deepcopy(battle)
    finally:
        battle.logger = logger
    b.logger = logger
    for line in lines:
        split = line.split("|")
        if len(split) > 1 and split[1] not in IGNORE:
            b.parse_message(split)
    return b


@torch.no_grad()
def our_candidates(model: Policy, obs: dict[str, np.ndarray], mask: np.ndarray, k: int,
                   ka: int = 4) -> list[tuple[np.ndarray, float]]:
    """The policy's k most likely joint actions (slot a from its top `ka`), with log-probs."""
    b = {key: torch.from_numpy(np.ascontiguousarray(v))[None] for key, v in obs.items()}
    m = torch.from_numpy(mask)[None]
    enc = model.encode(b)
    lp_a = F.log_softmax(masked(model.slot_logits(enc, b, 0), m[:, 0]), -1)[0]
    out = []
    for a0 in lp_a.topk(min(ka, int(mask[0].sum()))).indices.tolist():
        prev = torch.tensor([a0])
        lp_b = F.log_softmax(masked(model.slot_logits(enc, b, 1, prev), slot_b_mask(m[:, 1], prev)), -1)[0]
        for a1 in lp_b.topk(min(k, int(slot_b_mask(m[:, 1], prev).sum()))).indices.tolist():
            out.append((np.array([a0, a1]), (lp_a[a0] + lp_b[a1]).item()))
    return sorted(out, key=lambda c: -c[1])[:k]


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
    if bench and probs is not None:
        out[f"switch {bench[0]}"] = float(probs[0][4])
    total = sum(out.values()) or 1.0
    return [(c, p / total) for c, p in out.items()]


@torch.no_grad()
def opp_candidates(opp_model: Policy | None, battle: DoubleBattle, obs: dict[str, np.ndarray],
                   state: dict, k: int) -> list[tuple[str, float]]:
    """The opponent's k most likely joint choices (in their side's choice syntax), with
    probabilities renormalised over the k (a random k, equally likely, without the head). A side that can still Mega Evolve does it with the
    first Pokemon holding its stone, as players nearly always do."""
    probs = [None, None]
    if opp_model is not None and opp_model.config.get("aux"):
        b = {key: torch.from_numpy(np.ascontiguousarray(v))[None] for key, v in obs.items()}
        mv, tg = opp_model.opp_logits(opp_model.encode(b), b)
        mv, tg = F.softmax(mv[0], -1).numpy(), F.softmax(tg[0], -1).numpy()
        probs = [(mv[0], tg[0]), (mv[1], tg[1])]
    per_slot = [_slot_options(battle, state, s, probs[s]) for s in (0, 1)]
    mega = -1
    if not state["sides"]["them"]["megaUsed"]:
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
    joint = sorted(joint, key=lambda c: -c[1])[:k]
    total = sum(p for _, p in joint) or 1.0
    return [(c, p / total) for c, p in joint]


@torch.no_grad()
def values(model: Policy, battles: list[DoubleBattle], rating: float) -> np.ndarray:
    """Value head's win probability for each battle, from our side."""
    obs = [encode(b, rating) for b in battles]
    b = {k: torch.from_numpy(np.stack([o[k] for o in obs])) for k in obs[0]}
    return torch.sigmoid(model.value(model.encode(b)[0])[:, 0]).numpy()


def search(model: Policy, opp_model: Policy | None, bridge: Bridge, battle: DoubleBattle,
           mask: np.ndarray, fmt: str, teams_dir: str, rating: float, k: int = 6, opp_k: int = 6,
           seeds: int = 2, prior: float = 0.0, n_worlds: int = 1) -> tuple[np.ndarray, dict] | None:
    """(our chosen action, details) by one-turn search, or None when the position can't be
    rebuilt. With closed team sheets the opponent's hidden sets are guessed `n_worlds` ways
    (`worlds`); each candidate is scored in every world and the scores averaged by how likely
    each world is, so a play has to hold up against the sets they commonly run, not just one.
    `M` and `q` in the details are the most likely world's."""
    obs = encode(battle, rating)
    ours = our_candidates(model, obs | {"mask": mask}, mask, k)
    if len(ours) < 2:
        return None
    choices = [DoublesEnv.action_to_order(a, battle, strict=False).message.removeprefix("/choose ")
               for a, _ in ours]
    seed_list = [int(s) for s in np.random.randint(1, 2**30, seeds)]
    value, first, errors, sims = np.zeros(len(ours)), None, [], 0
    for world, w in worlds(battle, teams_dir, n_worlds, fmt):
        state = position(battle, fmt, teams_dir, world)
        if state is None:
            return None
        theirs = opp_candidates(opp_model, battle, obs, state, opp_k)
        pairs = [(c, t) for c in choices for t, _ in theirs]
        res = bridge.run(state, pairs, seed_list)
        v = np.full(len(res), np.nan)
        live, idx = [], []
        for i, r in enumerate(res):
            if r["winner"]:
                v[i] = {"us": 1.0, "them": 0.0}.get(r["winner"], 0.5)
            else:
                live.append(after(battle, r["lines"]))
                idx.append(i)
        if live:
            v[idx] = values(model, live, rating)
        M = v.reshape(len(ours), len(theirs), seeds).mean(-1)
        q = np.array([p for _, p in theirs])
        value += w * (M @ q)
        errors += [r["err"] for r in res if r["err"]]
        sims += len(res)
        if first is None:
            first = (M, q, [t for t, _ in theirs])
    M, q, their_choices = first
    score = value + prior * np.array([lp for _, lp in ours])
    best = int(score.argmax())
    return ours[best][0], {"M": M, "q": q, "ours": choices, "theirs": their_choices,
                           "actions": np.stack([a for a, _ in ours]), "value": value,
                           "logp": np.array([lp for _, lp in ours]), "obs": obs | {"mask": mask},
                           "score": score, "errors": errors, "sims": sims}


class SearchPlayer(NNPlayer):
    """NNPlayer that picks ordinary move-turn actions by `search` (the policy otherwise)."""

    def __init__(self, *args, opp_model: Policy | str | Path | None = None, fmt: str,
                 teams_dir: str | Path, showdown: str | Path = "pokemon-showdown", k: int = 6,
                 opp_k: int = 6, seeds: int = 2, prior: float = 0.0, worlds: int = 1, **kwargs):
        super().__init__(*args, battle_format=fmt, **kwargs)
        self.opp_model = load_policy(opp_model) if isinstance(opp_model, (str, Path)) else opp_model
        self.fmt, self.teams_dir = fmt, str(teams_dir)
        self.bridge = Bridge(showdown)
        self.k, self.opp_k, self.seeds, self.prior, self.worlds = k, opp_k, seeds, prior, worlds
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

    async def teampreview(self, battle: AbstractBattle) -> str:
        # Opposing species with no MunchStats data yet: fetch them in the background (search
        # picks them up on a later turn; until then it uses the team pool's spreads).
        if self.refresher is not None:
            self.refresher.need([m.species for m in battle.teampreview_opponent_team])
        return await super().teampreview(battle)

    async def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        assert isinstance(battle, DoubleBattle)
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
        try:
            out = search(self.model, self.opp_model, self.bridge, battle, mask, self.fmt, self.teams_dir,
                         self.rating, self.k, self.opp_k, self.seeds, self.prior, self.worlds)
        except Exception as e:  # a position the bridge can't handle: play the policy
            self.logger.warning("search failed on turn %s: %r", battle.turn, e)
            out = None
            self.failed_in_a_row += 1
            if self.failed_in_a_row >= 3:  # e.g. it loaded a half-rebuilt dist/: start a fresh one
                self.bridge.proc.kill()
                self.bridge.proc.wait()
                self.failed_in_a_row = 0
        else:
            self.failed_in_a_row = 0
        self.search_time += time.time() - t
        if out is None:
            self.fallbacks += 1
            return await super().choose_move(battle)
        self.searched += 1
        self.sims += out[1]["sims"]
        self.sim_errors += len(out[1]["errors"])
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
        order = DoublesEnv.action_to_order(action, battle, strict=False)
        if key:
            self.last_t1[key] = order.message.removeprefix("/choose ")
        return order
