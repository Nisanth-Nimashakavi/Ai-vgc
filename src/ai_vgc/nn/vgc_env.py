"""Native vgc-engine rollout environment for PPO self-play.

This module deliberately keeps the existing Policy/PPO transition contract while
removing poke-env, websockets, protocol parsing, and Pokémon Showdown from the
rollout path.  Battles, legal actions, and terminal rewards all come from
``vgc_engine.Battle``.

The policy's historical action space has 47 meaningful per-slot actions:
pass, six team-index switches, and move/target choices in normal and special
(Mega) variants. The native Regulation M-C rollout disables Tera; the special
variant therefore means Mega Evolution only, preserving old checkpoint
semantics without inventing a Tera action mapping.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from poke_env.data import GenData
from poke_env.teambuilder import Teambuilder

from ai_vgc.nn.encode import (
    BOOSTS,
    EFFECTS,
    FIELDS,
    MEGA_OFFSET,
    N_ACT,
    N_DMG,
    N_GLOB,
    N_SER_GLOB,
    N_SER_MV,
    N_SER_TOK,
    N_TOK,
    SIDE,
    STATUSES,
    T,
    TYPES,
    WEATHERS,
    vocab,
)
from ai_vgc.nn.player import sample_batch, sample_packed_batch
from ai_vgc.teams import TEAMS_DIR

_STATS = ("hp", "atk", "def", "spa", "spd", "spe")
_STATUS = {"burn": "brn", "paralysis": "par", "poison": "psn", "toxic": "tox", "sleep": "slp", "freeze": "frz"}
_WEATHER = {"sun": "sunnyday", "rain": "raindance", "sand": "sandstorm", "snow": "snowscape", "hail": "hail"}
_TERRAIN = {"electric": "electricterrain", "grassy": "grassyterrain", "misty": "mistyterrain", "psychic": "psychicterrain"}
_VOLATILE = {
    "substitute": "substitute_hp", "taunt": "taunt", "encore": "encore", "disable": "disable",
    "confusion": "confusion", "leech_seed": "leech_seed", "perish1": "perish", "perish2": "perish",
    "perish3": "perish", "salt_cure": "salt_cure", "heal_block": "heal_block",
}
_FIELD = {
    "trick_room": "trick_room", "electric_terrain": "electric", "grassy_terrain": "grassy",
    "misty_terrain": "misty", "psychic_terrain": "psychic", "gravity": "gravity",
    "wonder_room": "wonder_room", "magic_room": "magic_room",
}
_SIDE = {
    "tailwind": "tailwind", "reflect": "reflect", "light_screen": "light_screen",
    "aurora_veil": "aurora_veil", "safeguard": "safeguard", "mist": "mist",
}

# The model/learner observation contract. Native packed rollouts accumulate
# these arrays directly, rather than first materialising a Python dict per state.
_PACKED_KEYS = (
    "tok_cat", "tok_num", "mv_cat", "mv_dyn", "dmg", "glob", "act_tok", "mask",
    "ser_tok", "ser_mv", "ser_glob",
)


def _id(value: str | None) -> str:
    return "".join(c for c in (value or "").lower() if c.isalnum())


def _team_json(text: str, fmt: str = "") -> str:
    """Convert a local Showdown export into Battle.from_teams JSON."""
    team = []
    for mon in Teambuilder.parse_showdown_team(text):
        evs = mon.evs or [0] * 6
        ivs = mon.ivs or [31] * 6
        if fmt.startswith("gen9champions"):
            if any(x < 0 or x > 32 for x in evs) or sum(evs) > 66:
                raise ValueError("Champions requires at most 66 Stat Points total and 32 per stat")
            evs = [0 if x == 0 else 8 * x - 4 for x in evs]
            ivs = [31] * 6
        team.append({
            "species": mon.species or mon.nickname,
            "level": mon.level or 50,
            "ability": mon.ability or None,
            "item": mon.item or None,
            "nature": mon.nature or "Serious",
            "moves": mon.moves,
            "evs": dict(zip(_STATS, evs)),
            "ivs": dict(zip(_STATS, ivs)),
            "tera_type": mon.tera_type or None,
        })
    if len(team) < 2:
        raise ValueError("vgc-engine doubles rollout needs at least two Pokémon per team")
    return json.dumps(team)


def _pool(teams_dir: str | Path | None) -> list[str]:
    root = Path(teams_dir) if teams_dir else TEAMS_DIR
    teams = [p.read_text() for p in sorted(root.glob("*.txt"))]
    if not teams:
        raise FileNotFoundError(f"no teams in {root}")
    return teams


def _info(species: str) -> dict:
    return GenData.from_gen(9).pokedex.get(_id(species), {})


def _effects(mon: dict) -> dict:
    return mon.get("volatiles", {})


def _tok_num(mon: dict, ours: bool, active_slot: int | None) -> list[float]:
    info = _info(mon["species"])
    base = info.get("baseStats", {})
    boosts = mon.get("boosts", {})
    vol = _effects(mon)
    status = _STATUS.get(mon.get("status", ""), mon.get("status", ""))
    first_turn = float(mon.get("turns_active", 1) == 0)
    return [
        1.0, float(ours), float(not ours),
        float(mon.get("hp", 0)) / max(int(mon.get("max_hp", 1)), 1), float(mon.get("fainted", False)),
        float(active_slot == 0), float(active_slot == 1),
        *[float(base.get(s, 0)) / 255 for s in _STATS],
        float(mon.get("max_hp", 0)) / 255 if ours else 0.0,
        *([0.0] * 5),  # opponent exact stat fields are intentionally unavailable in old encoder too
        *[float(t in mon.get("types", [])) for t in map(lambda x: x.name.lower(), TYPES)],
        *[float(status == s.name.lower()) for s in STATUSES],
        *[float(boosts.get({"accuracy": "acc", "evasion": "eva"}.get(b, b), 0)) / 6 for b in BOOSTS],
        *[float(bool(vol.get(_VOLATILE.get(e.name.lower(), ""), False))) for e in EFFECTS],
        first_turn, float(vol.get("protect", False)) / 3, 0.0, 0.0, 1.0, 1.0,
        float("ite" in _id(mon.get("item"))), float("mega" in _id(mon["species"])),
        min(float(info.get("weightkg", 0)), 500.0) / 500,
    ]


def perspective_obs(obs: dict, side: int) -> dict:
    """Return a side-relative view so each policy sees itself in token slots 0–5."""
    if side == 0:
        return obs
    return obs | {"sides": [obs["sides"][1], obs["sides"][0]]}



def preview_mask(obs: dict, selected: set[int] | None = None) -> np.ndarray:
    """Legal four-Pokémon selection mask in the historical preview action space."""
    selected = selected or set()
    mask = np.zeros((2, N_ACT), bool)
    eligible = [i + 1 for i, mon in enumerate([*obs["sides"][0]["active"], *obs["sides"][0]["bench"]][:6])
                if not mon.get("fainted", False) and i not in selected]
    mask[:, eligible] = True
    return mask


def choose_preview(model: torch.nn.Module, obs: dict, rating: float, greedy: bool) -> tuple[list[int], list[dict]]:
    """Sample four distinct team indices through the existing preview policy."""
    chosen: list[int] = []
    rows: list[dict] = []
    for _ in range(2):
        row = encode_vgc_obs(obs, rating, preview=True)
        row["mask"] = preview_mask(obs, set(chosen))
        action, logp, value = sample_batch(model, [row], [greedy])[0]
        picks = [int(action[0]), int(action[1])]
        # Slot-B masking prevents a duplicate inside one decision. Retain a
        # defensive deterministic fallback for malformed legacy logits.
        for pick in picks:
            available = np.flatnonzero(row["mask"][0])
            if pick in chosen or not row["mask"][0, pick]:
                pick = int(next(x for x in available if x not in chosen))
            chosen.append(pick)
        rows.append(row | {"action": action.astype(np.int16), "logp": logp, "value": value, "preview": True})
    return chosen[:4], rows



def choose_preview_batch(model: torch.nn.Module, full: list[dict], rating: float, greedy: bool,
                         side: int) -> tuple[list[list[int]], list[list[dict]]]:
    """Batched VGC bring-four selection for one player perspective.

    Two policy calls select two Pokémon each, giving GPU batches of all live
    games rather than four Python-sized inference calls per game.
    """
    chosen = [[] for _ in full]
    rows = [[] for _ in full]
    for _ in range(2):
        obs = []
        for i, state in enumerate(full):
            row = encode_vgc_obs(perspective_obs(state, side), rating, preview=True)
            row["mask"] = preview_mask(perspective_obs(state, side), set(chosen[i]))
            obs.append(row)
        sampled = sample_batch(model, obs, [greedy] * len(obs))
        for i, (row, (action, logp, value)) in enumerate(zip(obs, sampled)):
            picks = []
            available = set(np.flatnonzero(row["mask"][0]).tolist()) - set(chosen[i])
            for pick in action:
                pick = int(pick)
                if pick not in available:
                    pick = min(available)
                available.remove(pick)
                picks.append(pick)
            chosen[i].extend(picks)
            # PPO must record the actual legal fallback rather than the stale
            # sampled id if a legacy checkpoint produced an invalid selection.
            rows[i].append(row | {"action": np.asarray(picks, np.int16), "logp": logp,
                                  "value": value, "preview": True})
    return chosen, rows


def policy_vocab_tables() -> dict[str, np.ndarray]:
    """Engine-indexed categorical ids for the persistent Python policy vocab.

    Rust owns hot-path state traversal, while these compact tables preserve the
    checkpoint's existing vocabulary IDs without coupling its order to engine
    generated-data order. They are installed once on each BattleBatch.
    """
    import vgc_engine

    cached = getattr(policy_vocab_tables, "_cached", None)
    if cached is not None:
        return cached
    engine_vocab = vgc_engine.policy_vocab()
    persistent = vocab()
    policy_vocab_tables._cached = {
        key: np.asarray([persistent[key].get(name, 0) for name in engine_vocab[key]], np.uint32)
        for key in ("species", "items", "abilities", "moves")
    }
    return policy_vocab_tables._cached


def cts_usage_tables() -> dict[str, np.ndarray]:
    """Public MunchStats priors for native closed-sheet observations.

    This intentionally receives only a species name and the public usage
    cache.  It never sees a sampled opponent team's item, ability, moves, EVs,
    IVs, or nature.  The Rust batch uses it exactly as the ladder's fallback
    species posterior: revealed moves replace these guesses as a battle goes
    on, while unrevealed slots remain guesses.
    """
    import vgc_engine
    from ai_vgc import munchstats
    from ai_vgc.bulk import stats_for

    try:
        mtime = munchstats.CACHE.stat().st_mtime_ns
    except OSError:
        mtime = -1
    cached = getattr(cts_usage_tables, "_cached", None)
    if cached is not None and cached[0] == mtime:
        return cached[1]

    names = vgc_engine.policy_vocab()
    lookup = {key: {_id(name): i for i, name in enumerate(names[key])}
              for key in ("items", "abilities", "moves")}
    usage = munchstats.load()
    count = len(names["species"])
    unknown = np.uint16(0xFFFF)
    items = np.full(count, unknown, np.uint16)
    abilities = np.full(count, unknown, np.uint16)
    moves = np.full((count, 4), unknown, np.uint16)
    stats = np.zeros((count, 6), np.uint16)
    for i, species in enumerate(names["species"]):
        entry = usage.get(_id(species), {})
        item_rows, ability_rows, move_rows = (entry.get("items") or [], entry.get("abilities") or [],
                                              entry.get("moves") or [])
        if item_rows:
            items[i] = lookup["items"].get(_id(item_rows[0][0]), unknown)
        if ability_rows:
            abilities[i] = lookup["abilities"].get(_id(ability_rows[0][0]), unknown)
        for slot, row in enumerate(move_rows[:4]):
            moves[i, slot] = lookup["moves"].get(_id(row[0]), unknown)
        base = _info(species).get("baseStats", {})
        if not base:
            continue
        try:
            points = tuple(int(x) for x in (entry.get("spreads") or [["0/0/0/0/0/0"]])[0][0].split("/"))
            nature = (entry.get("natures") or [["Hardy"]])[0][0]
            estimate = stats_for(base, points, nature)
        except (TypeError, ValueError, KeyError):
            estimate = stats_for(base, (0, 0, 0, 0, 0, 0), "Hardy")
        stats[i] = [estimate[k] for k in _STATS]
    tables = {"items": items, "abilities": abilities,
              "moves": moves.reshape(-1), "stats": stats.reshape(-1)}
    cts_usage_tables._cached = (mtime, tables)
    return tables


def repair_packed_actions(
    rows: list[dict[str, np.ndarray]],
    actions: list[tuple[np.ndarray, float, float]],
) -> list[tuple[np.ndarray, float, float]]:
    """Defend the Rust action boundary against an invalid backend sample.

    The packed mask is the engine's authoritative legality contract. CUDA has
    occasionally returned an ID inconsistent with the Torch-side mask, so
    repair that output at the boundary rather than ever passing it to Rust.
    This is a fail-safe for rollout continuity; the eager, full-precision
    sampler above remains the primary path.
    """
    repaired = []
    for row, (action, logp, value) in zip(rows, actions):
        action = action.astype(np.int16, copy=True)
        for slot, action_id in enumerate(action):
            legal = np.flatnonzero(row["mask"][slot])
            if not len(legal):
                raise RuntimeError(f"native policy mask is empty for slot {slot}")
            if not row["mask"][slot, int(action_id)]:
                action[slot] = legal[0]
        repaired.append((action, logp, value))
    return repaired


def packed_policy_arrays(packed: dict) -> dict[str, np.ndarray]:
    """Return zero-copy NumPy views of one packed engine payload.

    The Rust binding returns raw byte buffers to keep the Python ABI simple.
    Parse each buffer exactly once, preserving its batch dimension; native
    rollouts pass these arrays directly to ``sample_packed_batch`` rather than
    materializing one observation dictionary per player state.
    """
    states = int(packed["states"])
    shapes = {
        "tok_cat": (states, T, 3),
        "tok_num": (states, T, N_TOK),
        "mv_cat": (states, T, 4),
        "mv_dyn": (states, T, 4, 2),
        "dmg": (states, 4, 4, 2, N_DMG),
        "glob": (states, N_GLOB),
        "act_tok": (states, 4),
        "mask": (states, 2, N_ACT),
        "ser_tok": (states, T, N_SER_TOK),
        "ser_mv": (states, T, 4, N_SER_MV),
        "ser_glob": (states, N_SER_GLOB),
    }
    dtypes = {
        "tok_cat": np.int32, "tok_num": np.float16, "mv_cat": np.int32,
        "mv_dyn": np.float16, "dmg": np.float16, "glob": np.float16,
        "act_tok": np.int8, "mask": np.bool_, "ser_tok": np.float16,
        "ser_mv": np.float16, "ser_glob": np.float16,
    }
    # Keys whose only legitimate drift is the trailing feature axis: the engine
    # may emit more or fewer per-slot features than the trained model reads
    # (e.g. a newer .so adds damage categories, or the ladder's zero-filled
    # series buffers are sized by a newer engine's constants). Slice or zero-pad
    # the last axis to the model's width; the leading structure stays aligned.
    trailing_axis_keys = {"dmg", "ser_tok", "ser_mv", "ser_glob"}
    out = {}
    mismatches = {}
    for key, shape in shapes.items():
        if key not in packed:
            continue
        arr = np.frombuffer(packed[key], dtype=dtypes[key])
        if states > 0 and key in trailing_axis_keys:
            mid = int(np.prod(shape[1:-1]))  # per-state size excluding the feature axis
            expected_last = shape[-1]
            if mid > 0 and arr.size % (states * mid) == 0:
                incoming_last = arr.size // (states * mid)
                if incoming_last != expected_last:
                    tensor = arr.reshape((states, *shape[1:-1], incoming_last))
                    if incoming_last < expected_last:
                        pad = [(0, 0)] * (tensor.ndim - 1) + [(0, expected_last - incoming_last)]
                        tensor = np.pad(tensor, pad)
                    else:
                        tensor = tensor[..., :expected_last]
                    out[key] = tensor
                    continue
        # Structural keys (tok_cat, tok_num, mv_cat, mv_dyn, glob, act_tok, mask):
        # any size drift means a genuine engine/Python contract break, not a
        # tolerable feature-axis difference. Collect every offender for one
        # actionable error rather than crashing on whichever reshapes first.
        per_state = int(np.prod(shape[1:]))
        expected = states * per_state
        if states > 0 and per_state > 0 and arr.size != expected:
            ratio = arr.size / expected if expected else float("inf")
            mismatches[key] = (arr.size, states, per_state, ratio)
            continue
        out[key] = arr.reshape(shape)
    if mismatches:
        detail = ", ".join(
            f"{key}: buffer={size} states={states} per_state={per_state} ratio={ratio:.3f}"
            for key, (size, states, per_state, ratio) in mismatches.items()
        )
        raise ValueError(f"packed_policy_arrays size mismatch ({detail})")
    return out


def _merge_rollouts(d_a, s_a, d_b, s_b):
    """Concatenate two homogeneous persistent-league rollouts (an open- and a
    closed-team-sheet half) into one training batch. Trajectory rows are
    independent decisions, so a straight row concatenation is valid; counts in
    the stats are summed and rates re-weighted by their game counts."""
    data = {k: np.concatenate([d_a[k], d_b[k]]) for k in d_a}
    ga, gb = s_a["completed_games"], s_b["completed_games"]

    def wmean(a, na, b, nb):
        return (a * na + b * nb) / (na + nb) if (na + nb) else 0.5

    stats = {k: (s_a[k] + s_b[k]) for k in s_a}  # default: sum counts
    stats["waves"] = s_a["waves"]                # shared horizon, not additive
    stats["mean_turns"] = wmean(s_a["mean_turns"], ga, s_b["mean_turns"], gb)
    stats["learner_p1_reward"] = wmean(s_a["learner_p1_reward"], s_a["learner_p1_games"],
                                       s_b["learner_p1_reward"], s_b["learner_p1_games"])
    stats["learner_p2_reward"] = wmean(s_a["learner_p2_reward"], s_a["learner_p2_games"],
                                       s_b["learner_p2_reward"], s_b["learner_p2_games"])
    stats["elapsed_s"] = s_a["elapsed_s"] + s_b["elapsed_s"]
    stats["games_s"] = (ga + gb) / stats["elapsed_s"] if stats["elapsed_s"] else 0.0
    stats["policy_version"] = max(s_a["policy_version"], s_b["policy_version"])
    return data, stats


def packed_policy_rows(packed: dict) -> list[dict[str, np.ndarray]]:
    """Legacy row adapter for non-batched callers.

    ``NativeBatchSelfPlay`` deliberately does not use this: constructing these
    dictionaries is exactly the Python overhead its packed route avoids.
    """
    arrays = packed_policy_arrays(packed)
    return [{key: value[i] for key, value in arrays.items()} for i in range(len(next(iter(arrays.values()))))]


def encode_vgc_obs(obs: dict, rating: float = 0.0, preview: bool = False) -> dict[str, np.ndarray]:
    """Encode an engine observation into the existing Policy tensor contract.

    Damage features remain zero during the first native rollout stage: the old
    calculator depends on a poke-env battle object.  Every other feature comes
    directly from vgc-engine's authoritative observation plus static GenData.
    """
    v = vocab()
    tok_cat = np.zeros((T, 3), np.int32)
    tok_num = np.zeros((T, N_TOK), np.float16)
    mv_cat = np.zeros((T, 4), np.int32)
    mv_dyn = np.zeros((T, 4, 2), np.float16)
    dmg = np.zeros((4, 4, 2, N_DMG), np.float16)
    act_tok = np.full(4, -1, np.int8)

    for side_i, side in enumerate(obs["sides"]):
        mons = [*side["active"], *side["bench"]]
        for local_i, mon in enumerate(mons[:6]):
            i = side_i * 6 + local_i
            active_slot = mon.get("slot") if local_i < len(side["active"]) else None
            act_slot = side_i * 2 + active_slot if active_slot is not None else None
            if act_slot is not None:
                act_tok[act_slot] = i
            tok_cat[i] = [
                v["species"].get(_id(mon["species"]), 0), v["items"].get(_id(mon.get("item")), 0),
                v["abilities"].get(_id(mon.get("ability")), 0),
            ]
            tok_num[i] = _tok_num(mon, side_i == 0, active_slot)
            for k, move in enumerate(mon.get("moves", [])[:4]):
                mv_cat[i, k] = v["moves"].get(_id(move["id"]), 0)
                mv_dyn[i, k] = [float(move.get("pp", 0)) / max(int(move.get("max_pp", 1)), 1), 0.0]

    glob = np.zeros(N_GLOB, np.float16)
    weather = _WEATHER.get(obs.get("weather", {}).get("kind", ""), "")
    glob[:len(WEATHERS)] = [float(weather == w.name.lower()) for w in WEATHERS]
    o = len(WEATHERS)
    terrain = _TERRAIN.get(obs.get("terrain", {}).get("kind", ""), "")
    field = obs.get("field", {})
    glob[o:o + len(FIELDS)] = [
        float(terrain == _FIELD[f.name.lower()]) if "terrain" in f.name.lower()
        else float(field.get(_FIELD[f.name.lower()], 0))
        for f in FIELDS
    ]
    o += len(FIELDS)
    for side in obs["sides"]:
        cond = side["conditions"]
        glob[o:o + len(SIDE)] = [float(cond.get(_SIDE.get(s.name.lower(), ""), 0)) for s in SIDE]
        o += len(SIDE)
    ours, foes = obs["sides"]
    faint = lambda side: sum(m.get("fainted", False) for m in [*side["active"], *side["bench"]])
    glob[o:] = [min(int(obs.get("turn", 0)), 20) / 20, 0.0, rating / 2000, 0.0, 0.0,
                faint(foes) / 4 - faint(ours) / 4]
    return {
        "tok_cat": tok_cat, "tok_num": tok_num, "mv_cat": mv_cat, "mv_dyn": mv_dyn,
        "dmg": dmg, "glob": glob, "act_tok": act_tok,
        "mask": preview_mask(obs) if preview else np.zeros((2, N_ACT), bool),
        "ser_tok": np.zeros((T, N_SER_TOK), np.float16),
        "ser_mv": np.zeros((T, 4, N_SER_MV), np.float16),
        "ser_glob": np.zeros(N_SER_GLOB, np.float16),
    }


def _action_index(choice: tuple, actor_side: int = 0) -> int | None:
    """Map an engine choice into the side-relative historical action id."""
    kind, _actor, slot, target_side, target_slot = choice
    if kind == "pass":
        return 0
    if kind == "switch":
        return slot + 1
    if kind not in {"move", "mega"}:
        return None
    target = 0 if target_side < 0 else (
        target_slot + 1 if target_side != actor_side else -(target_slot + 1)
    )
    # Core represents Struggle as synthetic move slot 4 while the historical
    # 107-way head only has four move planes. It is exclusive with ordinary
    # moves, so share plane 0 without colliding with another legal action.
    if kind == "move" and slot == 4:
        slot = 0
    # Match encode.legal_mask(): index = 7 + 5 * move + showdown_target + 2.
    # Targets are foe 1/2, ally -1/-2, or 0 for self/spread/no-target moves.
    if target not in {-2, -1, 0, 1, 2}:
        return None
    base = 9 + 5 * slot + target
    return base + (MEGA_OFFSET if kind in {"mega", "tera"} else 0)


def legal_mask_choices(choices: list[list[tuple]], actor_side: int = 0) -> np.ndarray:
    """Project one side's two engine choice lists into side-relative policy masks."""
    mask = np.zeros((2, N_ACT), bool)
    for slot, legal in enumerate(choices[:2]):
        for choice in legal:
            idx = _action_index(choice, actor_side)
            if idx is not None:
                mask[slot, idx] = True
        if not mask[slot].any():
            mask[slot, 0] = True
    return mask


def legal_mask_vgc(battle: Any, side: int = 0) -> np.ndarray:
    """Project engine legal choices into the existing two-slot action space."""
    return legal_mask_choices([battle.legal_choices(side, slot) for slot in range(2)], side)


def action_from_choices(choices: list[tuple], action: int, actor_side: int = 0) -> tuple:
    """Resolve a side-relative action id to one exact legal engine tuple."""
    wanted = [c for c in choices if _action_index(c, actor_side) == int(action)]
    if wanted:
        return next((c for c in wanted if c[0] == "mega"), wanted[0])
    passes = [c for c in choices if c[0] == "pass"]
    return passes[0] if passes else choices[0]


def action_choice(battle: Any, side: int, slot: int, action: int) -> tuple:
    """Resolve a policy action to the exact currently legal engine tuple."""
    return action_from_choices(battle.legal_choices(side, slot), action, side)


@dataclass
class NativeVGCConfig:
    fmt: str
    teams_dir: str | Path | None = None
    # A fixed learner team turns a shared pool into a specialist curriculum.
    # Opponent teams may be supplied separately for a native policy league.
    learner_team: str | Path | None = None
    opponent_teams_dir: str | Path | None = None
    # A single fixed opposing team for native specialist curriculum games.
    # This is distinct from `opponent_teams_dir`: choosing a random team from
    # a folder would silently erase the specialist matchup.
    opponent_team: str | Path | None = None
    rating: float = 1700.0
    greedy: bool = False
    tera_allowed: bool = False
    max_turns: int = 128
    team_preview: bool = False
    bo3: bool = False
    ots: bool = False # Open Team Sheets: benched mons expose moves/items/abilities
    # Fraction of persistent Bo1 slots trained with Open Team Sheets while the
    # rest stay closed (cts_version). >0 makes one net learn both views at once
    # so it never collapses when the live format differs from training — the
    # open/closed split that sank the separate-sheet v8-scale models. 0 keeps
    # the whole batch on `ots`.
    ots_mix: float = 0.0
    cts_version: int = 2
    # Persistent normal waves have a stable batch shape, so callers may opt
    # into compiled inference after measuring it on their target accelerator.
    compile_inference: bool = False
    # Benchmark-only: record final engine turns after a rollout. This avoids an
    # extra Python observation conversion on the RL collection path.
    collect_rollout_stats: bool = False
    seed: int = 0


class NativeSelfPlay:
    """Synchronous pure-vgc-engine self-play collector for existing PPO."""

    def __init__(self, config: NativeVGCConfig):
        try:
            import vgc_engine
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("native RL needs the installed vgc_engine package") from e
        self.engine = vgc_engine
        self.config = config
        self.teams = _pool(config.teams_dir)
        self.learner_team = (
            _team_json(Path(config.learner_team).read_text(), config.fmt) if config.learner_team else None
        )
        self.opponent_team = (
            _team_json(Path(config.opponent_team).read_text(), config.fmt) if config.opponent_team else None
        )
        self.opponent_teams = _pool(config.opponent_teams_dir) if config.opponent_teams_dir else self.teams
        self.rng = random.Random(config.seed)

    def _battle(self):
        return self.engine.Battle.from_teams(
            _team_json(self.rng.choice(self.teams), self.config.fmt), _team_json(self.rng.choice(self.teams), self.config.fmt),
            format=self.config.fmt, seed=self.rng.randrange(2**63), tera_allowed=self.config.tera_allowed,
            decision_phases=True,
        )

    def _replacement_choices(self, battle: Any, side: int) -> list[tuple]:
        """Pick distinct legal replacement switches without consulting the policy.

        Replacements occur after a completed turn, not as a policy decision in
        the current 107-action training contract.  Random distinct choices
        keep the simulator progressing while a later action-space revision can
        make these decisions learnable.
        """
        used: set[int] = set()
        choices: list[tuple] = []
        for slot in range(2):
            legal = battle.legal_choices(side, slot)
            switches = [c for c in legal if c[0] == "switch" and c[2] not in used]
            choice = self.rng.choice(switches) if switches else next((c for c in legal if c[0] == "pass"), legal[0])
            if choice[0] == "switch":
                used.add(choice[2])
            choices.append(choice)
        return choices

    def play(self, model: torch.nn.Module, games: int) -> tuple[list[tuple[list[dict], float]], list[float]]:
        episodes: list[tuple[list[dict], float]] = []
        rewards: list[float] = []
        for _ in range(games):
            battle = self._battle()
            ours: list[dict] = []
            theirs: list[dict] = []
            terminal = "tie"
            for _turn in range(self.config.max_turns):
                full = battle.observe()
                o0 = encode_vgc_obs(full, self.config.rating)
                o1 = encode_vgc_obs(perspective_obs(full, 1), self.config.rating)
                o0["mask"] = legal_mask_vgc(battle, 0)
                o1["mask"] = legal_mask_vgc(battle, 1)
                (a0, lp0, v0), (a1, lp1, v1) = sample_batch(model, [o0, o1], [self.config.greedy] * 2)
                c0 = [action_choice(battle, 0, slot, int(a0[slot])) for slot in range(2)]
                c1 = [action_choice(battle, 1, slot, int(a1[slot])) for slot in range(2)]
                ours.append(o0 | {"action": a0.astype(np.int16), "logp": lp0, "value": v0, "preview": False})
                theirs.append(o1 | {"action": a1.astype(np.int16), "logp": lp1, "value": v1, "preview": False})
                terminal = battle.step_move(c0, c1)
                if terminal != "continue":
                    break
                while battle.needs_replacements():
                    terminal = battle.step_move(self._replacement_choices(battle, 0), self._replacement_choices(battle, 1))
                    if terminal != "continue":
                        break
                if terminal != "continue":
                    break
            r0 = 1.0 if terminal == "p1_win" else 0.0 if terminal == "p2_win" else 0.5
            r1 = 1.0 - r0 if r0 != 0.5 else 0.5
            episodes.extend(((ours, r0), (theirs, r1)))
            rewards.append(r0)
        return episodes, rewards


class NativeBatchSelfPlay(NativeSelfPlay):
    """Native vectorized collector backed by Rust-owned ``BattleBatch`` storage."""

    def _new_slot_payloads(
        self,
        indices: np.ndarray,
        learner_sides: np.ndarray | None = None,
    ) -> tuple[list[str], list[str], list[int]]:
        """Sample matchups/RNG streams, pinning the learner team when requested."""
        indices = np.asarray(indices, dtype=np.intp)
        if learner_sides is not None and len(learner_sides) != len(indices):
            raise ValueError("learner side assignments must align with reset slots")
        p1: list[str] = []
        p2: list[str] = []
        for row, _slot in enumerate(indices):
            learner_team = self.learner_team or _team_json(self.rng.choice(self.teams), self.config.fmt)
            opponent = self.opponent_team or _team_json(self.rng.choice(self.opponent_teams), self.config.fmt)
            if learner_sides is None:
                p1.append(_team_json(self.rng.choice(self.teams), self.config.fmt))
                p2.append(opponent)
            elif int(learner_sides[row]) == 0:
                p1.append(learner_team)
                p2.append(opponent)
            else:
                p1.append(opponent)
                p2.append(learner_team)
        return p1, p2, [self.rng.randrange(2**63) for _ in indices]

    def _reset_slots(
        self,
        battles: Any,
        indices: np.ndarray,
        learner_sides: np.ndarray | None = None,
        reuse_series: np.ndarray | None = None,
    ) -> None:
        """Refill slots, retaining their team pair for games within a Bo3."""
        indices = np.asarray(indices, dtype=np.intp)
        if not len(indices):
            return
        reuse = np.zeros(len(indices), dtype=bool) if reuse_series is None else np.asarray(reuse_series, dtype=bool)
        if len(reuse) != len(indices):
            raise ValueError("series reuse flags must align with reset slots")
        fresh_rows = np.flatnonzero(~reuse)
        p1: list[str] = [""] * len(indices)
        p2: list[str] = [""] * len(indices)
        if len(fresh_rows):
            fresh_indices = indices[fresh_rows]
            fresh_sides = learner_sides[fresh_rows] if learner_sides is not None else None
            fresh_p1, fresh_p2, _ = self._new_slot_payloads(fresh_indices, fresh_sides)
            for row, team1, team2 in zip(fresh_rows, fresh_p1, fresh_p2):
                p1[int(row)], p2[int(row)] = team1, team2
                self._slot_teams[int(indices[row])] = (team1, team2)
        for row in np.flatnonzero(reuse):
            p1[int(row)], p2[int(row)] = self._slot_teams[int(indices[row])]
        seeds = [self.rng.randrange(2**63) for _ in indices]
        battles.reset_slots(indices.tolist(), p1, p2, seeds)

    def _apply_series_context(
        self,
        packed: dict[str, np.ndarray],
        wins: np.ndarray,
        last_winner: np.ndarray,
        game_number: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Overlay per-perspective Bo3 score features on the engine's zero buffers."""
        if not self.config.bo3:
            return packed
        result = dict(packed)
        series = np.zeros_like(packed["ser_glob"])
        series[0::2, 0] = game_number > 1
        series[0::2, 1] = game_number > 2
        series[0::2, 2] = wins[:, 0] / 2.0
        series[0::2, 3] = wins[:, 1] / 2.0
        series[0::2, 4] = last_winner == 0
        series[1::2, 0] = game_number > 1
        series[1::2, 1] = game_number > 2
        series[1::2, 2] = wins[:, 1] / 2.0
        series[1::2, 3] = wins[:, 0] / 2.0
        series[1::2, 4] = last_winner == 1
        result["ser_glob"] = series.astype(np.float16, copy=False)
        return result

    def _sample_packed(
        self,
        model: torch.nn.Module,
        packed: dict[str, np.ndarray],
        game_indices: np.ndarray,
        compile_inference: bool | None = None,
    ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
        """Sample both player perspectives in one batched policy forward pass.

        Only full normal persistent waves use compilation: reset preview and
        replacement subsets deliberately keep eager inference because their
        cardinality changes with game outcomes.
        """
        game_indices = np.asarray(game_indices, dtype=np.intp)
        rows = np.empty(len(game_indices) * 2, dtype=np.intp)
        rows[::2] = 2 * game_indices
        rows[1::2] = rows[::2] + 1
        if compile_inference is None:
            compile_inference = self.config.compile_inference
        return sample_packed_batch(
            model, packed, rows, greedy=self.config.greedy,
            use_fp16=True, use_compile=compile_inference,
        )

    @staticmethod
    def _append_preview_transitions(
        stores: dict[str, list[np.ndarray]],
        episode_rows: list[list[int]],
        game_indices: np.ndarray,
        obs: dict[str, np.ndarray],
        actions: np.ndarray,
        logp: np.ndarray,
        values: np.ndarray,
        side: int,
    ) -> None:
        """Append one packed bring-four decision for one player perspective."""
        NativeBatchSelfPlay._append_packed_transitions(
            stores, episode_rows, game_indices, obs, actions, logp, values, side,
            preview=True,
        )

    def _select_preview_packed(
        self,
        battles: Any,
        model: torch.nn.Module,
        games: int,
        stores: dict[str, list[np.ndarray]],
        episode_rows: list[list[int]],
    ) -> None:
        """Sample native packed bring-four choices without Python row encoding.

        The engine's pre-turn policy rows retain the six-member team ordering:
        actions 1..6 therefore map directly to the indices accepted by
        ``select_teams``. Each policy decision selects a pair; after two
        decisions per side, only the one-time ``select_teams`` boundary needs
        nested Python lists.
        """
        game_indices = np.arange(games, dtype=np.intp)
        source = packed_policy_arrays(
            battles.policy_tensors(self.config.rating, preview=True, ots=self.config.ots, cts_version=self.config.cts_version)
        )
        states = games * 2
        selected = np.empty((states, 0), dtype=np.uint8)

        for _ in range(2):
            preview = dict(source)
            preview["tok_num"] = source["tok_num"].copy()
            preview["tok_num"][:, :6, 76] = 0
            for prior in selected.T:
                preview["tok_num"][np.arange(states), prior - 1, 76] = 1
            mask = np.zeros((states, 2, N_ACT), dtype=bool)
            mask[:, :, 1:7] = True
            for prior in selected.T:
                mask[np.arange(states), :, prior] = False
            preview["mask"] = mask
            obs, actions, logp, values = self._sample_packed(
                model, preview, game_indices, compile_inference=False,
            )
            if np.any((actions < 1) | (actions > 6)) or np.any(actions[:, 0] == actions[:, 1]):
                raise RuntimeError("packed preview emitted an invalid or duplicate team slot")
            self._append_preview_transitions(
                stores, episode_rows, game_indices, obs, actions, logp, values, 0,
            )
            self._append_preview_transitions(
                stores, episode_rows, game_indices, obs, actions, logp, values, 1,
            )
            selected = np.concatenate((selected, actions), axis=1)

        battles.select_teams(
            (selected[::2] - 1).tolist(),
            (selected[1::2] - 1).tolist(),
        )


    def _select_preview_packed_indices(
        self,
        battles: Any,
        model: torch.nn.Module,
        game_indices: np.ndarray,
        stores: dict[str, list[np.ndarray]],
        episode_rows: list[list[int]],
        wins: np.ndarray | None = None,
        last_winner: np.ndarray | None = None,
        game_number: np.ndarray | None = None,
    ) -> None:
        """Run packed bring-four preview for selected fresh persistent slots only."""
        game_indices = np.asarray(game_indices, dtype=np.intp)
        if not len(game_indices):
            return
        source = packed_policy_arrays(
            battles.policy_tensors(self.config.rating, preview=True, indices=game_indices.tolist(), ots=self.config.ots, cts_version=self.config.cts_version)
        )
        if wins is not None and last_winner is not None and game_number is not None:
            source = self._apply_series_context(source, wins[game_indices], last_winner[game_indices], game_number[game_indices])
        states = len(game_indices) * 2
        selected = np.empty((states, 0), dtype=np.uint8)
        local_rows = np.arange(states, dtype=np.intp)
        local_game_indices = np.arange(len(game_indices), dtype=np.intp)
        for _ in range(2):
            preview = dict(source)
            preview["tok_num"] = source["tok_num"].copy()
            preview["tok_num"][:, :6, 76] = 0
            for prior in selected.T:
                preview["tok_num"][np.arange(states), prior - 1, 76] = 1
            mask = np.zeros((states, 2, N_ACT), dtype=bool)
            mask[:, :, 1:7] = True
            for prior in selected.T:
                mask[local_rows, :, prior] = False
            preview["mask"] = mask
            obs, actions, logp, values = self._sample_packed(
                model, preview, local_game_indices, compile_inference=False,
            )
            if np.any((actions < 1) | (actions > 6)) or np.any(actions[:, 0] == actions[:, 1]):
                raise RuntimeError("packed preview emitted an invalid or duplicate team slot")
            self._append_preview_transitions(
                stores, episode_rows, game_indices, obs, actions, logp, values, 0,
            )
            self._append_preview_transitions(
                stores, episode_rows, game_indices, obs, actions, logp, values, 1,
            )
            selected = np.concatenate((selected, actions), axis=1)
        battles.select_teams_at(
            game_indices.tolist(), (selected[::2] - 1).tolist(), (selected[1::2] - 1).tolist(),
        )

    @staticmethod
    def _append_packed_transitions(
        stores: dict[str, list[np.ndarray]],
        episode_rows: list[list[int]],
        game_indices: np.ndarray,
        obs: dict[str, np.ndarray],
        actions: np.ndarray,
        logp: np.ndarray,
        values: np.ndarray,
        side: int,
        preview: bool = False,
    ) -> None:
        """Append compact rollout arrays; defer Python record objects to the end."""
        state_rows = np.arange(len(game_indices), dtype=np.intp) * 2 + side
        for key in _PACKED_KEYS:
            stores[key].append(obs[key][state_rows])
        stores["action"].append(actions[state_rows].astype(np.int16, copy=False))
        stores["logp"].append(logp[state_rows].astype(np.float32, copy=False))
        stores["value"].append(values[state_rows].astype(np.float32, copy=False))
        stores["preview"].append(np.full(len(state_rows), preview, dtype=bool))
        total_rows = stores.setdefault("_total_rows", [0])
        base_row = total_rows[0]
        total_rows[0] += len(state_rows)
        for game, row in zip(game_indices, range(base_row, base_row + len(state_rows))):
            episode_rows[2 * int(game) + side].append(row)

    @staticmethod
    def _rollout_episodes(
        stores: dict[str, list[np.ndarray]],
        episode_rows: list[list[int]],
        prefixes: list[list[dict]],
        terminal: list[str],
    ) -> tuple[list[tuple[list[dict], float]], list[float]]:
        """Convert compact native rollout stores to the legacy PPO contract."""
        values = {
            key: np.concatenate(parts, axis=0)
            for key, parts in stores.items()
            if parts and key != "_total_rows"
        }
        episodes: list[tuple[list[dict], float]] = []
        rewards: list[float] = []
        for game, result in enumerate(terminal):
            r0 = 1.0 if result == "p1_win" else 0.0 if result == "p2_win" else 0.5
            r1 = 1.0 - r0 if r0 != 0.5 else 0.5
            for side, reward in ((0, r0), (1, r1)):
                episode = list(prefixes[2 * game + side])
                episode.extend(
                    {**{key: values[key][row] for key in _PACKED_KEYS},
                     "action": values["action"][row],
                     "logp": float(values["logp"][row]),
                     "value": float(values["value"][row]),
                     "preview": bool(values["preview"][row])}
                    for row in episode_rows[2 * game + side]
                )
                if episode:
                    episodes.append((episode, reward))
            rewards.append(r0)
        return episodes, rewards


    @staticmethod
    def _rollout_data(
        stores: dict[str, list[np.ndarray]],
        episode_rows: list[list[int]],
        terminal: list[str],
    ) -> tuple[dict[str, np.ndarray], list[float]]:
        """Build PPO-ready arrays without per-transition Python dictionaries.

        Rows are gathered in the existing episode-major order so GAE and PPO
        retain their established assumptions: player one then player two for
        each game, with `last` marking each individual player episode.
        """
        values = {key: np.concatenate(parts, axis=0) for key, parts in stores.items() if parts and key != "_total_rows"}
        order: list[int] = []
        rewards: list[float] = []
        row_rewards: list[float] = []
        last: list[bool] = []
        for game, result in enumerate(terminal):
            r0 = 1.0 if result == "p1_win" else 0.0 if result == "p2_win" else 0.5
            r1 = 1.0 - r0 if r0 != 0.5 else 0.5
            rewards.append(r0)
            for side, reward in ((0, r0), (1, r1)):
                rows = episode_rows[2 * game + side]
                order.extend(rows)
                row_rewards.extend([0.0] * len(rows))
                last.extend([False] * len(rows))
                if rows:
                    row_rewards[-1] = reward
                    last[-1] = True
        index = np.asarray(order, dtype=np.intp)
        return (
            {
                **{key: values[key][index] for key in _PACKED_KEYS},
                "action": values["action"][index],
                "logp": values["logp"][index],
                "value": values["value"][index],
                "preview": values["preview"][index],
                "reward": np.asarray(row_rewards, dtype=np.float32),
                "last": np.asarray(last, dtype=bool),
            },
            rewards,
        )


    @staticmethod
    def _completed_rollout_data(
        stores: dict[str, list[np.ndarray]],
        completed: list[tuple[list[int], float]],
    ) -> tuple[dict[str, np.ndarray], list[float]]:
        """Compact terminal persistent-actor episodes into PPO's legacy layout.

        A persistent slot may host many games. `completed` therefore owns the
        exact global transition rows for each finished player episode, rather
        than assuming a slot corresponds to one episode for the whole actor
        horizon. Keeping only terminal episodes lets the established terminal
        GAE remain mathematically unchanged.
        """
        if not completed:
            empty = {
                "tok_cat": np.empty((0, T, 3), dtype=np.uint32),
                "tok_num": np.empty((0, T, N_TOK), dtype=np.float16),
                "mv_cat": np.empty((0, T, 4), dtype=np.uint32),
                "mv_dyn": np.empty((0, T, 4, 2), dtype=np.float16),
                "dmg": np.empty((0, 4, 4, 2, N_DMG), dtype=np.float16),
                "glob": np.empty((0, N_GLOB), dtype=np.float16),
                "act_tok": np.empty((0, 4), dtype=np.int8),
                "mask": np.empty((0, 2, N_ACT), dtype=bool),
                "ser_tok": np.empty((0, T, N_SER_TOK), dtype=np.float16),
                "ser_mv": np.empty((0, T, 4, N_SER_MV), dtype=np.float16),
                "ser_glob": np.empty((0, N_SER_GLOB), dtype=np.float16),
                "action": np.empty((0, 2), dtype=np.int16),
                "logp": np.empty((0,), dtype=np.float32),
                "value": np.empty((0,), dtype=np.float32),
                "preview": np.empty((0,), dtype=bool),
                "reward": np.empty((0,), dtype=np.float32),
                "last": np.empty((0,), dtype=bool),
            }
            return empty, []
        values = {key: np.concatenate(parts, axis=0) for key, parts in stores.items() if parts and key != "_total_rows"}
        order: list[int] = []
        rewards: list[float] = []
        row_rewards: list[float] = []
        last: list[bool] = []
        for rows, reward in completed:
            if not rows:
                continue
            order.extend(rows)
            rewards.append(reward)
            row_rewards.extend([0.0] * len(rows))
            last.extend([False] * len(rows))
            row_rewards[-1] = reward
            last[-1] = True
        index = np.asarray(order, dtype=np.intp)
        return ({
            **{key: values[key][index] for key in _PACKED_KEYS},
            "action": values["action"][index],
            "logp": values["logp"][index],
            "value": values["value"][index],
            "preview": values["preview"][index],
            "reward": np.asarray(row_rewards, dtype=np.float32),
            "last": np.asarray(last, dtype=bool),
        }, rewards)

    @staticmethod
    def _merge_step_results(terminal: list[str], raw: bytes) -> np.ndarray:
        """Merge compact Rust status codes into the persistent batch state."""
        outcomes = ("inactive", "continue", "p1_win", "p2_win", "tie")
        raw_arr = np.frombuffer(raw, dtype=np.uint8)
        if len(raw_arr) != len(terminal):
            raise RuntimeError(f"native step returned {len(raw_arr)} results for {len(terminal)} battles")
        for game in np.flatnonzero(raw_arr > 1):
            code = int(raw_arr[game])
            if code >= len(outcomes):
                raise RuntimeError(f"native step returned unknown result code {code}")
            terminal[game] = outcomes[code]
        return np.asarray([result == "continue" for result in terminal], dtype=bool)

    def _step_packed(
        self,
        battles: Any,
        p1_actions: np.ndarray,
        p2_actions: np.ndarray,
        game_indices: np.ndarray,
        terminal: list[str],
    ) -> np.ndarray:
        raw = battles.step_policy_action_array(
            np.ascontiguousarray(p1_actions, dtype=np.uint8),
            np.ascontiguousarray(p2_actions, dtype=np.uint8),
            np.ascontiguousarray(game_indices, dtype=np.int64),
        )
        return self._merge_step_results(terminal, raw)

    def _replacement_batch(
        self,
        battles: Any,
        active: np.ndarray,
        terminal: list[str],
        stores: dict[str, list[np.ndarray]],
        episode_rows: list[list[int]],
        model: torch.nn.Module,
        wins: np.ndarray | None = None,
        last_winner: np.ndarray | None = None,
        game_number: np.ndarray | None = None,
    ) -> tuple[np.ndarray, int]:
        """Record and execute policy-selected forced switches for all live games."""
        pending: list[int] | None = np.flatnonzero(active).tolist() if not active.all() else None
        replacement_count = 0
        has_indices_api = hasattr(battles, "needs_replacements_indices")
        while True:
            if has_indices_api:
                indices = battles.needs_replacements_indices(pending)
            else:
                replacing = np.asarray(battles.needs_replacements(), dtype=bool)
                indices = [i for i in pending if replacing[i]] if pending is not None else np.flatnonzero(active & replacing).tolist()
            if not indices:
                break
            replacement_count += len(indices)
            packed = packed_policy_arrays(
                battles.policy_tensors(self.config.rating, indices=indices, ots=self.config.ots, cts_version=self.config.cts_version)
            )
            if wins is not None and last_winner is not None and game_number is not None:
                packed = self._apply_series_context(packed, wins[np.asarray(indices)], last_winner[np.asarray(indices)], game_number[np.asarray(indices)])
            obs, actions, logp, values = self._sample_packed(
                model, packed, np.arange(len(indices)), compile_inference=False,
            )
            idx_arr = np.asarray(indices, dtype=np.int64)
            self._append_packed_transitions(stores, episode_rows, idx_arr, obs, actions, logp, values, 0)
            self._append_packed_transitions(stores, episode_rows, idx_arr, obs, actions, logp, values, 1)
            active = self._step_packed(
                battles, actions[::2], actions[1::2], idx_arr, terminal,
            )
            pending = [i for i in indices if active[i]]
            if not pending:
                break
        return active, replacement_count

    def _battles(self, games: int, learner_sides: np.ndarray | None = None):
        slots = np.arange(games, dtype=np.intp)
        p1, p2, seeds = self._new_slot_payloads(slots, learner_sides)
        battles = self.engine.BattleBatch.from_teams(
            p1, p2, format=self.config.fmt,
            seeds=seeds,
            tera_allowed=self.config.tera_allowed, decision_phases=True,
        )
        self._slot_teams = dict(enumerate(zip(p1, p2)))
        tables = policy_vocab_tables()
        battles.set_policy_vocab(tables["species"], tables["items"], tables["abilities"], tables["moves"])
        if not self.config.ots:
            usage = cts_usage_tables()
            battles.set_cts_usage(usage["items"], usage["abilities"], usage["moves"], usage["stats"])
        return battles

    def play(self, model: torch.nn.Module, games: int) -> tuple[list[tuple[list[dict], float]], list[float]]:
        if games < 1:
            return [], []
        battles = self._battles(games)
        # During play retain dense arrays, not a Python dict for every state.
        # The legacy episode representation is constructed once at the PPO
        # boundary in `_rollout_episodes`.
        stores = {key: [] for key in (*_PACKED_KEYS, "action", "logp", "value", "preview")}
        episode_rows: list[list[int]] = [[] for _ in range(2 * games)]
        prefixes: list[list[dict]] = [[] for _ in range(2 * games)]
        terminal = ["continue"] * games
        active = np.ones(games, bool)

        if self.config.team_preview:
            self._select_preview_packed(battles, model, games, stores, episode_rows)

        for _turn in range(self.config.max_turns):
            if not active.any():
                break
            indices = np.flatnonzero(active)
            packed = packed_policy_arrays(battles.policy_tensors(self.config.rating, ots=self.config.ots, cts_version=self.config.cts_version))
            obs, actions, logp, values = self._sample_packed(model, packed, indices)
            self._append_packed_transitions(stores, episode_rows, indices, obs, actions, logp, values, 0)
            self._append_packed_transitions(stores, episode_rows, indices, obs, actions, logp, values, 1)
            active = self._step_packed(
                battles, actions[::2], actions[1::2], indices, terminal,
            )
            active, _ = self._replacement_batch(battles, active, terminal, stores, episode_rows, model)

        return self._rollout_episodes(stores, episode_rows, prefixes, terminal)


    def play_data(self, model: torch.nn.Module, games: int) -> tuple[dict[str, np.ndarray] | None, list[float]]:
        """Play native self-play and return PPO arrays without legacy row dicts."""
        if games < 1:
            return None, []
        battles = self._battles(games)
        stores = {key: [] for key in (*_PACKED_KEYS, "action", "logp", "value", "preview")}
        episode_rows: list[list[int]] = [[] for _ in range(2 * games)]
        terminal = ["continue"] * games
        active = np.ones(games, bool)

        if self.config.team_preview:
            self._select_preview_packed(battles, model, games, stores, episode_rows)

        for _turn in range(self.config.max_turns):
            if not active.any():
                break
            indices = np.flatnonzero(active)
            packed = packed_policy_arrays(battles.policy_tensors(self.config.rating, ots=self.config.ots, cts_version=self.config.cts_version))
            obs, actions, logp, values = self._sample_packed(model, packed, indices)
            self._append_packed_transitions(stores, episode_rows, indices, obs, actions, logp, values, 0)
            self._append_packed_transitions(stores, episode_rows, indices, obs, actions, logp, values, 1)
            active = self._step_packed(battles, actions[::2], actions[1::2], indices, terminal)
            active, _ = self._replacement_batch(battles, active, terminal, stores, episode_rows, model)

        data, rewards = self._rollout_data(stores, episode_rows, terminal)
        if self.config.collect_rollout_stats:
            # `turn` is the authoritative completed battle-turn count. Unlike
            # PPO rows, it excludes forced replacement decision phases.
            data["battle_turns"] = np.asarray(
                [state["turn"] for state in battles.observe()], dtype=np.int16,
            )
        return data, rewards

    def collect_persistent_data(
        self,
        model: torch.nn.Module,
        slots: int,
        decision_horizon: int,
    ) -> tuple[dict[str, np.ndarray], dict[str, float | int]]:
        """Collect terminal episodes, preserving matchup and context across Bo3 games."""
        if slots < 1 or decision_horizon < 1:
            raise ValueError("persistent collection needs positive slots and decision_horizon")
        if not self.config.bo3:
            from ai_vgc.nn.native_rollout import Bo1Rollout
            if not hasattr(self, "_bo1_rollout"):
                self._bo1_rollout = Bo1Rollout(self, slots, league=False)
            if self._bo1_rollout.slots != slots or self._bo1_rollout.league:
                raise ValueError("Cannot change a live runner's slot count or league mode")
            return self._bo1_rollout.collect(model, model, decision_horizon)
        from ai_vgc.nn.native_rollout import Bo3Rollout
        if not hasattr(self, "_bo3_rollout"):
            self._bo3_rollout = Bo3Rollout(self, slots, league=False)
        if self._bo3_rollout.slots != slots or self._bo3_rollout.league:
            raise ValueError("Cannot change a live runner's slot count or league mode")
        return self._bo3_rollout.collect(model, model, decision_horizon)


    def collect_persistent_league_data(
        self,
        learner: torch.nn.Module,
        opponent: torch.nn.Module,
        slots: int,
        decision_horizon: int,
        learner_first: bool = True,
    ) -> tuple[dict[str, np.ndarray], dict[str, float | int]]:
        """Collect learner-only terminal episodes against a frozen native policy."""
        if slots < 1 or decision_horizon < 1:
            raise ValueError("persistent league collection needs positive slots and horizon")
        if not self.config.bo3:
            from ai_vgc.nn.native_rollout import Bo1Rollout
            mix = float(getattr(self.config, "ots_mix", 0.0) or 0.0)
            if 0.0 < mix < 1.0:
                # 50/50 open+closed curriculum via two homogeneous persistent
                # rollouts so one net learns both views without mixing sheet
                # modes inside a single native batch. Slots split once and stay
                # put; each half is a proven pure-open / pure-closed rollout.
                n_open = int(round(slots * mix))
                n_closed = slots - n_open
                if n_open < 1 or n_closed < 1:
                    raise ValueError(f"ots_mix={mix} with slots={slots} leaves an empty half")
                if not hasattr(self, "_bo1_open"):
                    self._bo1_open = Bo1Rollout(self, n_open, league=True, learner_first=learner_first, ots=True)
                    self._bo1_closed = Bo1Rollout(self, n_closed, league=True, learner_first=learner_first, ots=False)
                if self._bo1_open.slots != n_open or self._bo1_closed.slots != n_closed:
                    raise ValueError("Cannot change a live runner's slot split")
                d_o, s_o = self._bo1_open.collect(learner, opponent, decision_horizon)
                d_c, s_c = self._bo1_closed.collect(learner, opponent, decision_horizon)
                return _merge_rollouts(d_o, s_o, d_c, s_c)
            if not hasattr(self, "_bo1_rollout"):
                self._bo1_rollout = Bo1Rollout(self, slots, league=True, learner_first=learner_first)
            if self._bo1_rollout.slots != slots or not self._bo1_rollout.league:
                raise ValueError("Cannot change a live runner's slot count or league mode")
            return self._bo1_rollout.collect(learner, opponent, decision_horizon)
        from ai_vgc.nn.native_rollout import Bo3Rollout
        if not hasattr(self, "_bo3_rollout"):
            self._bo3_rollout = Bo3Rollout(self, slots, league=True, learner_first=learner_first)
        if self._bo3_rollout.slots != slots or not self._bo3_rollout.league:
            raise ValueError("Cannot change a live runner's slot count or league mode")
        return self._bo3_rollout.collect(learner, opponent, decision_horizon)
