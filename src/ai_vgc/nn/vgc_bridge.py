"""vgc-engine backend for ai-vgc one-turn search.

The live bot and PPO game runner remain poke-env/Showdown clients.  This
module replaces the *search playout* simulator only: it rebuilds the public
position prepared by ``nn.search.position``, advances a candidate joint turn
inside Rust, and returns the same lightweight result shape that search uses.

The boundary is deliberately small.  It lets us validate vgc-engine against
Showdown playouts before changing the live-game or neural-encoding contracts.
"""

from __future__ import annotations

import copy
import json
import random
import threading
from pathlib import Path
from typing import Any

from poke_env.data import to_id_str


# `position()` uses Showdown ids.  The Rust serde state stores core enum names.
_STATUS = {
    "": "None", "brn": "Burn", "par": "Paralysis", "psn": "Poison",
    "tox": "Toxic", "slp": "Sleep", "frz": "Freeze",
}
_WEATHER = {
    "": "None", "raindance": "Rain", "primordialsea": "Rain",
    "sunnyday": "Sun", "desolateland": "Sun", "sandstorm": "Sand",
    "snow": "Snow", "snowscape": "Snow", "hail": "Snow",
}
_TERRAIN = {
    "": "None", "electricterrain": "Electric", "grassyterrain": "Grassy",
    "psychicterrain": "Psychic", "mistyterrain": "Misty",
}


class VGCBridge:
    """In-process Rust playout backend compatible with one-turn ``Bridge.run``.

    Results contain ``value`` rather than Showdown protocol ``lines``.  The
    caller uses that value directly, which keeps this first migration isolated
    from poke-env's protocol parser and from the neural observation adapter.
    ``value`` is a bounded material/HP leaf until Phase B installs the neural
    observation encoder; terminal games are always returned exactly.
    """

    def __init__(self, timeout: float = 60.0, **_: Any):
        try:
            import vgc_engine
        except ImportError as e:  # pragma: no cover - environment error
            raise RuntimeError(
                "vgc-engine backend requires the project's vgc_engine package; "
                "run through .venv/bin/python or `uv run`."
            ) from e
        self.engine = vgc_engine
        self.timeout = timeout
        self.lock = threading.RLock()
        self._kept: dict[int, tuple[Any, bool]] = {}

    @staticmethod
    def _team(side: dict) -> str:
        """Convert search's Showdown-set dictionaries to ``Battle.from_teams`` JSON."""
        team = []
        for mon in side["mons"]:
            s = mon["set"]
            team.append({
                "species": s["species"], "level": s.get("level", 50),
                "ability": s.get("ability") or None, "item": mon.get("item", s.get("item")) or None,
                "nature": s.get("nature", "Serious"), "moves": s.get("moves", []),
                "evs": s.get("evs", {}), "ivs": s.get("ivs", {}),
            })
        return json.dumps(team)

    @staticmethod
    def _sides(state: dict) -> tuple[dict, dict, bool]:
        """Return engine p1/p2 side states and whether ``us`` is engine p1."""
        us_is_p1 = state["role"] == "p1"
        if us_is_p1:
            return state["sides"]["us"], state["sides"]["them"], True
        return state["sides"]["them"], state["sides"]["us"], False

    @staticmethod
    def _patch_side(raw: dict, source: dict, side: int = 0, rng=None) -> None:
        """Overlay the observable current state onto a newly built Rust battle.

        Search intentionally reconstructs only information its player knows.
        Hidden sleep/confusion durations are sampled conditional on elapsed
        attempts. Explicit remaining counters/HP, when available, take priority.
        """
        raw["active"] = [0, 1]
        rng = rng or random.Random(0)
        # ``ps_order`` is a fixed six-slot serialization array even when a
        # VGC side only brought four Pokémon; keep the engine-generated shape.
        conds = raw["conditions"]
        for src, dst in (
            ("tailwind", "tailwind_turns"), ("reflect", "reflect_turns"),
            ("lightscreen", "light_screen_turns"), ("auroraveil", "aurora_veil_turns"),
            ("safeguard", "safeguard_turns"), ("mist", "mist_turns"),
            ("spikes", "spikes_layers"), ("toxicspikes", "toxic_spikes_layers"),
        ):
            conds[dst] = int(source.get("conditions", {}).get(src, 0))
        conds["stealth_rock"] = bool(source.get("conditions", {}).get("stealthrock", False))
        conds["sticky_web"] = bool(source.get("conditions", {}).get("stickyweb", False))
        conds["mega_used"] = bool(source.get("megaUsed", False))

        for target, source_mon in zip(raw["team"], source["mons"]):
            # The local player's exact stats are part of `position`; preserve
            # them so the reconstructed engine state has the same HP scale and
            # speed/damage stats rather than merely an EV-derived approximation.
            for stat, value in source_mon.get("stats", {}).items():
                if stat in target["stats"] and value:
                    target["stats"][stat] = int(value)
            if source_mon.get("fainted"):
                target["current_hp"] = 0
                target["fainted"] = True
            elif "hp" in source_mon:
                target["current_hp"] = max(1, min(int(source_mon["hp"]), target["stats"]["hp"]))
                target["fainted"] = False
            else:
                frac = float(source_mon.get("hpFrac", 1.0))
                target["current_hp"] = max(1, min(round(frac * target["stats"]["hp"]), target["stats"]["hp"]))
                target["fainted"] = False
            target["boosts"] = [int(source_mon.get("boosts", {}).get(k, 0))
                                for k in ("atk", "def", "spa", "spd", "spe", "accuracy", "evasion")]
            target["status"] = _STATUS.get(source_mon.get("status", ""), "None")
            target["turns_active"] = 0 if source_mon.get("firstTurn") else 1
            moves = [to_id_str(m) for m in source_mon["set"].get("moves", [])]
            for slot, move in enumerate(moves):
                if move in source_mon.get("pp", {}):
                    target["pp"][slot] = max(0, min(int(source_mon["pp"][move]), target["pp"][slot]))
            last = source_mon.get("lastMove")
            target["last_used_move_slot"] = moves.index(last) if last in moves else 255
            target["must_recharge"] = bool(source_mon.get("mustRecharge", False))
            charging = source_mon.get("chargingMove")
            if charging:
                if charging not in moves:
                    raise ValueError(f"charging move {charging} absent from reconstructed set")
                target["charging_turns"] = 1
                target["charging_move_slot"] = moves.index(charging)
                target["semi_invuln"] = {"dig": 1, "dive": 2, "fly": 3, "bounce": 4,
                                         "phantomforce": 5, "shadowforce": 6, "skydrop": 7}.get(charging, 0)

            vols = target["volatiles"]
            vols["len"] = 0
            vols["present"] = 0  # from_json rehydrates the cache from items.

            def add(kind, turns=0, payload=0):
                index = vols["len"]
                if index >= len(vols["items"]):
                    raise ValueError("too many reconstructed volatiles")
                vols["items"][index] = dict(kind=kind, turns_remaining=int(turns), payload=int(payload))
                vols["len"] += 1

            elapsed = source_mon.get("volatileElapsed", {})
            for name in source_mon.get("volatiles", []):
                age = int(elapsed.get(name, 0))
                if name in ("perish1", "perish2", "perish3"):
                    add("PerishSong", int(name[-1]))
                elif name == "substitute":
                    add("Substitute", payload=source_mon.get("substituteHp", max(1, target["stats"]["hp"] // 4)))
                elif name == "confusion":
                    add("Confusion", payload=rng.randint(max(1, 2-age), max(1, 5-age)))
                elif name == "leechseed":
                    src_side, src_slot = source_mon.get("leechSeedSource", (1-side, 0))
                    add("LeechSeed", payload=(src_side << 8) | src_slot)
                elif name in ("taunt", "yawn", "healblock"):
                    kind, duration = {"taunt": ("Taunt", 3), "yawn": ("Yawn", 2),
                                      "healblock": ("HealBlock", 5)}[name]
                    add(kind, max(1, duration-age), (1-side) << 8 if name == "yawn" else 0)
                elif name in ("saltcure", "charge"):
                    add({"saltcure": "SaltCure", "charge": "Charge"}[name])
                else:
                    raise ValueError(f"unsupported reconstructed volatile: {name}")

            attempts = int(source_mon.get("statusTurns", 0))
            if target["status"] == "Sleep":
                remaining = source_mon.get("sleepRemaining")
                if remaining is None:
                    remaining = max(1, 3-attempts) if last == "rest" else rng.randint(max(1, 2-attempts), max(1, 4-attempts))
                add("Sleep", payload=remaining)
            elif target["status"] == "Toxic":
                add("ToxicCounter", payload=min(15, attempts+1))
            protect_counter = int(source_mon.get("protect", 0))
            if protect_counter > 0:
                add("Stall", payload=min(protect_counter, 255))

    def _battle(self, state: dict, seed: int):
        p1, p2, us_is_p1 = self._sides(state)
        b = self.engine.Battle.from_teams(
            self._team(p1), self._team(p2), format=state["format"], stat_points=state["format"].startswith("gen9champions"), seed=int(seed),
            decision_phases=True, tera_allowed="champions" not in state["format"],
        )
        raw = json.loads(b.to_json())
        raw["turn"] = int(state.get("turn", 0))
        raw["weather"] = _WEATHER.get(state.get("weather", ""), "None")
        raw["weather_turns"] = int(state.get("weatherTurns", 0))
        raw["terrain"] = _TERRAIN.get(state.get("terrain", ""), "None")
        raw["terrain_turns"] = int(state.get("terrainTurns", 0))
        pseudo = state.get("pseudo", {})
        for key, raw_key in (("trickroom", "trick_room_turns"), ("gravity", "gravity_turns"),
                             ("magicroom", "magic_room_turns"), ("wonderroom", "wonder_room_turns")):
            raw[raw_key] = int(pseudo.get(key, 0))
        rng = random.Random(seed)
        self._patch_side(raw["p1"], p1, 0, rng)
        self._patch_side(raw["p2"], p2, 1, rng)
        return self.engine.Battle.from_json(json.dumps(raw)), us_is_p1

    @staticmethod
    def _choice(choice: str, state_side: dict, side: int) -> list[tuple]:
        """Translate one Showdown joint choice string into engine wire choices."""
        parts = [p.strip() for p in choice.split(",")]
        out = []
        for actor, part in enumerate(parts[:2]):
            tokens = part.lower().split()
            if not tokens or tokens[0] in {"pass", "default"}:
                out.append(("pass", actor, 0, -1, -1))
                continue
            if tokens[0] == "switch":
                target_str = " ".join(tokens[1:])
                team_index = None
                if len(tokens) > 1 and tokens[1].isdigit():
                    team_index = int(tokens[1]) - 1
                else:
                    target_id = to_id_str(target_str)
                    for i, mon in enumerate(state_side["mons"]):
                        m_set = mon.get("set", {})
                        if (to_id_str(m_set.get("name", "")) == target_id or
                            to_id_str(m_set.get("species", "")) == target_id):
                            team_index = i
                            break
                if team_index is None or team_index < 0 or team_index >= len(state_side["mons"]):
                    raise ValueError(f"unknown switch target: {target_str}")
                out.append(("switch", actor, team_index, -1, -1))
                continue
            if tokens[0] != "move" or len(tokens) < 2:
                raise ValueError(f"unrecognized search choice: {part}")
            move_token = tokens[1]
            moves = state_side["mons"][actor]["set"].get("moves", [])
            move_slot = None
            if move_token.isdigit():
                move_slot = int(move_token) - 1
            else:
                move_id = to_id_str(move_token)
                moves_ids = [to_id_str(m) for m in moves]
                if move_id in moves_ids:
                    move_slot = moves_ids.index(move_id)
                elif move_token in moves:
                    move_slot = moves.index(move_token)
                else:
                    for idx, mid in enumerate(moves_ids):
                        if mid.startswith(move_id) or move_id.startswith(mid):
                            move_slot = idx
                            break
            if move_slot is None or move_slot < 0 or move_slot >= len(moves):
                raise ValueError(f"unknown move in search choice: {move_token}")
            kind = "mega" if "mega" in tokens else "tera" if "tera" in tokens else "move"
            target_side = target_slot = -1
            for token in tokens[2:]:
                try:
                    target = int(token)
                except ValueError:
                    continue
                if target > 0:
                    target_side, target_slot = 1 - side, target - 1
                elif target < 0:
                    target_side, target_slot = side, -target - 1
                break
            out.append((kind, actor, move_slot, target_side, target_slot))
        while len(out) < 2:
            out.append(("pass", len(out), 0, -1, -1))
        return out

    @staticmethod
    def legal_joint(battle, side: int, choices: list[tuple]) -> bool:
        """Validate before stepping: the core runner may ignore invalid choices."""
        switches = [c[2] for c in choices if c[0] == "switch"]
        if len(set(switches)) != len(switches) or sum(c[0] == "mega" for c in choices) > 1:
            return False
        for choice in choices:
            legal = battle.legal_choices(side, choice[1])
            if not legal and choice[0] == "pass":
                continue
            if choice in legal:
                continue
            # An omitted target is valid for engine choices without a target.
            if any(tuple(c[:3]) == tuple(choice[:3]) and tuple(c[3:]) == (-1, -1) for c in legal):
                continue
            # Showdown accepts a single-target move into a vacated opposing
            # slot and retargets it. Rust's execution does too, while its legal
            # enumerator lists only occupied targets.
            if choice[0] in ("move", "mega", "tera") and choice[3] == 1-side and choice[4] in (0, 1):
                raw = json.loads(battle.to_json())["p2" if side == 0 else "p1"]
                index = raw["active"][choice[4]]
                empty = index >= len(raw["team"]) or raw["team"][index]["fainted"]
                if empty and any(tuple(c[:3]) == tuple(choice[:3]) and c[3] == 1-side for c in legal):
                    continue
            return False
        return True

    @staticmethod
    def _material_value(battle: Any, us_is_p1: bool) -> float:
        """Cheap bounded leaf: win chance proxy from remaining HP fractions.

        This deliberately is only Phase-A fallback scoring.  Phase B replaces
        it with vgc-engine observations encoded for the existing value head.
        """
        obs = battle.observe()
        ours, theirs = obs["sides"][0 if us_is_p1 else 1], obs["sides"][1 if us_is_p1 else 0]

        def health(side: dict) -> float:
            mons = [*side["active"], *side["bench"]]
            return sum(float(m["hp"]) / max(int(m["max_hp"]), 1) for m in mons if not m["fainted"])

        a, b = health(ours), health(theirs)
        return 0.5 if a + b == 0 else max(0.0, min(1.0, 0.5 + 0.5 * (a - b) / max(a + b, 1e-9)))

    def run(self, state: dict, pairs: list[tuple[str, str]], seeds: list[int], keep: bool = False) -> list[dict]:
        """Play one Rust turn for every action pair/seed (pair-major results)."""
        results: list[dict] = []
        with self.lock:
            self._kept.clear()
            for us, them in pairs:
                for seed in seeds:
                    try:
                        battle, us_is_p1 = self._battle(state, seed)
                        p1_state, p2_state, _ = self._sides(state)
                        our_side, their_side = (0, 1) if us_is_p1 else (1, 0)
                        p1 = self._choice(us if us_is_p1 else them, p1_state, 0)
                        p2 = self._choice(them if us_is_p1 else us, p2_state, 1)
                        result = battle.step_move(p1, p2)
                        winner = ({"p1_win": "us" if us_is_p1 else "them",
                                   "p2_win": "them" if us_is_p1 else "us", "tie": "tie"}
                                  .get(result))
                        record = {"winner": winner, "err": "", "value": self._material_value(battle, us_is_p1)}
                        if keep and result == "continue" and not battle.needs_replacements():
                            key = len(self._kept)
                            self._kept[key] = (battle, us_is_p1)
                            record["next"] = {"id": key}
                        results.append(record)
                    except Exception as e:  # search must be able to fall back safely
                        results.append({"winner": None, "err": f"vgc-engine: {e}", "value": 0.5})
        return results

    def cont(self, conts: list[dict]) -> list[list[dict]]:
        """Phase-A placeholder: depth-2 requires native action/obs adapters."""
        raise NotImplementedError("vgc-engine backend currently supports --search-depth 1 only")

    def dump(self, state: dict) -> dict:
        """Inspectable engine state reconstructed from a search position."""
        with self.lock:
            battle, _ = self._battle(state, 0)
            return json.loads(battle.to_json())

    def kill(self) -> None:
        self._kept.clear()

    def close(self) -> None:
        self._kept.clear()
