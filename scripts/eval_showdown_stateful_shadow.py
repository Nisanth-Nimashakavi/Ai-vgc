import argparse
import asyncio
import json
import os
import random
import socket
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from poke_env import AccountConfiguration, ServerConfiguration
from poke_env.battle import Move, Pokemon
from poke_env.data import to_id_str
from poke_env.player import Player
from poke_env.player.battle_order import DoubleBattleOrder, DefaultBattleOrder
from poke_env.teambuilder import Teambuilder

from ai_vgc.nn.belief19 import PublicBeliefBank
from ai_vgc.nn.history18 import apply_known_active_sets, known_history
from ai_vgc.nn.model19 import Transformer19, architecture_for_state
from ai_vgc.nn.player import NNPlayer, load_policy, configure_inference
from ai_vgc.nn.ppo18 import mask_closed_observation, conditioned_b_mask
from ai_vgc.nn.public_sync19 import sync_public_pokemon_state, sync_own_pp
from ai_vgc.nn.search_inputs19 import search_node_inputs, search_leaf_values
from ai_vgc.nn.showdown_history19 import observed_opponent_facts
from ai_vgc.nn.preview18 import TeamPreviewPool, PREVIEW_PLANS
from ai_vgc.nn.dynamic_team import DynamicOpponentBuilder, create_dynamic_preview_pool
from ai_vgc.nn.rules import move_fails
from ai_vgc.showdown import ensure_server, account
from ai_vgc.teams import RandomPoolTeambuilder
import vgc_engine

VOCAB = vgc_engine.policy_vocab()
SPECIES_VOCAB = {to_id_str(s): i for i, s in enumerate(VOCAB["species"])}
ITEMS_VOCAB = {to_id_str(s): i for i, s in enumerate(VOCAB["items"])}
ABILITIES_VOCAB = {to_id_str(s): i for i, s in enumerate(VOCAB["abilities"])}
MOVES_VOCAB = {to_id_str(s): i for i, s in enumerate(VOCAB["moves"])}

def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]

def safe_to_id(val: Any) -> str:
    if not val:
        return ""
    try:
        return to_id_str(str(val))
    except Exception:
        return ""

def species_matches(poke_species: Any, roster_species: Any) -> bool:
    if not poke_species or not roster_species:
        return False
    s1 = safe_to_id(poke_species)
    s2 = safe_to_id(roster_species)
    if not s1 or not s2:
        return False
    if s1 == s2:
        return True
    for suffix in ("megay", "megax", "mega"):
        if s1.endswith(suffix) and s1[:-len(suffix)] == s2:
            return True
        if s2.endswith(suffix) and s2[:-len(suffix)] == s1:
            return True
    return False

def order_to_action18(order: Any, slot: int, active_mon: Any, active_set: dict, raw_side: dict) -> int:
    if order is None:
        return 18
    mon = getattr(order, "order", None)
    if mon is None:
        return 18
    if isinstance(mon, Pokemon):
        active_indices = raw_side.get("active", [])
        bench_indices = [idx for idx in range(4) if idx not in active_indices]
        bench_species = [VOCAB["species"][raw_side["team"][idx]["species_id"]] for idx in bench_indices if idx < len(raw_side.get("team", []))]
        matched_idx = None
        for b_i, b_sp in enumerate(bench_species):
            if species_matches(mon.species, b_sp):
                matched_idx = b_i
                break
        if matched_idx is not None:
            return 16 if matched_idx == 0 else 17
        return 16
    if isinstance(mon, Move):
        if active_mon is None:
            return 18
        move_id = MOVES_VOCAB.get(safe_to_id(getattr(mon, "id", None)), 65535)
        raw_moves = active_set.get("moves", [])
        mid_str = safe_to_id(getattr(mon, "id", None))
        if move_id in raw_moves:
            ms = raw_moves.index(move_id)
        elif active_mon is not None and hasattr(active_mon, "moves") and mid_str in [safe_to_id(m) for m in active_mon.moves.keys()]:
            ms = [safe_to_id(m) for m in active_mon.moves.keys()].index(mid_str)
        else:
            ms = 0
        target = getattr(order, "move_target", 0)
        ti = 0 if target == 1 else 1 if target == 2 else 2 if target < 0 else 3
        base = ms * 4 + ti
        if getattr(order, "mega", False):
            return 19 + base
        return base
    return 18

def parse_opp_turn_actions(replay_data: list, last_turn: int, opp_role: str) -> tuple[dict | None, dict | None]:
    turn_str = str(last_turn)
    start_idx = -1
    for i in range(len(replay_data) - 1, -1, -1):
        msg = replay_data[i]
        if len(msg) >= 3 and msg[1] == 'turn' and msg[2] == turn_str:
            start_idx = i
            break
    if start_idx == -1:
        start_idx = 0

    end_idx = len(replay_data)
    next_turn_str = str(last_turn + 1)
    for i in range(start_idx + 1, len(replay_data)):
        msg = replay_data[i]
        if len(msg) >= 3 and msg[1] == 'turn' and msg[2] == next_turn_str:
            end_idx = i
            break

    turn_msgs = replay_data[start_idx:end_idx]
    opp_a = f"{opp_role}a"
    opp_b = f"{opp_role}b"

    act_0 = None
    act_1 = None

    for msg in turn_msgs:
        if len(msg) < 3:
            continue
        tag = msg[1]
        if tag == '-mega':
            actor = msg[2].split(':')[0].strip()
            if actor == opp_a:
                if act_0 is None: act_0 = {}
                act_0['mega'] = True
            elif actor == opp_b:
                if act_1 is None: act_1 = {}
                act_1['mega'] = True
        elif tag == 'move':
            actor = msg[2].split(':')[0].strip()
            move_id = safe_to_id(msg[3])
            target_str = msg[4].split(':')[0].strip() if len(msg) > 4 else ''
            act_dict = {'type': 'move', 'move': move_id, 'target': target_str, 'mega': False}
            if actor == opp_a and (act_0 is None or 'type' not in act_0):
                if act_0 and act_0.get('mega'): act_dict['mega'] = True
                act_0 = act_dict
            elif actor == opp_b and (act_1 is None or 'type' not in act_1):
                if act_1 and act_1.get('mega'): act_dict['mega'] = True
                act_1 = act_dict
        elif tag == 'switch':
            actor = msg[2].split(':')[0].strip()
            from_parting = any('Parting Shot' in part or 'U-turn' in part or 'Volt Switch' in part or '[from]' in part for part in msg)
            if not from_parting:
                species = safe_to_id(msg[3].split(',')[0].strip())
                act_dict = {'type': 'switch', 'species': species}
                if actor == opp_a and act_0 is None:
                    act_0 = act_dict
                elif actor == opp_b and act_1 is None:
                    act_1 = act_dict

    return act_0, act_1

def parse_upkeep_switches(replay_data: list, last_turn: int, opp_role: str, my_role: str):
    turn_str = str(last_turn)
    next_turn_str = str(last_turn + 1)
    start_idx = 0
    for i in range(len(replay_data) - 1, -1, -1):
        msg = replay_data[i]
        if len(msg) >= 3 and msg[1] == 'turn' and msg[2] == turn_str:
            start_idx = i
            break

    end_idx = len(replay_data)
    for i in range(start_idx + 1, len(replay_data)):
        msg = replay_data[i]
        if len(msg) >= 3 and msg[1] == 'turn' and msg[2] == next_turn_str:
            end_idx = i
            break

    opp_switches = {}
    my_switches = {}
    upkeep_seen = False
    for i in range(start_idx, end_idx):
        msg = replay_data[i]
        if len(msg) >= 2 and msg[1] == 'upkeep':
            upkeep_seen = True
        elif upkeep_seen and len(msg) >= 4 and msg[1] == 'switch':
            if not any('[from]' in part for part in msg):
                actor = msg[2].split(':')[0].strip()
                sp = to_id_str(msg[3].split(',')[0].strip())
                if actor.startswith(opp_role):
                    slot = 0 if actor.endswith('a') else 1
                    opp_switches[slot] = sp
                elif actor.startswith(my_role):
                    slot = 0 if actor.endswith('a') else 1
                    my_switches[slot] = sp

    if not opp_switches and not my_switches:
        # Fallback: any switch with no [from] appearing after the last 'move'
        last_move_idx = -1
        for i in range(start_idx, end_idx):
            if len(replay_data[i]) >= 2 and replay_data[i][1] == 'move':
                last_move_idx = i
        if last_move_idx != -1:
            for i in range(last_move_idx + 1, end_idx):
                msg = replay_data[i]
                if len(msg) >= 4 and msg[1] == 'switch' and not any('[from]' in part for part in msg):
                    actor = msg[2].split(':')[0].strip()
                    sp = to_id_str(msg[3].split(',')[0].strip())
                    if actor.startswith(opp_role):
                        slot = 0 if actor.endswith('a') else 1
                        opp_switches[slot] = sp
                    elif actor.startswith(my_role):
                        slot = 0 if actor.endswith('a') else 1
                        my_switches[slot] = sp
    return my_switches, opp_switches

class ShadowBattleManager:
    def __init__(self, pool: TeamPreviewPool, t_us: list[dict], t_them: list[dict], side: int, sets_dir: str, tracker: Any = None):
        self.pool = pool
        self.t_us = t_us
        self.t_them = t_them
        self.side = side  # 0 for P1, 1 for P2
        self.opp_side = 1 - side
        self.sets_dir = sets_dir
        self.tracker = tracker
        self.picks_us: list[int] = [0, 1, 2, 3]
        self.picks_them: list[int] = [0, 1, 2, 3]
        self.revealed_them: set[int] = set()
        self.fmt = "gen9championsvgc2026regmc"
        self.shadow_battle: Any = None
        self.last_turn: int = 1
        self.last_choices_us: list[tuple] | None = None
        self.pending_repl_us: list[tuple] | None = None
        self.last_act: tuple[int, int] | None = None
        self.full_teams: np.ndarray | None = None
        self.native_picks: np.ndarray | None = None

    def set_picks(self, picks_us: list[int]):
        self.picks_us = picks_us

    def init_turn1(self, battle: Any, opp_true_picks: list[int] | None = None):
        opp_act_pokes = [p for p in battle.opponent_active_pokemon[:2] if p is not None]
        opp_leads = []
        for p in opp_act_pokes:
            for idx, tm in enumerate(self.t_them):
                if species_matches(p.species, tm["species"]) and idx not in opp_leads:
                    opp_leads.append(idx)
                    break

        if opp_true_picks is not None and len(opp_true_picks) >= 4:
            picks_them = list(opp_true_picks[:4])
        else:
            preferred_bench = [0, 5, 1, 4, 2, 3]
            bench_cands = [i for i in preferred_bench if i < len(self.t_them) and i not in opp_leads]
            picks_them = (opp_leads + bench_cands)[:4]
        self.picks_them = picks_them
        self.revealed_them = set(opp_leads)

        roster_us = [dict(self.t_us[i]) for i in self.picks_us]
        roster_them = [dict(self.t_them[i]) for i in self.picks_them]

        p1_roster = roster_us if self.side == 0 else roster_them
        p2_roster = roster_them if self.side == 0 else roster_us

        self.shadow_battle = vgc_engine.Battle.from_teams(
            json.dumps(p1_roster), json.dumps(p2_roster),
            format=self.fmt, stat_points=True, decision_phases=True, seed=random.getrandbits(64)
        )
        self.shadow_battle.enable_belief_observations19()
        self._update_native_picks()
        self.last_turn = 1

    def _update_native_picks(self):
        # Pool IDs are [learner, opponent], regardless of physical battle side.
        # Only native_picks is indexed by physical side (P1/P2).
        self.full_teams = np.array([[0, 1]], dtype=np.int64)
        if self.side == 0:
            self.native_picks = np.array([[(self.picks_us[0], self.picks_us[1], self.picks_us[2], self.picks_us[3]),
                                           (self.picks_them[0], self.picks_them[1], self.picks_them[2], self.picks_them[3])]], dtype=np.int64)
        else:
            self.native_picks = np.array([[(self.picks_them[0], self.picks_them[1], self.picks_them[2], self.picks_them[3]),
                                           (self.picks_us[0], self.picks_us[1], self.picks_us[2], self.picks_us[3])]], dtype=np.int64)

    def reveal_opponent_species(self, species_name: str):
        if self.shadow_battle is None or not species_name:
            return
        sp_idx = None
        for idx, tm in enumerate(self.t_them):
            if species_matches(species_name, tm["species"]):
                sp_idx = idx
                break
        if sp_idx is None:
            return

        self.revealed_them.add(sp_idx)
        if sp_idx in self.picks_them:
            return  # Already in active roster

        # Newly revealed bench mon that was not in our initial guess.
        # Find an unrevealed bench slot in picks_them (indices 2 or 3)
        unrevealed = [slot for slot in (2, 3) if self.picks_them[slot] not in self.revealed_them]
        if not unrevealed:
            unrevealed = [2]
        bench_slot = unrevealed[0]
        self.picks_them[bench_slot] = sp_idx

        # Re-splice into the existing shadow_battle JSON
        roster_us = [dict(self.t_us[i]) for i in self.picks_us]
        roster_them = [dict(self.t_them[i]) for i in self.picks_them]
        p1_r = roster_us if self.side == 0 else roster_them
        p2_r = roster_them if self.side == 0 else roster_us

        temp_b = vgc_engine.Battle.from_teams(
            json.dumps(p1_r), json.dumps(p2_r),
            format=self.fmt, stat_points=True, decision_phases=True, seed=random.getrandbits(64)
        )
        temp_raw = json.loads(temp_b.to_json())
        opp_key = "p2" if self.side == 0 else "p1"
        new_mon_serde = temp_raw[opp_key]["team"][bench_slot]

        raw = json.loads(self.shadow_battle.to_json())
        raw[opp_key]["team"][bench_slot] = new_mon_serde
        self.shadow_battle = vgc_engine.Battle.from_json(json.dumps(raw))
        self.shadow_battle.enable_belief_observations19()
        self._update_native_picks()


    def match_choice(self, side: int, slot: int, act_dict: dict | None) -> tuple:
        legals = self.shadow_battle.legal_choices(side, slot)
        if not legals:
            return ("pass", slot, 0, -1, -1)
        if legals[0][0] == "pass":
            return legals[0]
        if act_dict is None:
            return legals[0]

        a_type = act_dict.get("type")
        if a_type == "switch":
            sp = act_dict.get("species", "")
            raw = json.loads(self.shadow_battle.to_json())
            s_data = raw["p1"] if side == 0 else raw["p2"]
            team = s_data["team"]
            for c in legals:
                if c[0] == "switch":
                    bench_idx = c[2]
                    if bench_idx < len(team):
                        b_sp = VOCAB["species"][team[bench_idx]["species_id"]]
                        if species_matches(sp, b_sp):
                            return c
            for c in legals:
                if c[0] == "switch":
                    return c
            return legals[0]

        if a_type == "move":
            mid = act_dict.get("move", "")
            is_mega = act_dict.get("mega", False)
            target_str = act_dict.get("target", "")
            t_side = 0 if "p1" in target_str else 1 if "p2" in target_str else -1
            t_slot = 0 if target_str.endswith("a") else 1 if target_str.endswith("b") else -1

            raw = json.loads(self.shadow_battle.to_json())
            s_data = raw["p1"] if side == 0 else raw["p2"]
            active_idx = s_data["active"][slot]
            mon_moves = s_data["team"][active_idx]["moves"]
            mon_move_ids = [to_id_str(VOCAB["moves"][m]) for m in mon_moves]
            ms = mon_move_ids.index(mid) if mid in mon_move_ids else None

            # 1. Exact match on move slot, mega, and target
            if t_side != -1 and t_slot != -1:
                for c in legals:
                    c_is_mega = (c[0] == "mega")
                    c_ms = c[2]
                    c_ts = c[3]
                    c_tslot = c[4]
                    if (c[0] in ("move", "mega")) and (ms is None or c_ms == ms) and (c_is_mega == is_mega):
                        if c_ts == t_side and c_tslot == t_slot:
                            return c
            # 2. Match on move slot and mega
            for c in legals:
                c_is_mega = (c[0] == "mega")
                c_ms = c[2]
                if (c[0] in ("move", "mega")) and (ms is None or c_ms == ms) and (c_is_mega == is_mega):
                    return c
            # 3. Match on move slot
            for c in legals:
                if (c[0] in ("move", "mega")) and (ms is None or c[2] == ms):
                    return c
            # 4. Any move
            for c in legals:
                if c[0] in ("move", "mega"):
                    return c
            return legals[0]

        return legals[0]

    def advance_turn_moves(self, battle: Any):
        if self.last_choices_us is None:
            return
        last_t = self.last_turn
        opp_role = "p2" if self.side == 0 else "p1"

        opp_act0, opp_act1 = parse_opp_turn_actions(battle._replay_data, last_t, opp_role)
        for act in (opp_act0, opp_act1):
            if act and act.get("type") == "switch":
                self.reveal_opponent_species(act.get("species", ""))

        c_opp0 = self.match_choice(self.opp_side, 0, opp_act0)
        c_opp1 = self.match_choice(self.opp_side, 1, opp_act1)
        c_them = [c_opp0, c_opp1]
        c_us = self.last_choices_us

        c0 = c_us if self.side == 0 else c_them
        c1 = c_them if self.side == 0 else c_us

        self.shadow_battle.step_move(c0, c1)
        self.last_choices_us = None

    def resolve_upkeep_replacements(self, battle: Any):
        if not self.shadow_battle.needs_replacements():
            return
        opp_role = "p2" if self.side == 0 else "p1"
        my_role = "p1" if self.side == 0 else "p2"
        my_switches, opp_switches = parse_upkeep_switches(battle._replay_data, self.last_turn, opp_role, my_role)
        for sp in opp_switches.values():
            self.reveal_opponent_species(sp)

        r_us = []
        for slot in range(2):
            legals = self.shadow_battle.legal_choices(self.side, slot)
            if not legals or legals[0][0] == "pass":
                r_us.append(("pass", slot, 0, -1, -1))
            elif self.pending_repl_us is not None:
                r_us.append(self.pending_repl_us[slot])
            else:
                sw_sp = my_switches.get(slot)
                choice = self.match_choice(self.side, slot, {"type": "switch", "species": sw_sp} if sw_sp else None)
                r_us.append(choice)

        r_them = []
        for slot in range(2):
            legals = self.shadow_battle.legal_choices(self.opp_side, slot)
            if not legals or legals[0][0] == "pass":
                r_them.append(("pass", slot, 0, -1, -1))
            else:
                sw_sp = opp_switches.get(slot)
                choice = self.match_choice(self.opp_side, slot, {"type": "switch", "species": sw_sp} if sw_sp else None)
                r_them.append(choice)

        r0 = r_us if self.side == 0 else r_them
        r1 = r_them if self.side == 0 else r_us
        self.shadow_battle.step_move(r0, r1)
        self.pending_repl_us = None

    def sync_from_showdown(self, battle: Any):
        if self.shadow_battle is None:
            return

        try:
            for p in battle.opponent_active_pokemon:
                if p is not None:
                    self.reveal_opponent_species(p.species)
            for p in battle.opponent_team.values():
                if getattr(p, "_revealed", False) or (p.current_hp is not None and p.current_hp > 0):
                    self.reveal_opponent_species(p.species)

            raw = json.loads(self.shadow_battle.to_json())
            my_key = "p1" if self.side == 0 else "p2"
            opp_key = "p2" if self.side == 0 else "p1"
            changed = False

            # Shadow rollouts use guessed sets and independent RNG. Their public
            # outcomes must be overwritten by the facts actually announced by PS.
            for key, observed in ((my_key, battle.team.values()),
                                  (opp_key, battle.opponent_team.values())):
                for p in observed:
                    for mon_data in raw[key]["team"]:
                        if species_matches(p.species, VOCAB["species"][mon_data["species_id"]]):
                            changed = sync_public_pokemon_state(mon_data, p) or changed
                            if key == my_key:
                                changed = sync_own_pp(mon_data, p, MOVES_VOCAB) or changed
                            break

            # 1. Sync our team HP and fainted status
            our_team = raw[my_key]["team"]
            for idx, mon_data in enumerate(our_team):
                sp_id = mon_data["species_id"]
                sp_name = VOCAB["species"][sp_id]
                for p in battle.team.values():
                    if species_matches(p.species, sp_name):
                        sd_hp = int(p.current_hp) if p.current_hp is not None else (0 if p.fainted else mon_data["current_hp"])
                        sd_fainted = bool(p.fainted)
                        if mon_data["current_hp"] != sd_hp or mon_data["fainted"] != sd_fainted:
                            mon_data["current_hp"] = sd_hp
                            mon_data["fainted"] = sd_fainted
                            changed = True
                        break

            # 2. Sync our active slots
            new_active = list(raw[my_key]["active"])
            for slot, p in enumerate(battle.active_pokemon):
                if p is not None and not p.fainted and slot < len(new_active):
                    for idx, mon_data in enumerate(our_team):
                        sp_id = mon_data["species_id"]
                        sp_name = VOCAB["species"][sp_id]
                        if species_matches(p.species, sp_name):
                            if new_active[slot] != idx:
                                new_active[slot] = idx
                                changed = True
                            break
            raw[my_key]["active"] = new_active

            # 3. Sync opponent active HP and fainted
            opp_team = raw[opp_key]["team"]
            opp_active = list(raw[opp_key]["active"])
            for slot, p in enumerate(battle.opponent_active_pokemon):
                if p is not None and slot < len(opp_active):
                    opp_idx = opp_active[slot]
                    if opp_idx < len(opp_team):
                        mon_data = opp_team[opp_idx]
                        sp_id = mon_data["species_id"]
                        sp_name = VOCAB["species"][sp_id]
                        if not species_matches(p.species, sp_name):
                            for candidate_idx, c_data in enumerate(opp_team):
                                c_sp = VOCAB["species"][c_data["species_id"]]
                                if species_matches(p.species, c_sp):
                                    opp_active[slot] = candidate_idx
                                    mon_data = c_data
                                    changed = True
                                    break

                        sd_fainted = bool(p.fainted)
                        if p.max_hp and p.current_hp is not None:
                            hp_pct = float(p.current_hp) / float(p.max_hp)
                            max_hp_calc = float(mon_data["stats"]["hp"])
                            sd_hp = int(round(hp_pct * max_hp_calc))
                            if sd_hp == 0 and not sd_fainted:
                                sd_hp = 1
                        else:
                            sd_hp = 0 if sd_fainted else mon_data["current_hp"]

                        if mon_data["fainted"] != sd_fainted or abs(mon_data["current_hp"] - sd_hp) > 5:
                            mon_data["current_hp"] = sd_hp
                            mon_data["fainted"] = sd_fainted
                            changed = True

            raw[opp_key]["active"] = opp_active

            # 4. Sync Mega Evolution forms & abilities
            for slot, p in enumerate(battle.active_pokemon):
                if p is not None and not p.fainted and slot < len(new_active):
                    act_idx = new_active[slot]
                    if act_idx < len(our_team):
                        mon_data = our_team[act_idx]
                        base_sp = safe_to_id(getattr(p, "species", None))
                        stone = safe_to_id(getattr(p, "item", None))
                        is_mega = (
                            (getattr(battle, "used_mega_evolve", False) and getattr(p, "_forme_change_ability", None) is not None)
                            or base_sp.endswith("mega") or base_sp.endswith("megay") or base_sp.endswith("megax")
                        )
                        if is_mega:
                            if "charizarditey" in stone: mega_name = "charizardmegay"
                            elif "charizarditex" in stone: mega_name = "charizardmegax"
                            elif "raichunitey" in stone: mega_name = "raichumegay"
                            elif "raichunitex" in stone: mega_name = "raichumegax"
                            elif base_sp.endswith("mega") or base_sp.endswith("megay") or base_sp.endswith("megax"):
                                mega_name = base_sp
                            else:
                                mega_name = base_sp + "mega"
                            if mega_name in SPECIES_VOCAB:
                                mega_id = SPECIES_VOCAB[mega_name]
                                if mon_data["species_id"] != mega_id:
                                    mon_data["species_id"] = mega_id
                                    changed = True
                                ab_forme = safe_to_id(getattr(p, "_forme_change_ability", None))
                                if ab_forme and ab_forme in ABILITIES_VOCAB:
                                    ab_id = ABILITIES_VOCAB[ab_forme]
                                    if mon_data["ability_id"] != ab_id:
                                        mon_data["ability_id"] = ab_id
                                        changed = True

            for slot, p in enumerate(battle.opponent_active_pokemon):
                if p is not None and not p.fainted and slot < len(opp_active):
                    opp_idx = opp_active[slot]
                    if opp_idx < len(opp_team):
                        mon_data = opp_team[opp_idx]
                        base_sp = safe_to_id(getattr(p, "species", None))
                        stone = safe_to_id(getattr(p, "item", None))
                        is_mega = (
                            (getattr(battle, "opponent_used_mega_evolve", False) and getattr(p, "_forme_change_ability", None) is not None)
                            or base_sp.endswith("mega") or base_sp.endswith("megay") or base_sp.endswith("megax")
                        )
                        if is_mega:
                            if "charizarditey" in stone: mega_name = "charizardmegay"
                            elif "charizarditex" in stone: mega_name = "charizardmegax"
                            elif "raichunitey" in stone: mega_name = "raichumegay"
                            elif "raichunitex" in stone: mega_name = "raichumegax"
                            elif base_sp.endswith("mega") or base_sp.endswith("megay") or base_sp.endswith("megax"):
                                mega_name = base_sp
                            else:
                                mega_name = base_sp + "mega"
                            if mega_name in SPECIES_VOCAB:
                                mega_id = SPECIES_VOCAB[mega_name]
                                if mon_data["species_id"] != mega_id:
                                    mon_data["species_id"] = mega_id
                                    changed = True
                                ab_forme = safe_to_id(getattr(p, "_forme_change_ability", None))
                                if ab_forme and ab_forme in ABILITIES_VOCAB:
                                    ab_id = ABILITIES_VOCAB[ab_forme]
                                    if mon_data["ability_id"] != ab_id:
                                        mon_data["ability_id"] = ab_id
                                        changed = True

            # 5. Sync Weather from Showdown
            sd_weather = "None"
            sd_weather_turns = 0
            if hasattr(battle, "weather") and battle.weather:
                for w_k, w_v in battle.weather.items():
                    w_s = str(w_k).lower()
                    if "rain" in w_s or "primordial" in w_s:
                        sd_weather = "Rain"
                    elif "sun" in w_s or "desolate" in w_s:
                        sd_weather = "Sun"
                    elif "sand" in w_s:
                        sd_weather = "Sand"
                    elif "snow" in w_s or "hail" in w_s:
                        sd_weather = "Snow"
                    sd_weather_turns = int(w_v) if isinstance(w_v, (int, float)) else 5
                    break

            if raw.get("weather") != sd_weather:
                raw["weather"] = sd_weather
                raw["weather_turns"] = sd_weather_turns
                changed = True

            # 6. Sync Terrain & Trick Room from Showdown
            sd_terrain = "None"
            sd_terrain_turns = 0
            sd_trick_room = 0
            if hasattr(battle, "fields") and battle.fields:
                for f_k, f_v in battle.fields.items():
                    f_s = str(f_k).lower()
                    if "grassy" in f_s:
                        sd_terrain = "Grassy"
                        sd_terrain_turns = int(f_v) if isinstance(f_v, (int, float)) else 5
                    elif "psychic" in f_s:
                        sd_terrain = "Psychic"
                        sd_terrain_turns = int(f_v) if isinstance(f_v, (int, float)) else 5
                    elif "electric" in f_s:
                        sd_terrain = "Electric"
                        sd_terrain_turns = int(f_v) if isinstance(f_v, (int, float)) else 5
                    elif "misty" in f_s:
                        sd_terrain = "Misty"
                        sd_terrain_turns = int(f_v) if isinstance(f_v, (int, float)) else 5
                    elif "trick_room" in f_s:
                        sd_trick_room = int(f_v) if isinstance(f_v, (int, float)) else 5

            if raw.get("terrain") != sd_terrain:
                raw["terrain"] = sd_terrain
                raw["terrain_turns"] = sd_terrain_turns
                changed = True
            if raw.get("trick_room_turns", 0) != sd_trick_room:
                raw["trick_room_turns"] = sd_trick_room
                changed = True

            # 7. Audit Logging
            us_sp = [VOCAB["species"][our_team[i]["species_id"]] for i in new_active if i < len(our_team)]
            opp_sp = [VOCAB["species"][opp_team[i]["species_id"]] for i in opp_active if i < len(opp_team)]
            print(f"[Showdown Audit T{battle.turn:02d}] Weather={raw.get('weather')}({raw.get('weather_turns')}) Terrain={raw.get('terrain')} | Us={us_sp} | Opp={opp_sp}", flush=True)

            if changed:
                self.shadow_battle = vgc_engine.Battle.from_json(json.dumps(raw))
                self.shadow_battle.enable_belief_observations19()
        except Exception as e:
            print(f"[Showdown Audit Warning] sync_from_showdown error on T{getattr(battle, 'turn', '?')}: {e}", flush=True)

def solve_regret_matching(
    M: np.ndarray,
    opp_prior: np.ndarray | None = None,
    iterations: int = 1024,
) -> tuple[np.ndarray, np.ndarray, float]:
    K, M_cols = M.shape
    cum_regret_row = np.zeros(K, dtype=np.float64)
    cum_regret_col = np.zeros(M_cols, dtype=np.float64)
    avg_strat_row = np.zeros(K, dtype=np.float64)
    avg_strat_col = np.zeros(M_cols, dtype=np.float64)

    strat_row = np.ones(K, dtype=np.float64) / K
    if opp_prior is not None:
        strat_col = np.asarray(opp_prior, dtype=np.float64).copy()
        s = strat_col.sum()
        strat_col = strat_col / s if s > 1e-8 else np.ones(M_cols, dtype=np.float64) / M_cols
    else:
        strat_col = np.ones(M_cols, dtype=np.float64) / M_cols

    for _ in range(iterations):
        row_payoffs = M @ strat_col
        col_payoffs = strat_row @ M
        val = float(strat_row @ row_payoffs)

        avg_strat_row += strat_row
        avg_strat_col += strat_col

        cum_regret_row += row_payoffs - val
        cum_regret_col += val - col_payoffs

        pos_row = np.maximum(cum_regret_row, 0.0)
        s_row = pos_row.sum()
        strat_row = pos_row / s_row if s_row > 1e-8 else np.ones(K, dtype=np.float64) / K

        pos_col = np.maximum(cum_regret_col, 0.0)
        s_col = pos_col.sum()
        strat_col = pos_col / s_col if s_col > 1e-8 else np.ones(M_cols, dtype=np.float64) / M_cols

    pi_row = avg_strat_row / iterations
    pi_col = avg_strat_col / iterations
    game_value = float(pi_row @ M @ pi_col)
    return pi_row, pi_col, game_value


@torch.no_grad()
def search_turn_19(
    model: Transformer19,
    native: vgc_engine.Battle,
    side: int,
    c: torch.Tensor,
    n: torch.Tensor,
    history: dict[str, torch.Tensor],
    c_opp: torch.Tensor,
    n_opp: torch.Tensor,
    m0: torch.Tensor,
    m1: torch.Tensor,
    valid_pairs: list[tuple[int, int]],
    device: torch.device,
    depth: int = 1,
    k: int = 20,
    m: int = 40,
    k2: int = 2,
    m2: int = 2,
    blend_alpha: float = 0.8,
    search_beta: float = 15.0,
    search_gamma: float = 0.91,
    deterministic: bool = False,
    diagnostics: dict | None = None,
    input_context: dict | None = None,
    chance_branching: bool = True,
    sequential_halving: bool = True,
    win_rate_cap: float = 0.06,
) -> tuple[tuple[int, int], float]:
    """Simultaneous lookahead search (Depth 1 or Depth 2) for Model 19."""
    if diagnostics is not None:
        diagnostics["calls"] += 1
    try:
        if input_context is None:
            raise ValueError("Model 19 search requires its training input context")
        opp_side = 1 - side
        _, (c_opp, n_opp, opp_history) = search_node_inputs(native, side, input_context, device)

        # 1. Opponent candidates for Turn 1
        opp_m0 = torch.tensor(native.legal_action_mask_18(opp_side, 0), dtype=torch.bool, device=device)
        opp_m1 = torch.tensor(native.legal_action_mask_18(opp_side, 1), dtype=torch.bool, device=device)
        if opp_m0.sum() == 0:
            opp_m0[18] = True
        if opp_m1.sum() == 0:
            opp_m1[18] = True

        opp_out_a = model(c_opp, n_opp, **opp_history)
        ola = opp_out_a["policy_a"][0].clone()
        ola[~opp_m0] = -1e9
        ola = ola.log_softmax(-1)
        num_oa = min(7, max(1, int(opp_m0.sum().item())))
        top_oa = ola.topk(num_oa).indices.tolist()

        # Vectorized batch evaluation across all slot-A opponent candidates
        oa_tensor = torch.tensor(top_oa, device=device)
        B_oa = len(top_oa)
        c_opp_exp = c_opp.expand(B_oa, -1, -1)
        n_opp_exp = n_opp.expand(B_oa, -1, -1)
        opp_hist_exp = {k: v.expand(B_oa, -1, -1) if v.dim() == 3 else v.expand(B_oa, -1) for k, v in opp_history.items()}
        opp_out_b = model(c_opp_exp, n_opp_exp, action_a=oa_tensor, **opp_hist_exp)
        olb_all = opp_out_b["policy_b"].clone()

        opp_cands = []
        for i_oa, oa in enumerate(top_oa):
            olb = olb_all[i_oa]
            omb = conditioned_b_mask(opp_m1.unsqueeze(0), torch.tensor([oa], device=device))[0]
            if omb.sum() == 0:
                omb[18] = True
            olb[~omb] = -1e9
            olb = olb.log_softmax(-1)
            num_ob = min(7, max(1, int(omb.sum().item())))
            top_ob = olb.topk(num_ob).indices.tolist()
            for ob in top_ob:
                opp_cands.append(((oa, ob), (ola[oa] + olb[ob]).item()))

        opp_cands.sort(key=lambda x: -x[1])
        opp_cands = opp_cands[:m]
        if not opp_cands:
            opp_cands = [((18, 18), 0.0)]

        # 2. Our candidates for Turn 1 (must be in valid_pairs)
        out_a = model(c, n, **history)
        la = out_a["policy_a"][0].clone()
        la[~m0] = -1e9
        prob_a = la.softmax(-1)
        top_prob_a, top_a_idx = prob_a.max().item(), int(prob_a.argmax().item())

        # Fast-Path: If policy is >=85% confident on an obvious move, exit immediately (<2ms)
        if top_prob_a >= 0.85:
            mb_fast = conditioned_b_mask(m1.unsqueeze(0), torch.tensor([top_a_idx], device=device))[0]
            out_b_fast = model(c, n, action_a=torch.tensor([top_a_idx], device=device), **history)
            lb_fast = out_b_fast["policy_b"][0].clone()
            lb_fast[~mb_fast] = -1e9
            prob_b = lb_fast.softmax(-1)
            top_prob_b, top_b_idx = prob_b.max().item(), int(prob_b.argmax().item())
            if (top_prob_a * top_prob_b) >= 0.80 and (top_a_idx, top_b_idx) in valid_pairs:
                return (top_a_idx, top_b_idx), 0.0

        la = la.log_softmax(-1)
        num_a = min(5, max(1, int(m0.sum().item())))
        top_a = la.topk(num_a).indices.tolist()

        # Vectorized batch evaluation across all slot-A our candidates
        a0_tensor = torch.tensor(top_a, device=device)
        B_a = len(top_a)
        c_exp = c.expand(B_a, -1, -1)
        n_exp = n.expand(B_a, -1, -1)
        hist_exp = {k: v.expand(B_a, -1, -1) if v.dim() == 3 else v.expand(B_a, -1) for k, v in history.items()}
        out_b = model(c_exp, n_exp, action_a=a0_tensor, **hist_exp)
        lb_all = out_b["policy_b"].clone()

        valid_set = set(valid_pairs)
        our_cands = []
        for i_a, a0 in enumerate(top_a):
            lb = lb_all[i_a]
            mb = conditioned_b_mask(m1.unsqueeze(0), torch.tensor([a0], device=device))[0]
            if mb.sum() == 0:
                mb[18] = True
            lb[~mb] = -1e9
            lb = lb.log_softmax(-1)
            num_b = min(5, max(1, int(mb.sum().item())))
            top_b = lb.topk(num_b).indices.tolist()
            for b0 in top_b:
                if (a0, b0) in valid_set:
                    our_cands.append(((a0, b0), (la[a0] + lb[b0]).item()))

        our_cands.sort(key=lambda x: -x[1])
        our_cands = our_cands[:k]
        if not our_cands:
            our_cands = [(valid_pairs[0], 0.0)]

        # 3. Payoff Matrix M via Rust chance branching or multi-world rollouts
        num_our = len(our_cands)
        num_opp = len(opp_cands)
        M = np.zeros((num_our, num_opp), dtype=np.float32)

        # Pre-resolve valid action choice tuples
        our_choices_tuples = []
        valid_our_indices = []
        for i, ((a0, a1), _) in enumerate(our_cands):
            try:
                c_us = native.actions_18_to_choices(side, a0, a1)
                our_choices_tuples.append(c_us)
                valid_our_indices.append(i)
            except Exception:
                if diagnostics is not None:
                    diagnostics["branch_errors"] = diagnostics.get("branch_errors", 0) + 1

        opp_choices_tuples = []
        valid_opp_indices = []
        for j, ((o0, o1), _) in enumerate(opp_cands):
            try:
                c_them = native.actions_18_to_choices(opp_side, o0, o1)
                opp_choices_tuples.append(c_them)
                valid_opp_indices.append(j)
            except Exception:
                if diagnostics is not None:
                    diagnostics["branch_errors"] = diagnostics.get("branch_errors", 0) + 1

        ran_chance = False
        if chance_branching and our_choices_tuples and opp_choices_tuples and depth == 1:
            try:
                res = vgc_engine.multi_world_rollout_chance(
                    [native],
                    our_choices_tuples,
                    [opp_choices_tuples],
                    our_side=side,
                    rating=1500.0,
                    ots=False,
                )
                term_vals = np.array(res["terminal_values"], dtype=np.float32)
                # Map terminal values: win (1.0) -> +1.0, loss (0.0) -> -1.0, draw (0.5) -> 0.0
                term_vals = 2.0 * term_vals - 1.0

                nt_idx = np.array(res["non_terminal_indices"], dtype=np.intp)
                nt_battles = res["non_terminal_battles"]

                if len(nt_idx) > 0:
                    vals = search_leaf_values(model, nt_battles, side, input_context, device)
                    term_vals[nt_idx] = vals

                prob = np.array(res["outcome_prob"], dtype=np.float32)
                offs = np.asarray(res["cell_offsets"], dtype=np.intp)
                cnts = np.asarray(res["cell_counts"], dtype=np.intp)

                csum = np.concatenate(([0.0], np.cumsum(prob * term_vals, dtype=np.float32)))
                cell_vals = (csum[offs + cnts] - csum[offs]).astype(np.float32)

                k_our = len(our_choices_tuples)
                k_opp = len(opp_choices_tuples)
                M_sub = cell_vals.reshape(k_our, k_opp)
                for idx_i, vi in enumerate(valid_our_indices):
                    for idx_j, vj in enumerate(valid_opp_indices):
                        M[vi, vj] = M_sub[idx_i, idx_j]
                ran_chance = True
            except Exception as e:
                if diagnostics is not None:
                    diagnostics["branch_errors"] = diagnostics.get("branch_errors", 0) + 1
                ran_chance = False

        t1_leaves = []
        t1_leaf_coords = []
        continuation_nodes = []

        if not ran_chance:
            for i in valid_our_indices:
                c_us0, c_us1 = our_choices_tuples[valid_our_indices.index(i)]
                for j in valid_opp_indices:
                    c_them0, c_them1 = opp_choices_tuples[valid_opp_indices.index(j)]
                    try:
                        leaf = native.clone()
                        c_p1, c_p2 = ([c_us0, c_us1], [c_them0, c_them1]) if side == 0 else ([c_them0, c_them1], [c_us0, c_us1])
                        term = leaf.step_move(c_p1, c_p2)
                        win_term = "p1_win" if side == 0 else "p2_win"
                        lose_term = "p2_win" if side == 0 else "p1_win"
                        if term == win_term:
                            M[i, j] = 1.0
                        elif term == lose_term:
                            M[i, j] = -1.0
                        elif term == "draw":
                            M[i, j] = 0.0
                        elif depth == 1 or leaf.needs_replacements():
                            t1_leaves.append(leaf)
                            t1_leaf_coords.append((i, j))
                        else:
                            continuation_nodes.append((i, j, leaf))
                    except Exception:
                        if diagnostics is not None:
                            diagnostics["branch_errors"] = diagnostics.get("branch_errors", 0) + 1
                        M[i, j] = 0.0

        # 4. Turn 2 Lookahead Expansions (if depth >= 2)
        t2_subgames = {}
        t2_leaf_data = []
        t2_leaf_coords = []

        if depth >= 2 and continuation_nodes:
            for (ci, cj, leaf_b) in continuation_nodes:
                (c2, n2, h2), (c2_opp, n2_opp, h2_opp) = search_node_inputs(
                    leaf_b, side, input_context, device)
                m2_our0 = torch.tensor(leaf_b.legal_action_mask_18(side, 0), dtype=torch.bool, device=device)
                m2_our1 = torch.tensor(leaf_b.legal_action_mask_18(side, 1), dtype=torch.bool, device=device)
                m2_opp0 = torch.tensor(leaf_b.legal_action_mask_18(opp_side, 0), dtype=torch.bool, device=device)
                m2_opp1 = torch.tensor(leaf_b.legal_action_mask_18(opp_side, 1), dtype=torch.bool, device=device)
                if m2_our0.sum() == 0: m2_our0[18] = True
                if m2_our1.sum() == 0: m2_our1[18] = True
                if m2_opp0.sum() == 0: m2_opp0[18] = True
                if m2_opp1.sum() == 0: m2_opp1[18] = True

                # Turn 2 Our candidates
                o2_a = model(c2, n2, **h2)
                la2 = o2_a["policy_a"][0].clone()
                la2[~m2_our0] = -1e9
                la2 = la2.log_softmax(-1)
                top2_a = la2.topk(min(k2, int(m2_our0.sum().item()))).indices.tolist()

                cands2_our = []
                for a0 in top2_a:
                    o2_b = model(c2, n2, action_a=torch.tensor([a0], device=device), **h2)
                    lb2 = o2_b["policy_b"][0].clone()
                    mb2 = conditioned_b_mask(m2_our1.unsqueeze(0), torch.tensor([a0], device=device))[0]
                    if mb2.sum() == 0: mb2[18] = True
                    lb2[~mb2] = -1e9
                    lb2 = lb2.log_softmax(-1)
                    top2_b = lb2.topk(min(k2, int(mb2.sum().item()))).indices.tolist()
                    for b0 in top2_b:
                        cands2_our.append(((a0, b0), (la2[a0] + lb2[b0]).item()))
                cands2_our.sort(key=lambda x: -x[1])
                cands2_our = cands2_our[:k2]
                if not cands2_our:
                    cands2_our = [((18, 18), 0.0)]

                # Turn 2 Opponent candidates
                opp2_a = model(c2_opp, n2_opp, **h2_opp)
                ola2 = opp2_a["policy_a"][0].clone()
                ola2[~m2_opp0] = -1e9
                ola2 = ola2.log_softmax(-1)
                top_oa2 = ola2.topk(min(m2, int(m2_opp0.sum().item()))).indices.tolist()

                cands2_opp = []
                for oa in top_oa2:
                    opp2_b = model(c2_opp, n2_opp, action_a=torch.tensor([oa], device=device), **h2_opp)
                    olb2 = opp2_b["policy_b"][0].clone()
                    omb2 = conditioned_b_mask(m2_opp1.unsqueeze(0), torch.tensor([oa], device=device))[0]
                    if omb2.sum() == 0: omb2[18] = True
                    olb2[~omb2] = -1e9
                    olb2 = olb2.log_softmax(-1)
                    top_ob2 = olb2.topk(min(m2, int(omb2.sum().item()))).indices.tolist()
                    for ob in top_ob2:
                        cands2_opp.append(((oa, ob), (ola2[oa] + olb2[ob]).item()))
                cands2_opp.sort(key=lambda x: -x[1])
                cands2_opp = cands2_opp[:m2]
                if not cands2_opp:
                    cands2_opp = [((18, 18), 0.0)]

                t2_matrix = np.zeros((len(cands2_our), len(cands2_opp)), dtype=np.float32)
                opp2_lps = np.array([lp for (_, lp) in cands2_opp], dtype=np.float64)
                opp2_priors = np.exp(opp2_lps - np.max(opp2_lps))
                opp2_priors /= (opp2_priors.sum() + 1e-8)

                for u, ((a2_0, a2_1), _) in enumerate(cands2_our):
                    try:
                        c2_us0, c2_us1 = leaf_b.actions_18_to_choices(side, a2_0, a2_1)
                    except Exception:
                        if diagnostics is not None:
                            diagnostics["branch_errors"] = diagnostics.get("branch_errors", 0) + 1
                        continue
                    for v, ((o2_0, o2_1), _) in enumerate(cands2_opp):
                        try:
                            c2_them0, c2_them1 = leaf_b.actions_18_to_choices(opp_side, o2_0, o2_1)
                            leaf2 = leaf_b.clone()
                            c2_p1, c2_p2 = ([c2_us0, c2_us1], [c2_them0, c2_them1]) if side == 0 else ([c2_them0, c2_them1], [c2_us0, c2_us1])
                            res2 = leaf2.step_move(c2_p1, c2_p2)
                            win_term2 = "p1_win" if side == 0 else "p2_win"
                            lose_term2 = "p2_win" if side == 0 else "p1_win"
                            if res2 == win_term2:
                                t2_matrix[u, v] = 1.0
                            elif res2 == lose_term2:
                                t2_matrix[u, v] = -1.0
                            elif res2 == "draw":
                                t2_matrix[u, v] = 0.0
                            else:
                                t2_leaf_data.append(leaf2)
                                t2_leaf_coords.append((ci, cj, u, v))
                        except Exception:
                            if diagnostics is not None:
                                diagnostics["branch_errors"] = diagnostics.get("branch_errors", 0) + 1
                            t2_matrix[u, v] = 0.0

                t2_subgames[(ci, cj)] = {
                    "matrix": t2_matrix,
                    "opp_prior": opp2_priors,
                }

        # 5. Batched neural evaluation of all T1 and T2 leaves
        if t1_leaves:
            v1 = search_leaf_values(model, t1_leaves, side, input_context, device)
            for (ci, cj), val in zip(t1_leaf_coords, v1):
                M[ci, cj] = val

        if t2_leaf_data:
            v2 = search_leaf_values(model, t2_leaf_data, side, input_context, device)
            for (ci, cj, u, v), val in zip(t2_leaf_coords, v2):
                t2_subgames[(ci, cj)]["matrix"][u, v] = val

        # 6. Backward Induction: solve each Turn 2 subgame
        for (ci, cj), sub in t2_subgames.items():
            _, _, gv2 = solve_regret_matching(sub["matrix"], opp_prior=sub["opp_prior"])
            M[ci, cj] = gv2

        # 7. Solve root Turn 1 matrix M via Regret Matching
        opp_lps = np.array([lp for (_, lp) in opp_cands], dtype=np.float64)
        opp_priors = np.exp(opp_lps - np.max(opp_lps))
        opp_priors /= (opp_priors.sum() + 1e-8)

        our_lps = np.array([lp for (_, lp) in our_cands], dtype=np.float64)
        our_priors = np.exp(our_lps - np.max(our_lps))
        our_priors /= (our_priors.sum() + 1e-8)

        pi_search, pi_opp, game_val = solve_regret_matching(M, opp_prior=opp_priors)

        # Spec formulation: P(move) ∝ π(move)^γ * exp(β * (q(move) - V))
        q_vals = M @ pi_opp  # Expected value under opponent equilibrium
        v_root = float(game_val)
        advantages = q_vals - v_root

        prior_term = np.power(np.maximum(our_priors, 1e-12), search_gamma)
        exp_term = np.exp(np.clip(search_beta * advantages, -50.0, 50.0))
        p_unnorm = prior_term * exp_term
        p_sum = p_unnorm.sum()
        if p_sum > 0 and not np.isnan(p_sum):
            p_strat = p_unnorm / p_sum
        else:
            p_strat = (1.0 - blend_alpha) * our_priors + blend_alpha * pi_search
            p_strat /= (p_strat.sum() + 1e-8)

        if deterministic:
            best_idx = int(np.argmax(p_strat))
        else:
            best_idx = int(np.random.choice(len(our_cands), p=p_strat))

        return our_cands[best_idx][0], game_val
    except Exception as e:
        import traceback; traceback.print_exc()
        if diagnostics is not None:
            diagnostics["failures"] += 1
            diagnostics["last_error"] = str(e)
        return valid_pairs[0], 0.0


class StatefulShadow19Player(Player):
    def __init__(self, model: Transformer19, pool: TeamPreviewPool | None = None,
                 t_us: list[dict] | None = None, t_them: list[dict] | None = None,
                 sets_dir: str = "data/teams/reg_mc",
                 dynamic_opp: bool = False, dynamic_builder: DynamicOpponentBuilder | None = None,
                 priors_path: str | None = None, device: str = "cpu",
                 oracle_preview: bool = False, oracle_picks: dict[str, list[int]] | None = None,
                 use_search: bool = False, search_depth: int = 2, search_k: int = 20, search_m: int = 40,
                 search_k2: int = 2, search_m2: int = 2, search_blend: float = 0.8,
                 search_beta: float = 15.0, search_gamma: float = 0.91, deterministic: bool = False,
                 rules: bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.pool = pool
        self.t_us = t_us
        self.t_them = t_them
        self.sets_dir = sets_dir
        self.device = torch.device(device)
        self.dynamic_opp = dynamic_opp or (t_them is None)
        if self.dynamic_opp:
            self.dynamic_builder = dynamic_builder or DynamicOpponentBuilder(sets_dir)
        else:
            self.dynamic_builder = None
        self.belief_bank = PublicBeliefBank(priors_path, particles=16) if priors_path else None
        self.shadow_managers: dict[str, ShadowBattleManager] = {}
        self.oracle_preview = oracle_preview
        self.oracle_picks = oracle_picks if oracle_picks is not None else {}
        self.use_search = use_search
        self.search_depth = search_depth
        self.search_k = search_k
        self.search_m = search_m
        self.search_k2 = search_k2
        self.search_m2 = search_m2
        self.search_blend = search_blend
        self.search_beta = search_beta
        self.search_gamma = search_gamma
        self.deterministic = deterministic
        self.rules = rules
        self.search_diagnostics = {"calls": 0, "failures": 0, "branch_errors": 0}

    def _battle_finished_callback(self, battle: Any):
        w = self.n_won_battles
        l = self.n_lost_battles
        tot = w + l
        outcome = "WON" if getattr(battle, "won", False) else "LOST"
        print(f"[Game {tot:02d}] Model 19 ({self.username}) {outcome}! Score: {w}-{l} ({w / max(1, tot) * 100:.1f}%) [turns: {battle.turn}]", flush=True)
        if battle.battle_tag in self.shadow_managers:
            del self.shadow_managers[battle.battle_tag]


    def teampreview(self, battle: Any) -> str:
        side = 0 if battle.player_role == "p1" else 1
        tracker = self.belief_bank.tracker() if self.belief_bank else None

        t_us = getattr(self.team, "last_team_json", None) or self.t_us

        if self.dynamic_builder is not None:
            opp_species = list(battle.teampreview_opponent_team) if battle.teampreview_opponent_team else []
            t_them = self.dynamic_builder.build_team(opp_species)
            pool = create_dynamic_preview_pool(t_us, t_them)
            print(f"[Preview] Dynamically reconstructed opponent team: {[m['species'] for m in t_them]}", flush=True)
        else:
            t_them = self.t_them
            pool = create_dynamic_preview_pool(t_us, t_them) if getattr(self.team, "last_team_json", None) else self.pool

        mgr = ShadowBattleManager(pool, t_us, t_them, side, self.sets_dir, tracker)
        self.shadow_managers[battle.battle_tag] = mgr

        cat, num, moves = pool.observations([0], [1])
        with torch.no_grad():
            out = self.model(
                preview_cat=torch.from_numpy(cat).to(self.device),
                preview_num=torch.from_numpy(num).to(self.device),
                preview_moves=torch.from_numpy(moves).to(self.device),
                preview_only=True,
            )
            logits = out["preview_plan"][0].float()
            plan_id = int(logits.argmax().item() if self.deterministic else
                          torch.distributions.Categorical(logits=logits).sample().item())
        picks = list(PREVIEW_PLANS[plan_id])
        mgr.set_picks(picks)

        ours = list(battle.team.values())
        for i, mon in enumerate(ours):
            mon._selected_in_teampreview = i in picks
        order_str = "/team " + "".join(str(p + 1) for p in picks)
        return order_str

    def choose_move(self, battle: Any) -> Any:
        steps = self._choose_move_steps(battle)
        result = None
        while True:
            try:
                c, n, context = steps.send(result)
            except StopIteration as finished:
                return finished.value
            with torch.no_grad():
                result = self.model(c, n, **context)

    async def choose_move_batched(self, battle: Any, batcher) -> Any:
        steps = self._choose_move_steps(battle)
        result = None
        try:
            while True:
                try:
                    c, n, context = steps.send(result)
                except StopIteration as finished:
                    return finished.value
                result = await batcher.forward(c, n, **context)
        finally:
            steps.close()

    def _choose_move_steps(self, battle: Any) -> Any:
        orders = DoubleBattleOrder.join_orders(*battle.valid_orders)
        if not orders:
            return DefaultBattleOrder()
        try:
            side = 0 if battle.player_role == "p1" else 1
            mgr = self.shadow_managers.get(battle.battle_tag)
            if mgr is None:
                tracker = self.belief_bank.tracker() if self.belief_bank else None
                t_us = getattr(self.team, "last_team_json", None) or self.t_us
                if self.dynamic_builder is not None:
                    opp_species = list(battle.opponent_team.values()) if battle.opponent_team else []
                    t_them = self.dynamic_builder.build_team(opp_species)
                    pool = create_dynamic_preview_pool(t_us, t_them)
                else:
                    t_them = self.t_them
                    pool = create_dynamic_preview_pool(t_us, t_them) if getattr(self.team, "last_team_json", None) else self.pool
                mgr = ShadowBattleManager(pool, t_us, t_them, side, self.sets_dir, tracker)
                mgr.set_picks([0, 1, 2, 3])
                self.shadow_managers[battle.battle_tag] = mgr

            is_force_switch = any(battle.force_switch) if hasattr(battle, "force_switch") else False
            if mgr.shadow_battle is None:
                true_picks = None
                if self.oracle_preview:
                    true_picks = self.oracle_picks.get(battle.battle_tag)
                mgr.init_turn1(battle, opp_true_picks=true_picks)
            else:
                if mgr.last_choices_us is not None:
                    mgr.advance_turn_moves(battle)
                if not is_force_switch and mgr.shadow_battle.needs_replacements():
                    mgr.resolve_upkeep_replacements(battle)
                mgr.last_turn = battle.turn

            # Opening abilities and the owner's PP are authoritative on turn 1 too.
            mgr.sync_from_showdown(battle)

            native = mgr.shadow_battle
            enc_learner = native.encode_17(side)
            enc_opp = native.encode_17(1 - side)

            cat = np.zeros((2, 1, 17, 4), dtype=np.int64)
            num = np.zeros((2, 1, 17, 16), dtype=np.float32)
            cat[0, 0] = np.frombuffer(enc_learner["tok_cat"], np.uint32).reshape(17, 4)
            num[0, 0] = np.frombuffer(enc_learner["tok_num"], np.float32).reshape(17, 16)
            cat[1, 0] = np.frombuffer(enc_opp["tok_cat"], np.uint32).reshape(17, 4)
            num[1, 0] = np.frombuffer(enc_opp["tok_num"], np.float32).reshape(17, 16)

            mask_closed_observation(cat, num)
            learner_sides = np.array([side], dtype=np.int64)
            facts_raw = np.frombuffer(native.public_history_18(), dtype="<u4").reshape(2, 2, 6, 20).copy()
            facts = np.expand_dims(facts_raw, axis=1)
            facts = observed_opponent_facts(facts, json.loads(native.to_json()), battle,
                                            side, VOCAB, species_matches)
            apply_known_active_sets(cat, facts, learner_sides)

            enc_hist = known_history(mgr.pool, mgr.full_teams, mgr.native_picks, learner_sides, facts)

            history = {
                "history_cat": torch.from_numpy(enc_hist[0][0]).to(self.device),
                "history_num": torch.from_numpy(enc_hist[1][0]).to(self.device),
                "history_moves": torch.from_numpy(enc_hist[2][0]).to(self.device),
                "history_flags": torch.from_numpy(enc_hist[3][0]).to(self.device),
            }

            if mgr.tracker is not None:
                view = native.public_view19(side)
                bf = np.frombuffer(mgr.tracker.features(view), dtype="<f4").reshape(17, 16).copy()
                history["belief_features"] = torch.from_numpy(bf).unsqueeze(0).to(self.device)

            c = torch.from_numpy(cat[0]).long().to(self.device)
            n = torch.from_numpy(num[0]).float().to(self.device)

            m0 = native.legal_action_mask_18(side, 0)
            m1 = native.legal_action_mask_18(side, 1)

            raw = json.loads(native.to_json())
            raw_us = raw["p1"] if side == 0 else raw["p2"]
            act_set_0 = raw_us["team"][raw_us["active"][0]]
            act_set_1 = raw_us["team"][raw_us["active"][1]]

            act0 = battle.active_pokemon[0] if len(battle.active_pokemon) > 0 else None
            act1 = battle.active_pokemon[1] if len(battle.active_pokemon) > 1 else None

            pairs = [(order_to_action18(o.first_order, 0, act0, act_set_0, raw_us),
                      order_to_action18(o.second_order, 1, act1, act_set_1, raw_us))
                     for o in orders]

            valid_indices = [i for i, (a, b) in enumerate(pairs) if m0[a] and m1[b]]
            if valid_indices:
                orders = [orders[i] for i in valid_indices]
                pairs = [pairs[i] for i in valid_indices]

            def is_charge_trap(m_order, mon):
                if not m_order or not isinstance(getattr(m_order, 'order', None), Move):
                    return False
                mid = getattr(m_order.order, 'id', '')
                if mid == 'electroshot':
                    if getattr(mon, 'item', '') == 'powerherb':
                        return False
                    weather = getattr(battle, 'weather', {})
                    if not any('rain' in str(w).lower() for w in weather):
                        return True
                return False

            filtered_orders = []
            filtered_pairs = []
            for o, (a, b) in zip(orders, pairs):
                drop = False
                if is_charge_trap(getattr(o, 'first_order', None), act0):
                    drop = True
                if is_charge_trap(getattr(o, 'second_order', None), act1):
                    drop = True
                o1 = getattr(o, "first_order", None)
                if o1 and isinstance(getattr(o1, "order", None), Move) and act0 is not None:
                    target_code = getattr(o1, "move_target", 0)
                    if move_fails(battle, 0, o1.order, target_code):
                        drop = True
                    if getattr(o1.order, "id", "") in ("protect", "detect", "spikyshield", "banefulbunker", "burningbulwark", "silktrap") and getattr(act0, "protect_counter", 0) >= 1:
                        drop = True
                o2 = getattr(o, "second_order", None)
                if o2 and isinstance(getattr(o2, "order", None), Move) and act1 is not None:
                    target_code = getattr(o2, "move_target", 0)
                    if move_fails(battle, 1, o2.order, target_code):
                        drop = True
                    if getattr(o2.order, "id", "") in ("protect", "detect", "spikyshield", "banefulbunker", "burningbulwark", "silktrap") and getattr(act1, "protect_counter", 0) >= 1:
                        drop = True
                if not drop:
                    filtered_orders.append(o)
                    filtered_pairs.append((a, b))
            if filtered_orders and self.rules:
                orders = filtered_orders
                pairs = filtered_pairs

            raw = json.loads(native.to_json())
            raw_us = raw["p1"] if side == 0 else raw["p2"]
            for slot, m_arr in enumerate([m0, m1]):
                act_idx = raw_us["active"][slot]
                if act_idx < len(raw_us["team"]):
                    m_data = raw_us["team"][act_idx]
                    m_moves = [to_id_str(VOCAB["moves"][mv]) for mv in m_data["moves"]]
                if self.rules and "electroshot" in m_moves:
                        item_id = m_data.get("item_id", 65535)
                        item = to_id_str(VOCAB["items"][item_id]) if item_id < len(VOCAB["items"]) else ""
                        weather = raw.get("weather", "None")
                        if item != "powerherb" and weather != "Rain":
                            ms_idx = m_moves.index("electroshot")
                            for k in range(ms_idx*4, ms_idx*4 + 4): m_arr[k] = False

            mask_t = torch.from_numpy(np.stack([[m0, m1]])).to(self.device)
            request_mask = torch.zeros_like(mask_t)
            for a, b in pairs:
                request_mask[0, 0, a] = True
                request_mask[0, 1, b] = True
            mask_t = mask_t & request_mask

            if self.use_search and not is_force_switch:
                best_pair, g_val = search_turn_19(
                    model=self.model,
                    native=native,
                    side=side,
                    c=c,
                    n=n,
                    history=history,
                    c_opp=torch.from_numpy(cat[1]).long().to(self.device),
                    n_opp=torch.from_numpy(num[1]).float().to(self.device),
                    m0=mask_t[0, 0],
                    m1=mask_t[0, 1],
                    valid_pairs=pairs,
                    device=self.device,
                    depth=self.search_depth,
                    k=self.search_k,
                    m=self.search_m,
                    k2=self.search_k2,
                    m2=self.search_m2,
                    blend_alpha=self.search_blend,
                    search_beta=self.search_beta,
                    search_gamma=self.search_gamma,
                    deterministic=self.deterministic,
                    diagnostics=self.search_diagnostics,
                    input_context=dict(pool=mgr.pool, full_teams=mgr.full_teams,
                                       native_picks=mgr.native_picks, belief_bank=self.belief_bank,
                                       root_facts=facts, root_side=side),
                )
                if best_pair in pairs:
                    best_idx = pairs.index(best_pair)
                    ret = orders[best_idx]
                else:
                    raise RuntimeError(
                        f"search returned illegal action pair {best_pair} at turn "
                        f"{getattr(battle, 'turn', '?')}"
                    )
            else:
                out_a = yield c, n, history
                la = out_a["policy_a"][0].clone()
                la[~mask_t[0, 0]] = -1e9
                act_a = int(la.argmax().item() if self.deterministic else
                            torch.distributions.Categorical(logits=la).sample().item())

                mb = conditioned_b_mask(mask_t[:, 1], torch.tensor([act_a], device=self.device))[0]
                pair_mask = torch.zeros_like(mb)
                pair_mask[[b for a, b in pairs if a == act_a]] = True
                mb = mb & pair_mask
                out_b = yield c, n, dict(history, action_a=torch.tensor([act_a], device=self.device))
                lb = out_b["policy_b"][0].clone()
                lb[~mb] = -1e9
                act_b = int(lb.argmax().item() if self.deterministic else
                            torch.distributions.Categorical(logits=lb).sample().item())

                candidates_a = [i for i, (a, b) in enumerate(pairs) if a == act_a]
                if (act_a, act_b) in pairs:
                    best_idx = pairs.index((act_a, act_b))
                    ret = orders[best_idx]
                    best_pair = (act_a, act_b)
                elif candidates_a:
                    best_idx = max(candidates_a, key=lambda i: lb[pairs[i][1]].item())
                    ret = orders[best_idx]
                    best_pair = pairs[best_idx]
                else:
                    best_idx = max(range(len(orders)), key=lambda i: (la[pairs[i][0]] + lb[pairs[i][1]]).item())
                    ret = orders[best_idx]
                    best_pair = pairs[best_idx]
            try:
                ch0, ch1 = native.actions_18_to_choices(side, best_pair[0], best_pair[1])
            except Exception as error:
                raise RuntimeError(
                    f"failed to convert selected actions {best_pair} at turn "
                    f"{getattr(battle, 'turn', '?')}"
                ) from error
            if is_force_switch:
                if native.needs_replacements():
                    opp_needs_repl = any(c[0] == 'switch' for slot in (0, 1) for c in native.legal_choices(mgr.opp_side, slot))
                    if not opp_needs_repl:
                        r_us = [ch0, ch1]
                        r_them = [('pass', 0, 0, -1, -1), ('pass', 1, 0, -1, -1)]
                        r0 = r_us if side == 0 else r_them
                        r1 = r_them if side == 0 else r_us
                        native.step_move(r0, r1)
                        mgr.pending_repl_us = None
                    else:
                        mgr.pending_repl_us = [ch0, ch1]
                mgr.last_choices_us = None
                return ret
            else:
                mgr.last_choices_us = [ch0, ch1]
                mgr.last_act = best_pair
                return ret
        except Exception as error:
            import traceback
            traceback.print_exc()
            raise RuntimeError(
                f"Model 19 move selection failed at turn {getattr(battle, 'turn', '?')} "
                f"for {getattr(battle, 'battle_tag', '<unknown battle>')}"
            ) from error

async def run_benchmark(
    games: int = 50,
    oracle: bool = False,
    search: bool = False,
    depth: int = 2,
    search_k: int = 20,
    search_m: int = 40,
    search_k2: int = 2,
    search_m2: int = 2,
    search_blend: float = 0.8,
    model_path: str = "data/models/model19_best_1054m_ema.pt",
    v7f_rules: bool = False,
    dynamic_opp: bool = False,
    opp_team: str = "MC378",
    my_team: str = "MC378",
    pool_dir: str = "data/teams/reg_mc",
    opp_model: str | None = None,
    belief_priors: str | None = None,
    device: str = "auto",
):
    port = find_free_port()
    print("=" * 70)
    print(f"STATEFUL SHADOW BENCHMARK: MODEL 19 ({my_team}) vs {opp_team} (oracle={oracle}, search={search}, depth={depth}, v7f_rules={v7f_rules}, dynamic_opp={dynamic_opp})")
    print("=" * 70)
    print(f"Port:        {port}")
    print(f"Games:       {games} (half P1, half P2)")
    print(f"Oracle mode: {oracle}")
    print(f"Dynamic Opp: {dynamic_opp}")
    print(f"My Team:     {my_team}")
    print(f"Opp Team:    {opp_team}")
    print(f"Search:      {search} (depth={depth}, k={search_k}, m={search_m}, k2={search_k2}, m2={search_m2}, blend={search_blend})")
    print(f"Model 19:    {model_path}")
    print(f"v7f rules:   {v7f_rules}")
    print("-" * 70)

    proc = ensure_server(port, Path("pokemon-showdown-mc"))
    t0 = time.time()
    config = ServerConfiguration(f"ws://localhost:{port}/showdown/websocket", "https://play.pokemonshowdown.com/action.php?")
    fmt = "gen9championsvgc2026regmc"

    is_pool_us = my_team.lower() in ("pool", "all", "field", "top10", "top") or Path(my_team).is_dir()
    if is_pool_us:
        target_dir = Path("data/teams/reg_mc_top_meta") if my_team.lower() in ("top10", "top") else (Path(my_team) if Path(my_team).is_dir() else Path(pool_dir))
        tb_us = RandomPoolTeambuilder(target_dir, cycle=True)
        team_json = tb_us._team_jsons[0]
        pool = TeamPreviewPool([team_json, team_json])
        my_team_file = target_dir / f"{tb_us._names[0]}.txt"
    else:
        my_team_file = Path("data/teams/reg_mc") / f"{my_team}.txt" if not my_team.endswith(".txt") else Path(my_team)
        if not my_team_file.exists():
            if (Path(pool_dir) / f"{my_team}.txt").exists():
                my_team_file = Path(pool_dir) / f"{my_team}.txt"
            else:
                raise FileNotFoundError(f"My team file not found: {my_team_file}")
        paste = my_team_file.read_text("utf-8")
        parsed = Teambuilder.parse_showdown_team(paste)
        team_json = [{
            "species": m.species or m.nickname,
            "level": 50,
            "ability": m.ability or "",
            "item": m.item or "",
            "nature": m.nature or "Serious",
            "moves": m.moves,
            "evs": m.evs or {},
            "ivs": m.ivs or {},
        } for m in parsed]
        pool = TeamPreviewPool([team_json, team_json])
        tb_us = RandomPoolTeambuilder(Path("data/teams/reg_mc") if Path("data/teams/reg_mc").exists() else Path(pool_dir), names=[my_team_file.stem])

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device in (None, "auto") else torch.device(device)
    print(f"Device:      {dev}")
    print(f"Loading Model 19 from: {model_path}")
    ckpt_19 = torch.load(model_path, map_location=dev, weights_only=False)
    arch = architecture_for_state(ckpt_19["model_state_dict"])
    m19 = arch().to(dev)
    m19.load_state_dict({k.removeprefix("module."): v for k, v in ckpt_19["model_state_dict"].items()})
    m19.eval()

    is_pool_opp = opp_team.lower() in ("pool", "all", "field")
    if opp_model:
        opp_model_path = opp_model
    else:
        if is_pool_opp:
            opp_model_path = "data/models/mc-cts-rnad-v6.pt"
        else:
            opp_model_path = f"data/models/mc-cts-rnad-v8f-{opp_team}.pt"
            if not Path(opp_model_path).exists():
                opp_model_path = f"data/models/mc-cts-rnad-v7f-{opp_team}.pt"
            if not Path(opp_model_path).exists():
                opp_model_path = "data/models/mc-cts-rnad-v6.pt"
    print(f"Loading opponent Policy from: {opp_model_path} (team: {opp_team})...")
    m_v7f = load_policy(opp_model_path, str(dev)).eval()
    if is_pool_opp:
        tb_opp = RandomPoolTeambuilder(Path(pool_dir), cycle=True)
        dynamic_opp = True
    else:
        tb_opp = RandomPoolTeambuilder(Path(pool_dir), names=[opp_team])

    shared_oracle_picks = {}
    n_p1 = (games + 1) // 2
    n_p2 = games - n_p1

    if belief_priors and Path(belief_priors).exists():
        priors_file = belief_priors
    elif Path("data/public-priors.json").exists():
        priors_file = "data/public-priors.json"
    elif Path("runs/model19-beliefs-20261016/data/munchstats-priors.json").exists():
        priors_file = "runs/model19-beliefs-20261016/data/munchstats-priors.json"
    else:
        priors_file = None

    # Phase 1: Model 19 as P1
    print(f"\n--- Phase 1: Model 19 as P1 ({n_p1} games, oracle={oracle}, search={search}, depth={depth}, dynamic_opp={dynamic_opp}) ---", flush=True)
    p19_p1 = StatefulShadow19Player(
        m19, pool, team_json, team_json, sets_dir="data/teams/reg_mc",
        dynamic_opp=dynamic_opp,
        priors_path=priors_file,
        device=str(dev), team=tb_us, account_configuration=account("m19p1s"),
        battle_format=fmt, accept_open_team_sheet=False,
        max_concurrent_battles=1, server_configuration=config,
        oracle_preview=oracle, oracle_picks=shared_oracle_picks,
        use_search=search, search_depth=depth, search_k=search_k, search_m=search_m,
        search_k2=search_k2, search_m2=search_m2, search_blend=search_blend,
    )
    pv7f_p2 = NNPlayer(
        m_v7f, rating=1700, greedy=True, rules=v7f_rules, team=tb_opp,
        account_configuration=account("v7fp2s"), battle_format=fmt,
        accept_open_team_sheet=False, max_concurrent_battles=1,
        server_configuration=config,
    )
    configure_inference(pv7f_p2, "data/teams/reg_mc", fmt)
    orig_choose_p2 = pv7f_p2._choose_preview
    async def wrapped_choose_p2(b):
        order = await orig_choose_p2(b)
        if order.startswith("/team "):
            shared_oracle_picks[b.battle_tag] = [int(x) - 1 for x in order.removeprefix("/team ")]
        return order
    pv7f_p2._choose_preview = wrapped_choose_p2

    await p19_p1.battle_against(pv7f_p2, n_battles=n_p1)
    w1, l1 = p19_p1.n_won_battles, p19_p1.n_lost_battles
    print(f"Phase 1: Model 19 won {w1} / {w1 + l1} ({w1 / max(1, w1 + l1) * 100:.1f}%)", flush=True)

    # Phase 2: Model 19 as P2
    print(f"\n--- Phase 2: Model 19 as P2 ({n_p2} games, oracle={oracle}, search={search}, depth={depth}, dynamic_opp={dynamic_opp}) ---", flush=True)
    pv7f_p1 = NNPlayer(
        m_v7f, rating=1700, greedy=True, rules=v7f_rules, team=tb_opp,
        account_configuration=account("v7fp1s"), battle_format=fmt,
        accept_open_team_sheet=False, max_concurrent_battles=1,
        server_configuration=config,
    )
    configure_inference(pv7f_p1, "data/teams/reg_mc", fmt)
    orig_choose_p1 = pv7f_p1._choose_preview
    async def wrapped_choose_p1(b):
        order = await orig_choose_p1(b)
        if order.startswith("/team "):
            shared_oracle_picks[b.battle_tag] = [int(x) - 1 for x in order.removeprefix("/team ")]
        return order
    pv7f_p1._choose_preview = wrapped_choose_p1

    p19_p2 = StatefulShadow19Player(
        m19, pool, team_json, team_json, sets_dir="data/teams/reg_mc",
        dynamic_opp=dynamic_opp,
        priors_path=priors_file,
        device=str(dev), team=tb_us, account_configuration=account("m19p2s"),
        battle_format=fmt, accept_open_team_sheet=False,
        max_concurrent_battles=1, server_configuration=config,
        oracle_preview=oracle, oracle_picks=shared_oracle_picks,
        use_search=search, search_depth=depth, search_k=search_k, search_m=search_m,
        search_k2=search_k2, search_m2=search_m2, search_blend=search_blend,
    )
    await pv7f_p1.battle_against(p19_p2, n_battles=n_p2)
    w2, l2 = p19_p2.n_won_battles, p19_p2.n_lost_battles
    print(f"Phase 2: Model 19 won {w2} / {w2 + l2} ({w2 / max(1, w2 + l2) * 100:.1f}%)", flush=True)

    total_w = w1 + w2
    total_g = w1 + l1 + w2 + l2
    elapsed = time.time() - t0
    print("=" * 70, flush=True)
    print(f"FINAL RESULT: Model 19 won {total_w} / {total_g} ({total_w / max(1, total_g) * 100:.1f}%) in {elapsed:.1f}s", flush=True)
    print(f"P1 win rate: {w1} / {max(1, w1 + l1)} ({w1 / max(1, w1 + l1) * 100:.1f}%)", flush=True)
    print(f"P2 win rate: {w2} / {max(1, w2 + l2)} ({w2 / max(1, w2 + l2) * 100:.1f}%)", flush=True)
    print("=" * 70, flush=True)

    result_data = {
        "games": total_g,
        "oracle_preview": oracle,
        "search": search,
        "search_depth": depth,
        "search_k": search_k,
        "search_m": search_m,
        "search_k2": search_k2,
        "search_m2": search_m2,
        "search_blend": search_blend,
        "v7f_rules": v7f_rules,
        "model_19": model_path,
        "opponent": opp_model_path,
        "team": str(my_team_file),
        "opp_team": opp_team,
        "priors": "runs/model19-beliefs-20261016/data/munchstats-priors.json",
        "phase1_model19_p1": {
            "wins": w1,
            "losses": l1,
            "ties": 0,
            "games": w1 + l1,
            "win_rate": w1 / max(1, w1 + l1) * 100.0
        },
        "phase2_model19_p2": {
            "wins": w2,
            "losses": l2,
            "ties": 0,
            "games": w2 + l2,
            "win_rate": w2 / max(1, w2 + l2) * 100.0
        },
        "summary": {
            "total_games": total_g,
            "wins_model19": total_w,
            "wins_v7f": total_g - total_w,
            "ties": 0,
            "win_rate": total_w / max(1, total_g) * 100.0,
            "p1_win_rate": w1 / max(1, w1 + l1) * 100.0,
            "p2_win_rate": w2 / max(1, w2 + l2) * 100.0,
            "elapsed_s": elapsed
        }
    }
    search_tag = f"_search_d{depth}_k{search_k}m{search_m}_k{search_k2}m{search_m2}" if search else ""
    rules_tag = "_rules" if v7f_rules else "_norules"
    oracle_tag = "_oracle" if oracle else ""
    opp_tag = Path(opp_model_path).stem if opp_model_path else opp_team
    model_tag = Path(model_path).stem
    out_file = f"showdown_eval_{model_tag}_vs_{opp_tag}{search_tag}{oracle_tag}{rules_tag}.json"
    with open(out_file, "w") as f:
        json.dump(result_data, f, indent=2)
    print(f"Saved results to {out_file}", flush=True)
    proc.terminate()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=50)
    parser.add_argument("--oracle", action="store_true", help="Initialize shadow battle with opponent true brought Pokémon (native benchmark parity mode)")
    parser.add_argument("--search", action="store_true", help="Enable simultaneous-move search with Regret Matching")
    parser.add_argument("--depth", type=int, default=2, help="Search depth: 1 (1-turn) or 2 (2-turn lookahead)")
    parser.add_argument("--search-k", type=int, default=20, help="Turn 1 candidate actions for our side")
    parser.add_argument("--search-m", type=int, default=40, help="Turn 1 candidate actions for opponent")
    parser.add_argument("--search-k2", type=int, default=2, help="Turn 2 candidate actions for our side")
    parser.add_argument("--search-m2", type=int, default=2, help="Turn 2 candidate actions for opponent")
    parser.add_argument("--search-blend", type=float, default=0.8, help="Blend weight for search equilibrium vs net prior")
    parser.add_argument("--model-path", type=str, default="data/models/model19_best_1054m_ema.pt", help="Path to Model 19 checkpoint")
    parser.add_argument("--v7f-rules", action="store_true", help="Enable rule heuristics on v7f opponent (default False for pure policy parity)")
    parser.add_argument("--dynamic-opp", action="store_true", help="Dynamically reconstruct opponent team from Showdown Team Preview species")
    parser.add_argument("--my-team", type=str, default="MC378", help="Candidate team name in data/teams/reg_mc (default: MC378)")
    parser.add_argument("--opp-team", type=str, default="MC378", help="Opponent team name in data/teams/reg_mc (e.g. MC378, MC301, pool)")
    parser.add_argument("--pool-dir", type=str, default="data/teams/reg_mc", help="Opponent team pool directory (default: data/teams/reg_mc)")
    parser.add_argument("--opp-model", type=str, default=None, help="Explicit path to opponent model (e.g. data/models/mc-cts-rnad-v6.pt)")
    parser.add_argument("--belief-priors", type=str, default="data/public-priors.json", help="Path to public priors JSON")
    parser.add_argument("--device", type=str, default="auto", help="Compute device: auto, cuda, or cpu")
    args = parser.parse_args()
    asyncio.run(run_benchmark(
        games=args.games,
        oracle=args.oracle,
        search=args.search,
        depth=args.depth,
        search_k=args.search_k,
        search_m=args.search_m,
        search_k2=args.search_k2,
        search_m2=args.search_m2,
        search_blend=args.search_blend,
        model_path=args.model_path,
        v7f_rules=args.v7f_rules,
        dynamic_opp=args.dynamic_opp,
        opp_team=args.opp_team,
        my_team=args.my_team,
        pool_dir=args.pool_dir,
        opp_model=args.opp_model,
        belief_priors=args.belief_priors,
        device=args.device,
    ))
