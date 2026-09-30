"""Replay a Showdown log through poke-env to recover each decision a player made.

Replaying through poke-env (rather than our own log parser) means the battle
objects at training time are exactly what a live poke-env player sees, so the
encoder needs no separate code path for logs.

Adapted from vgc-bench's `logs2trajs.LogReader` (MIT License, Copyright (c)
2025 Cameron Angliss, https://github.com/cameronangliss/vgc-bench). Differences:
the callback encodes each state immediately instead of deep-copying battles,
and the Pokemon a player brought are read from the log up front, so the
"selected at team preview" flags are right at every decision.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Callable

import numpy as np
from poke_env import to_id_str
from poke_env.battle import SPECIAL_MOVES, DoubleBattle, Move, Target
from poke_env.concurrency import POKE_LOOP
from poke_env.environment import DoublesEnv
from poke_env.player import (
    DoubleBattleOrder,
    PassBattleOrder,
    Player,
    SingleBattleOrder,
)
from poke_env.ps_client import AccountConfiguration

SINGLE_TARGETS = {Target.ANY, Target.NORMAL, Target.ADJACENT_FOE}

# on_decision(battle, action, preview, opp): called before the turn's messages are applied.
# `opp` [2, 2] is what each opposing active did that turn (see `opp_labels`), -1 = unknown.
Callback = Callable[[DoubleBattle, np.ndarray, bool, np.ndarray], None]
NO_OPP = np.full((2, 2), -1, np.int8)


def opp_labels(battle: DoubleBattle, msg: str) -> np.ndarray:
    """Per opposing active slot: [move index in its known moves (0-3) or 4 = switched, target:
    0 = our slot a, 1 = our slot b, 2 = anything else]. Targets are -1 for switches."""
    out = NO_OPP.copy()
    me = battle.player_role
    them = "p2" if me == "p1" else "p1"
    for pos, slot in enumerate("ab"):
        mon = battle.opponent_active_pokemon[pos]
        for line in msg.split("\n"):
            if line.startswith(f"|move|{them}{slot}: ") and "[from]" not in line and mon is not None:
                parts = line.split("|")
                ids = [m.id for m in list(mon.moves.values())[:4]]
                mid = to_id_str(parts[3])
                if mid in ids:
                    tgt = parts[4] if len(parts) > 4 else ""
                    out[pos] = [ids.index(mid), 0 if tgt.startswith(f"{me}a:") else 1 if tgt.startswith(f"{me}b:") else 2]
                break
            if line.startswith(f"|switch|{them}{slot}: "):
                out[pos] = [4, -1]
                break
    return out


class LogReader(Player):
    MESSAGES_TO_IGNORE = Player.MESSAGES_TO_IGNORE | {"uhtml"}

    def __init__(self, username: str, battle_format: str, on_decision: Callback):
        super().__init__(
            account_configuration=AccountConfiguration(username, None),
            battle_format=battle_format,
            log_level=51,
            accept_open_team_sheet=True,
            start_listening=False,
        )
        self.on_decision = on_decision
        self.next_msg = ""
        self.brought: set[str] = set()

        async def _noop(*args, **kwargs):
            pass

        self.ps_client.send_message = _noop

    async def _handle_battle_request(self, battle, maybe_default_order: bool = False):
        pass

    def _mark_brought(self, battle: DoubleBattle) -> None:
        for mon in battle.team.values():
            mon._selected_in_teampreview = mon.name in self.brought

    def choose_move(self, battle):
        assert isinstance(battle, DoubleBattle)
        order = DoubleBattleOrder(
            self.get_order(battle, self.next_msg, 0), self.get_order(battle, self.next_msg, 1)
        )
        action = DoublesEnv.order_to_action(order, battle, fake=True)
        battle._available_moves = [[], []]
        # Skip turns where a slot "passed" only because it fainted before moving:
        # what it would have done is unknown.
        if 0 not in action or not (np.all(action == 0) or f"|faint|{battle.player_role}" in self.next_msg):
            self._mark_brought(battle)
            self.on_decision(battle, action, False, opp_labels(battle, self.next_msg))
        return order

    def teampreview(self, battle):
        """Leads are known from turn 1; the back two only if both were revealed later."""
        assert isinstance(battle, DoubleBattle)
        team = list(battle.team.values())
        leads = [self.get_teampreview_order(battle, self.next_msg, p) for p in (0, 1)]
        battle._teampreview = True
        for mon in team:
            mon._selected_in_teampreview = False
        self.on_decision(battle, np.array(leads), True, NO_OPP)
        back = [i for i, m in enumerate(team, 1) if m.name in self.brought and i not in leads]
        if len(back) == 2:
            for i in leads:
                team[i - 1]._selected_in_teampreview = True
            self.on_decision(battle, np.array(back), True, NO_OPP)
        battle._teampreview = False
        return ""

    @staticmethod
    def get_order(battle: DoubleBattle, msg: str, pos: int) -> SingleBattleOrder:
        slot = "a" if pos == 0 else "b"
        order = PassBattleOrder()
        for line in msg.split("\n"):
            if line.startswith(f"|move|{battle.player_role}{slot}: ") and "[from]" not in line:
                [_, _, identifier, move_id, target_identifier, *_] = line.split("|")
                if not target_identifier:
                    # Charge moves that fire at once (Electro Shot in rain) log the
                    # target only on the animation line.
                    anim = re.search(rf"^\|-anim\|{re.escape(identifier)}\|{re.escape(move_id)}\|([^|\n]+)",
                                     msg, re.M)
                    target_identifier = anim.group(1) if anim else ""
                active = battle.active_pokemon[pos]
                assert active is not None, battle.player_role
                if to_id_str(move_id) in SPECIAL_MOVES:
                    move = Move(to_id_str(move_id), gen=battle.gen)
                    battle._available_moves[pos] += [move]
                else:
                    move = active.moves[to_id_str(move_id)]
                target_lines = [ln for ln in msg.split("\n") if f"|switch|{target_identifier}" in ln]
                target_details = target_lines[0].split("|")[3] if target_lines else ""
                target = (
                    battle.get_pokemon(target_identifier, details=target_details)
                    if ": " in target_identifier
                    else None
                )
                move_target = battle.to_showdown_target(move, target)
                # If the target switched out this turn, poke-env can't place it;
                # the slot in the log line still says which position was chosen.
                pos_match = re.match(r"(p[12])([ab]): ", target_identifier)
                if move.target in SINGLE_TARGETS and pos_match:
                    ours = pos_match.group(1) == battle.player_role
                    idx = 1 if pos_match.group(2) == "a" else 2
                    move_target = -idx if ours else idx
                order = SingleBattleOrder(
                    move,
                    mega=f"|-mega|{identifier}|" in msg,
                    terastallize=f"|-terastallize|{identifier}|" in msg,
                    move_target=move_target,
                )
            elif line.startswith((f"|switch|{battle.player_role}{slot}: ", f"|drag|{battle.player_role}{slot}: ")):
                [_, _, identifier, details, *_] = line.split("|")
                order = SingleBattleOrder(battle.get_pokemon(identifier, details=details))
            elif line.startswith(f"|swap|{battle.player_role}{slot}: "):
                slot = "b" if slot == "a" else "a"
            elif line.startswith(("|switch|", "|drag|")):
                [_, _, identifier, details, *_] = line.split("|")
                battle.get_pokemon(identifier, details=details)
        return order

    @staticmethod
    def get_teampreview_order(battle: DoubleBattle, msg: str, pos: int) -> int:
        slot = "a" if pos == 0 else "b"
        start = msg.index(f"|switch|{battle.player_role}{slot}: ")
        end = msg.index("\n", start)
        [_, _, identifier, details, *_] = msg[start:end].split("|")
        mon = battle.get_pokemon(identifier, details=details)
        return list(battle.team.values()).index(mon) + 1

    async def follow_log(self, tag: str, log: str, role: str) -> DoubleBattle:
        # Nicknames this player sent out at any point = the 4 they brought.
        self.brought = set(re.findall(rf"^\|(?:switch|drag)\|{role}[ab]: ([^|]+)\|", log, re.M))
        tag = f"battle-{tag}"
        self.ps_client._battle_locks[tag] = asyncio.Lock()
        messages = [f">{tag}\n" + m for m in log.split("\n|\n")]
        battle = await self._create_battle(f">{tag}".split("-"))
        assert isinstance(battle, DoubleBattle)
        battle.logger = None
        await self._handle_battle_message([m.split("|") for m in messages[0].split("\n")])
        for i in range(1, len(messages)):
            self.next_msg = messages[i]
            if i == 1:
                self.teampreview(battle)
            elif "|switch|" in self.next_msg or "|move|" in self.next_msg:
                self.choose_move(battle)
            await self._handle_battle_message([m.split("|") for m in messages[i].split("\n")])
        return battle


def player_info(log: str, role: str) -> tuple[str, int]:
    """(username, rating) from the |player| line; rating 0 when unrated."""
    m = re.search(rf"^\|player\|{role}\|([^|]*)\|[^|]*\|(\d*)", log, re.M)
    if not m:
        raise ValueError(f"no player line for {role}")
    return m.group(1), int(m.group(2) or 0)


def winner(log: str) -> str | None:
    m = re.search(r"^\|win\|(.+)$", log, re.M)
    return m.group(1).strip() if m else None


def replay(tag: str, log: str, role: str, on_decision: Callback) -> DoubleBattle:
    """Replay `log` from `role`'s view ("p1"/"p2"), calling on_decision at each choice."""
    username, _ = player_info(log, role)
    reader = LogReader(username, tag.split("-")[0], on_decision)
    return asyncio.run_coroutine_threadsafe(reader.follow_log(tag, log, role), POKE_LOOP).result()
