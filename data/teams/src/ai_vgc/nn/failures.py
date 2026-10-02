"""Count the bot's moves that fail, get blocked or hit an immunity, by cause.

    uv run python -m ai_vgc.nn.failures --model data/models/archive/mb-bo3-do-v1.pt --n 2000

Plays the model against itself and reads Showdown's own messages for each used move,
so every cause is named by the simulator rather than guessed:

    cant|<holder>|ability: Armor Tail|<move>|[of] <user>   priority blocked by an ability
    -immune|<target>[|[from] ability: X]                   type/ability immunity, Prankster vs Dark
    -activate|<target>|move: Protect                       protected (a read, not a rule)
    -activate|<target>|move: Psychic Terrain               priority blocked by the terrain
    -fail|<user>...                                        move failed (Fake Out after turn 1, ...)

Both sides are the same bot, so both sides' moves count.
"""

from __future__ import annotations

import argparse
import asyncio
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
from poke_env import ServerConfiguration
from poke_env.battle import AbstractBattle, Move, MoveCategory, Target

from ai_vgc.nn.player import NNPlayer, load_policy

PROTECTS = {"Protect", "Detect", "Spiky Shield", "King's Shield", "Baneful Bunker", "Silk Trap",
            "Burning Bulwark", "Obstruct", "Max Guard", "Wide Guard", "Quick Guard", "Crafty Shield",
            "Mat Block"}


def _mon(ident: str) -> str:
    return ident.split(": ", 1)[-1]


def _move(name: str) -> Move | None:
    try:
        return Move(name.lower().replace(" ", "").replace("-", "").replace("'", ""), 9)
    except Exception:
        return None


def _status(name: str) -> bool:
    m = _move(name)
    return m is not None and m.category == MoveCategory.STATUS


def _spread(name: str) -> bool:
    m = _move(name)
    return m is not None and m.target in (Target.ALL_ADJACENT, Target.ALL_ADJACENT_FOES)


def causes(events: list[list[str]]) -> tuple[int, list[tuple[str, str]]]:
    """(moves used, [(cause, example)]) from one battle's events."""
    used, out = 0, []
    user = move = None
    for e in events:
        if len(e) < 2:
            continue
        kind = e[1]
        if kind == "move" and len(e) >= 4:
            used += 1
            user, move = _mon(e[2]), e[3]
        elif kind in ("turn", "switch", "drag"):
            user = move = None
        elif kind == "cant" and len(e) >= 5 and e[3].startswith("ability:"):
            src = next((_mon(x[5:]) for x in e[5:] if x.startswith("[of] ")), "?")
            out.append((f"blocked by {e[3][9:]}", f"{src}'s {e[4]} into {_mon(e[2])}"))
        elif move is None:
            continue
        elif kind == "-immune":
            frm = next((x[7:] for x in e[3:] if x.startswith("[from] ")), "")
            if _spread(move):
                cause = "spread move, one target immune (not a mistake)"
            elif frm:
                cause = f"immune: {frm.removeprefix('ability: ')}"
            elif _status(move):
                cause = "immune to status move (Prankster vs Dark, Grass vs powder, ...)"
            else:
                cause = "immune by type"
            out.append((cause, f"{user}'s {move} into {_mon(e[2])}"))
        elif kind == "-activate" and len(e) >= 4:
            what = e[3].removeprefix("move: ")
            if what in PROTECTS:
                out.append((f"protected ({what})", f"{user}'s {move} into {_mon(e[2])}"))
            elif what == "Psychic Terrain":
                out.append(("blocked by Psychic Terrain", f"{user}'s {move} into {_mon(e[2])}"))
        elif kind == "-fail" and len(e) >= 3:
            why = next((x for x in e[3:] if x.startswith("[")), "") or (e[3] if len(e) > 3 else "")
            out.append((f"failed: {move}" + (f" ({why})" if why else ""), f"{user}'s {move}"))
    return used, out


class Recorder(NNPlayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.used = 0
        self.found: list[tuple[str, str]] = []
        self.log: dict[str, list[list[str]]] = defaultdict(list)

    async def _handle_battle_message(self, split_messages: list[list[str]]):
        # The first entry is the room id (">battle-..."); battle tags drop the ">".
        self.log[split_messages[0][0][1:]].extend(split_messages[1:])
        await super()._handle_battle_message(split_messages)

    def _battle_finished_callback(self, battle: AbstractBattle) -> None:
        super()._battle_finished_callback(battle)
        used, found = causes(self.log.pop(battle.battle_tag, []))
        self.used += used
        self.found += found


def main() -> None:
    from ai_vgc.showdown import account, ensure_server
    from ai_vgc.teams import RandomPoolTeambuilder

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="data/models/archive/mb-bo3-do-v1.pt")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--format", default="gen9championsvgc2026regmb")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--showdown", type=Path, default=None)
    ap.add_argument("--teams", type=Path, default=None)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--log-level", type=int, default=40)
    ap.add_argument("--no-rules", action="store_true", help="don't mask moves that are sure to fail")
    args = ap.parse_args()
    torch.set_num_threads(1)

    proc = ensure_server(args.port, args.showdown)
    model = load_policy(args.model)
    common = dict(
        battle_format=args.format, max_concurrent_battles=args.concurrency, accept_open_team_sheet=True,
        server_configuration=ServerConfiguration(f"ws://localhost:{args.port}/showdown/websocket",
                                                 "https://play.pokemonshowdown.com/action.php?"),
    )
    try:
        a, b = (Recorder(model, greedy=True, rules=not args.no_rules, account_configuration=account(p), log_level=args.log_level,
                         team=RandomPoolTeambuilder(args.teams) if args.teams else RandomPoolTeambuilder(),
                         **common) for p in ("fa", "fb"))
        t = time.time()
        asyncio.run(a.battle_against(b, n_battles=args.n))
        # Both players see every public message, so one side's log covers both sides' moves.
        counts, examples = Counter(c for c, _ in a.found), defaultdict(Counter)
        for c, ex in a.found:
            examples[c][ex] += 1
        print(f"{args.n} games, {a.used} moves used, {len(a.found)} failed/blocked "
              f"({len(a.found) / max(a.used, 1):.1%})  {time.time() - t:.0f}s\n")
        for c, k in counts.most_common(args.top):
            top = ", ".join(f"{ex} x{m}" for ex, m in examples[c].most_common(2))
            print(f"{k:6d}  {k / max(a.used, 1):6.2%}  {c:<48s} e.g. {top}")
    finally:
        if proc:
            proc.terminate()


if __name__ == "__main__":
    main()
