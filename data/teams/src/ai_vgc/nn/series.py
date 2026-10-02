"""Bo3 series context (idea 8, step 3): what each player did in the earlier games of a series.

Built from game logs, so training (scraped Bo3 logs) and play (poke-env's replay log of each
finished game) read it the same way. Pokemon are matched by nickname, which stays the same
across a series (one team per series), unlike species (megas change it).

`encode` reads a `Context` from `battle._series` (set by the dataset builder and the players)
and adds, all zeros without one (game 1, Bo1):

    ser_tok   [T, N_SER_TOK]    per Pokemon: share of earlier games it was brought / led,
                                brought / led last game
    ser_mv    [T, 4, N_SER_MV]  per move: share of earlier games it used the move on turn 1 /
                                at all
    ser_glob  [N_SER_GLOB]      game 2+, game 3, our and their series wins, won last game
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from poke_env import to_id_str

N_SER_TOK = 4
N_SER_MV = 2
N_SER_GLOB = 5

GAME = re.compile(r'Game (\d)</strong> of <a href="/(game-bestof\d-[^"]+)"')
_PLAYER = re.compile(r"^\|player\|(p[12])\|([^|\n]+)", re.M)
_SWITCH = re.compile(r"^\|(?:switch|drag|replace)\|(p[12])[ab]: ([^|\n]+)\|", re.M)
_MOVE = re.compile(r"^\|move\|(p[12])[ab]: ([^|\n]+)\|([^|\n]+)\|?(.*)$", re.M)
_WIN = re.compile(r"^\|win\|(.+)$", re.M)


@dataclass
class Mon:
    brought: bool = False
    led: bool = False
    t1: set[str] = field(default_factory=set)    # move ids used on turn 1
    used: set[str] = field(default_factory=set)  # move ids used at all


@dataclass
class Game:
    mons: dict[str, dict[str, Mon]]  # player id -> nickname -> what it did
    winner: str | None               # player id


def series_key(log: str) -> tuple[str, int] | None:
    """(Bo3 room, game number) from the "Game N of" banner, or None outside a series."""
    g = GAME.search(log)
    return (g.group(2), int(g.group(1))) if g else None


def summarize(log: str) -> Game:
    names = {role: to_id_str(name) for role, name in _PLAYER.findall(log)}
    mons: dict[str, dict[str, Mon]] = {n: {} for n in names.values()}
    t1 = log.find("|turn|1\n")
    t2 = log.find("|turn|2\n")
    for m in _SWITCH.finditer(log):
        mon = mons[names[m.group(1)]].setdefault(m.group(2), Mon())
        mon.brought = True
        mon.led |= t1 < 0 or m.start() < t1
    for m in _MOVE.finditer(log):
        role, nick, move, rest = m.groups()
        if "[from]" in rest or role not in names:
            continue
        mon = mons[names[role]].setdefault(nick, Mon())
        mon.used.add(to_id_str(move))
        if t1 >= 0 and m.start() > t1 and (t2 < 0 or m.start() < t2):
            mon.t1.add(to_id_str(move))
    w = _WIN.search(log)
    return Game(mons, to_id_str(w.group(1)) if w else None)


@dataclass
class Context:
    """The earlier games of a series, seen by player `me` against `them` (player ids)."""
    games: list[Game]
    me: str
    them: str

    @classmethod
    def of(cls, logs: list[str], me: str, them: str) -> Context | None:
        return cls([summarize(x) for x in logs], to_id_str(me), to_id_str(them)) if logs else None

    def side(self, ours: bool) -> list[dict[str, Mon]]:
        who = self.me if ours else self.them
        return [g.mons.get(who, {}) for g in self.games]

    def tok(self, nick: str, ours: bool) -> list[float]:
        per = [s.get(nick) for s in self.side(ours)]
        n, last = len(per), per[-1]
        return [sum(bool(m and m.brought) for m in per) / n, sum(bool(m and m.led) for m in per) / n,
                float(bool(last and last.brought)), float(bool(last and last.led))]

    def mv(self, nick: str, ours: bool, move_id: str) -> list[float]:
        per = [s.get(nick) for s in self.side(ours)]
        n = len(per)
        return [sum(bool(m and move_id in m.t1) for m in per) / n,
                sum(bool(m and move_id in m.used) for m in per) / n]

    def glob(self) -> list[float]:
        ours = sum(g.winner == self.me for g in self.games)
        theirs = sum(g.winner == self.them for g in self.games)
        return [1.0, float(len(self.games) >= 2), ours / 2, theirs / 2,
                float(self.games[-1].winner == self.me)]
