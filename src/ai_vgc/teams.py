"""Random team pool drawn from Showdown-export files."""

from __future__ import annotations

import random
from pathlib import Path

from poke_env.teambuilder import Teambuilder

TEAMS_DIR = Path(__file__).resolve().parents[2] / "data" / "teams" / "reg_mb"


class RandomPoolTeambuilder(Teambuilder):
    def __init__(self, teams_dir: Path = TEAMS_DIR, seed: int | None = None,
                 names: list[str] | None = None, cycle: bool = False):
        """`names` (file stems, e.g. ["MC196", "MC147"]): only those teams, not the whole folder.
        `cycle`: hand them out in turn, one per match (a Bo3 series asks once), not at random."""
        self._rng = random.Random(seed)
        paths = [teams_dir / f"{t}.txt" for t in names] if names else sorted(teams_dir.glob("*.txt"))
        self._names = [p.stem for p in paths]
        self._teams = [self.join_team(self.parse_showdown_team(p.read_text())) for p in paths]
        if not self._teams:
            raise FileNotFoundError(f"no teams in {teams_dir}")
        self._next = 0 if cycle else None
        self._announce = bool(names)  # the ladder bot's --only / --cycle: say which team each match gets

    def yield_team(self) -> str:
        if self._next is None:
            i = self._rng.randrange(len(self._teams))
        else:
            i, self._next = self._next, (self._next + 1) % len(self._teams)
        self.last_name = self._names[i]
        if self._announce:
            print(f"team: {self._names[i]}", flush=True)
        return self._teams[i]


class GauntletTeambuilder(RandomPoolTeambuilder):
    """Ladder team elimination (successive halving). Every team plays `per_round` series; the worse
    half (by series, then game win rate) is dropped, the rest play `per_round` more, and so on until
    `keep` teams are left, which then share the ladder at random.

    It keeps no state of its own: each match it re-reads the bot's saved games (`games_dir`/*/games.jsonl,
    rows of `fmt` since `since`), so a restarted bot carries on where the last one stopped."""

    def __init__(self, teams_dir: Path, names: list[str], games_dir: Path, fmt: str, since: str,
                 per_round: int = 4, keep: int = 2, seed: int | None = None, user: str | None = None):
        super().__init__(teams_dir, seed, names)
        self._user = user  # count only this account's games: a second bot (challenges) saves to the same file
        self._games_dir, self._fmt, self._since = games_dir, fmt, since
        self._per_round, self._keep = per_round, keep

    def results(self) -> list[list[tuple[bool, int, int]]]:
        """Per team, its finished series since `since` in the order played: (won, games won, games)."""
        import json
        import re

        series: dict[str, tuple[int, list[bool]]] = {}  # dicts keep insertion (= time) order
        for f in sorted(self._games_dir.glob("*/games.jsonl")):
            for line in open(f):
                r = json.loads(line)
                if r["format"] != self._fmt or r["time"] < self._since or r["result"] not in ("win", "loss"):
                    continue
                if self._user and r.get("user", self._user) != self._user:
                    continue
                # The saved team name, not the species: MC358 and MC408 bring the same six.
                if r.get("team") not in self._names:
                    continue
                m = re.search(r'href="/(game-bestof\d-[^"]+)"', r["log"])
                series.setdefault(m.group(1) if m else r["tag"], (self._names.index(r["team"]), []))[1].append(
                    r["result"] == "win")
        out: list[list[tuple[bool, int, int]]] = [[] for _ in self._names]
        for i, games in series.values():
            w = sum(games)
            if len(games) == 1 or w >= 2 or len(games) - w >= 2:  # skip a series the bot left unfinished
                out[i].append((w > len(games) / 2, w, len(games)))
        return out

    def standing(self) -> tuple[list[int], list[int]]:
        """(active team indices, the ones still owed series this round; empty once down to `keep`).
        Round r's cut ranks each team on its first r * per_round series only, so later games of
        the survivors never change an earlier cut."""
        res = self.results()
        active, need = list(range(len(self._names))), self._per_round
        while len(active) > self._keep:
            owed = [i for i in active if len(res[i]) < need]
            if owed:
                return active, owed

            def rate(i: int, need: int = need) -> tuple[float, float]:
                first = res[i][:need]
                return sum(s[0] for s in first) / need, sum(s[1] for s in first) / sum(s[2] for s in first)

            active = sorted(active, key=rate, reverse=True)[:max(self._keep, (len(active) + 1) // 2)]
            need += self._per_round
        return active, []

    def yield_team(self) -> str:
        res = self.results()
        active, owed = self.standing()
        if owed:  # this round: the team furthest behind goes next
            i = min(owed, key=lambda j: (len(res[j]), self._rng.random()))
        else:
            i = self._rng.choice(active)
        table = "  ".join(f"{self._names[j]}{'' if j in active else '(out)'} {sum(s[0] for s in res[j])}/{len(res[j])}"
                          for j in sorted(range(len(res)), key=lambda j: j not in active))
        print(f"team: {self._names[i]}  | series {table}", flush=True)
        self.last_name = self._names[i]
        return self._teams[i]
