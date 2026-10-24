"""Full-six, closed-sheet preview inputs and ordered lead plans.

Own sets are known. Opponent inputs contain species only and never their
selected four, moves, items, abilities or stats. Numeric IDs and own stats
come from the same native loader used for battles.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, permutations
import json

import numpy as np
import vgc_engine


PREVIEW_PLANS = np.asarray([
    (*leads, *(i for i in bring if i not in leads))
    for bring in combinations(range(6), 4)
    for leads in permutations(bring, 2)
], dtype=np.int64)
NUM_PREVIEW_PLANS = len(PREVIEW_PLANS)  # 15 brings × 12 ordered lead pairs.
FORMAT = "gen9championsvgc2026regmc"
STATS = ("hp", "atk", "def", "spa", "spd", "spe")


def _known_id(value):
    return 0 if value == 65535 else value


class TeamPreviewPool:
    def __init__(self, teams):
        if not teams or any(len(t) != 6 for t in teams):
            raise ValueError("Learned preview requires full teams of exactly six")
        self.teams = teams
        self.mon_json = [[json.dumps(m, separators=(",", ":")) for m in t] for t in teams]
        self.cat = np.zeros((len(teams), 6, 3), dtype=np.int64)
        self.num = np.zeros((len(teams), 6, 6), dtype=np.float32)
        self.moves = np.zeros((len(teams), 6, 4), dtype=np.int64)
        self.item_present = np.zeros((len(teams), 6), dtype=np.bool_)
        dummy = [{"species": "pikachu", "moves": ["protect"]}] * 2
        dummy_json = json.dumps(dummy)
        for ti, team in enumerate(teams):
            # Keep real members on the bench while parsing: preview must retain
            # their original held items and abilities before any entry effects.
            for start in range(0, 6, 4):
                chunk = team[start:start + 4]
                battle = vgc_engine.Battle.from_teams(
                    json.dumps(dummy + chunk), dummy_json,
                    format=FORMAT, stat_points=True, seed=0,
                )
                mons = json.loads(battle.to_json())["p1"]["team"][2:]
                for mi, mon in enumerate(mons, start=start):
                    self.cat[ti, mi] = [_known_id(mon[k]) for k in ("species_id", "item_id", "ability_id")]
                    self.num[ti, mi] = [mon["stats"][k] / 500.0 for k in STATS]
                    self.moves[ti, mi] = [_known_id(m) for m in mon["moves"]]
                    self.item_present[ti, mi] = mon["item_id"] != 65535

    def __len__(self):
        return len(self.teams)

    def observations(self, ours, theirs):
        ours, theirs = np.asarray(ours), np.asarray(theirs)
        count = len(ours)
        cat = np.zeros((count, 12, 3), dtype=np.int64)
        num = np.zeros((count, 12, 6), dtype=np.float32)
        moves = np.zeros((count, 12, 4), dtype=np.int64)
        cat[:, :6], num[:, :6], moves[:, :6] = self.cat[ours], self.num[ours], self.moves[ours]
        cat[:, 6:, 0] = self.cat[theirs, :, 0]
        return cat, num, moves

    def selected_json(self, team, plan):
        return "[" + ",".join(self.mon_json[team][i] for i in PREVIEW_PLANS[plan]) + "]"


@dataclass
class PreviewDecision:
    team: int
    opponent: int
    action: int
    logp: float
    value: float


@dataclass
class PreviewResult:
    decision: PreviewDecision
    outcome: float
