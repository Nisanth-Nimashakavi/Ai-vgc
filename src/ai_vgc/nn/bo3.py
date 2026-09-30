"""Bo3 adaptation (idea 8): in games 2 and 3 of a series, pick our team preview as a best
response to the opponent's previous game.

    uv run --extra nn python -m ai_vgc.nn.player --model data/models/archive/mb-bo3-do-v1.pt --adapt \\
        --format gen9championsvgc2026regmbbo3 --opponent nn:data/models/archive/mb-bo3-do-v1.pt --n 100

Human players in the scraped Bo3 logs (`scripts/human_bo3.py`, 58k series) lead the same pair
again 49% of the time after winning the previous game and 24% after losing it. So for game N > 1:

1. Their expected order: last game's leads (in slot order), then the rest of the four they
   showed, filled up to four from their team sheet in order.
2. Our candidates: the policy's `k` most likely previews (`preview.preview_candidates`).
3. `sim_bridge.js` starts a battle for each candidate against that order and returns the
   start-of-battle log (leads out, switch-in abilities such as Intimidate, weather). The log is
   parsed into a copy of the preview battle and the value head scores it, as in search.
4. Score = P(they repeat) x (candidate's value - the policy's top candidate's value) + `prior` x
   its log-probability. The policy's choice stands unless another one looks clearly better
   against their leads, weighted by how likely they are to lead them again.

Game 1, and anything the bridge can't handle, uses the policy's preview.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from poke_env.battle import DoubleBattle

from ai_vgc.nn.preview import preview_candidates
from ai_vgc.nn.search import Bridge, _set, after, values


class Adapter:
    """Chooses games 2+ previews for an NNPlayer (`NNPlayer.adapter`)."""

    def __init__(self, fmt: str, teams_dir: str | Path, showdown: str | Path = "pokemon-showdown",
                 k: int = 8, prior: float = 0.1, bridge: Bridge | None = None,
                 repeat: tuple[float, float] = (0.24, 0.49)):
        self.fmt, self.teams_dir = fmt, str(teams_dir)
        self.repeat = repeat  # P(they lead the same pair) after they lost / won the last game
        self.bridge = bridge or Bridge(showdown)
        self.k, self.prior = k, prior
        self.log: list[dict] = []  # one entry per adapted preview, for checking

    def their_order(self, battle: DoubleBattle, prev: dict) -> list | None:
        """The opponent's Pokemon as they brought them last game: leads first, four in all."""
        team = list(battle.opponent_team.values())
        by_species = {m.species: m for m in team}
        order = [by_species[s] for s in (*prev["their_leads"], *prev["their_brought"]) if s in by_species]
        order = list(dict.fromkeys(order))
        if len(order) < 2 or set(prev["their_leads"]) - set(by_species):
            return None
        order += [m for m in team if m not in order][:4 - len(order)]
        return order[:4]

    def choose(self, player, battle: DoubleBattle, prev: dict) -> str | None:
        """"/team 1234" for this game, or None to use the policy's preview."""
        theirs = self.their_order(battle, prev)
        if theirs is None:
            return None
        cands = preview_candidates(player.model, battle, player.rating)[:self.k]
        team = list(battle.team.values())
        req = {"start": {"format": self.fmt, "role": battle.player_role,
                         "them": [_set(m, self.teams_dir) for m in theirs],
                         "us": [[_set(team[int(i) - 1], self.teams_dir) for i in c] for c, _ in cands]}}
        starts = self.bridge._ask(req)["starts"]
        positions = []
        for (c, _), lines in zip(cands, starts):
            b = after(battle, lines)
            b._teampreview = False
            chosen = {int(i) for i in c}
            for i, m in enumerate(b.team.values(), 1):
                m._selected_in_teampreview = i in chosen
            positions.append(b)
        v = values(player.model, positions, player.rating)
        p = self.repeat[not prev["won"]]
        score = p * (v - v[0]) + self.prior * np.array([lp for _, lp in cands])
        best = int(score.argmax())
        self.log.append({"tag": battle.battle_tag, "their_leads": prev["their_leads"], "p": p,
                         "cands": [c for c, _ in cands], "v": v.round(3).tolist(), "pick": best})
        return "/team " + cands[best][0]
