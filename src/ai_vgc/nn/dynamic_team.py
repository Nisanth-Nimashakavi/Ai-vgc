"""Dynamic opponent team and set reconstruction for open Showdown ladder.

Reconstructs the opponent's full 6-mon roster from public Showdown observations
(Team Preview species, revealed items, abilities, moves) by combining:
1. Exact/near whole-team lookup against tournament pastes and known meta teams.
2. MunchStats / usage frequency distributions for moves, items, abilities, and stat points.
3. Safe Pokédex fallbacks ensuring vgc_engine validity under Champions format rules
   (stat points <= 66, max 32 per stat).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable
import numpy as np

from poke_env.battle import Pokemon
from poke_env.data import GenData, to_id_str
import vgc_engine

from ai_vgc.nn.preview18 import TeamPreviewPool
from ai_vgc.nn.search import _pokedex, _core, _base, spread_prior, team_library

DEFAULT_USAGE_PATH = Path("data/teams/set_usage_gen9championsvgc2026regmc.json")
DEFAULT_TEAMS_DIR = "data/teams/reg_mc"
DEFAULT_FORMAT = "gen9championsvgc2026regmc"
STATS = ("hp", "atk", "def", "spa", "spd", "spe")

# Staple meta fillers when opponent roster has fewer than 6 Pokemon in preview
META_FILLERS = ("incineroar", "rillaboom", "fluttermane", "ogerponhearthflame", "kingambit", "gholdengo")


class DynamicOpponentBuilder:
    """Constructs legal, statistically-grounded 6-mon opponent teams from public Showdown observations."""

    def __init__(
        self,
        teams_dir: str = DEFAULT_TEAMS_DIR,
        fmt: str = DEFAULT_FORMAT,
        usage_path: Path | str | None = None,
    ):
        self.teams_dir = teams_dir
        self.fmt = fmt
        self.usage_path = Path(usage_path) if usage_path else DEFAULT_USAGE_PATH
        self.set_usage: dict[str, list] = {}
        if self.usage_path.exists():
            try:
                self.set_usage = json.loads(self.usage_path.read_text("utf-8"))
            except Exception as e:
                print(f"[DynamicOpponentBuilder] Warning: could not load set_usage: {e}")

        try:
            self.lib = team_library(self.teams_dir, self.fmt)
        except Exception as e:
            print(f"[DynamicOpponentBuilder] Warning: could not load team_library: {e}")
            self.lib = {}

        self.gd = GenData.from_gen(9)

    def normalize_species(self, mon: Any) -> str:
        """Extract and normalize species ID, resolving Mega forms and cosmetic variants to base forms."""
        if isinstance(mon, Pokemon):
            raw = mon.species or mon.name
        elif isinstance(mon, dict):
            raw = mon.get("species") or mon.get("name", "")
        else:
            raw = str(mon)
        sp_id = to_id_str(raw)
        return _core(sp_id)

    def sanitize_stat_points(self, evs: list[int] | tuple | dict[str, int]) -> list[int]:
        """Enforces Champions format constraints: total stat points <= 66, max 32 per stat."""
        if isinstance(evs, dict):
            pts = [evs.get(k, 0) for k in STATS]
        elif isinstance(evs, (list, tuple)):
            pts = list(evs)[:6]
            if len(pts) < 6:
                pts += [0] * (6 - len(pts))
        else:
            pts = [32, 0, 16, 0, 18, 0]

        pts = [min(32, max(0, int(p))) for p in pts]
        tot = sum(pts)
        if tot > 66:
            scale = 66.0 / tot
            pts = [int(p * scale) for p in pts]
            while sum(pts) > 66:
                idx = max(range(6), key=lambda i: pts[i])
                pts[idx] -= 1
        return pts

    def _default_mon(self, species: str) -> dict:
        """Safe legal fallback for obscure/untiered species not present in usage databases."""
        dex = _pokedex()
        entry = dex.get(species, dex.get(_base(species), {}))
        abilities = entry.get("abilities", {})
        ability = to_id_str(abilities.get("0", "pressure"))

        # Look up legal learnset if available
        ls = self.gd.learnset.get(species, self.gd.learnset.get(_base(species), {}))
        learn_moves = list(ls.get("learnset", {}).keys()) if isinstance(ls, dict) else []
        moves = []
        if "protect" in learn_moves:
            moves.append("protect")
        # Add STAB / high utility moves from learnset if found
        for m in learn_moves:
            if m not in moves:
                moves.append(m)
            if len(moves) == 4:
                break
        if len(moves) < 4:
            fallbacks = ["protect", "tackle", "substitute", "rest"]
            for fb in fallbacks:
                if fb not in moves:
                    moves.append(fb)
                if len(moves) == 4:
                    break

        return {
            "species": species,
            "level": 50,
            "ability": ability,
            "item": "sitrusberry",
            "nature": "Serious",
            "moves": moves[:4],
            "evs": [32, 0, 16, 0, 18, 0],
            "ivs": [31] * 6,
        }

    def build_team(self, species_list: Iterable[Any]) -> list[dict]:
        """Constructs a full 6-mon team dictionary given a list of species from Team Preview."""
        norm_species = [self.normalize_species(s) for s in species_list]
        norm_species = [s for s in norm_species if s]

        # Ensure exactly 6 species
        if len(norm_species) < 6:
            for f in META_FILLERS:
                if f not in norm_species:
                    norm_species.append(f)
                if len(norm_species) == 6:
                    break
        norm_species = norm_species[:6]

        # 1. Whole team library lookup (exact 6 or 5-of-6)
        key = frozenset(norm_species)
        matched_team = None
        if key in self.lib and self.lib[key]:
            matched_team = self.lib[key][0][0]
        else:
            for sp in norm_species:
                k5 = key - {sp}
                if k5 in self.lib and self.lib[k5]:
                    matched_team = self.lib[k5][0][0]
                    break

        team_out = []
        for sp in norm_species:
            mon_dict = None
            if matched_team and sp in matched_team:
                item, ability, moves, spread_tuple = matched_team[sp]
                evs = [0] * 6
                nature = "Hardy"
                if spread_tuple is not None:
                    evs = spread_tuple[0]
                    nature = spread_tuple[1]
                m_list = list(moves)[:4]
                if len(m_list) < 4:
                    m_list += ["protect"] * (4 - len(m_list))
                mon_dict = {
                    "species": sp,
                    "level": 50,
                    "ability": ability or "intimidate",
                    "item": item or "sitrusberry",
                    "nature": nature or "Serious",
                    "moves": m_list,
                    "evs": self.sanitize_stat_points(evs),
                    "ivs": [31] * 6,
                }

            if mon_dict is None and sp in self.set_usage and self.set_usage[sp]:
                best_usage = self.set_usage[sp][0]
                item, ability, moves = best_usage[0], best_usage[1], list(best_usage[2])[:4]
                if len(moves) < 4:
                    moves += ["protect"] * (4 - len(moves))

                spreads = spread_prior(sp)
                if spreads:
                    evs, nature = spreads[0][0][0], spreads[0][0][1]
                else:
                    evs, nature = [32, 0, 16, 0, 18, 0], "Serious"

                mon_dict = {
                    "species": sp,
                    "level": 50,
                    "ability": ability,
                    "item": item,
                    "nature": nature,
                    "moves": moves,
                    "evs": self.sanitize_stat_points(evs),
                    "ivs": [31] * 6,
                }

            if mon_dict is None:
                mon_dict = self._default_mon(sp)

            team_out.append(mon_dict)

        return team_out


def create_dynamic_preview_pool(t_us: list[dict], t_them: list[dict]) -> TeamPreviewPool:
    """Helper to instantiate TeamPreviewPool for [t_us, t_them]."""
    return TeamPreviewPool([t_us, t_them])
