"""Explicit chance event branching for search (Nessie-style).

Instead of averaging over simulator seeds, we explicitly enumerate the important
chance events (damage rolls, crits, secondary effects like burn/flinch) and
compute the true expected value by summing probability * value for each outcome.

This gives exact expected values (up to the events we enumerate) with far fewer
simulations than seed averaging, and allows proper handling of low-probability
high-impact events (e.g. a 6.25% crit that KOs).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch

from ai_vgc.nn.encode import encode
from ai_vgc.nn.model import Policy
from ai_vgc.calc import damage_pct, effective_speed
from poke_env.battle import (
    DoubleBattle, Effect, Move, MoveCategory, Pokemon, Status, Target, Weather
)
from poke_env.data import GenData, to_id_str


@dataclass
class ChanceEvent:
    """A chance event that can occur during a turn."""
    description: str
    probability: float
    # Function to apply this event to a battle state
    apply_fn: Callable[[DoubleBattle], DoubleBattle]


@dataclass
class ChanceOutcome:
    """Result of a chance event branch."""
    battle: DoubleBattle
    probability: float
    description: str


def enumerate_damage_rolls(
    battle: DoubleBattle,
    attacker: Pokemon,
    defender: Pokemon,
    move: Move,
) -> list[tuple[float, float]]:
    """Enumerate possible damage outcomes for a move.

    Returns list of (damage_fraction, probability) pairs.
    Uses the 16 damage rolls (216/256 to 256/256 of max) with their probabilities.
    """
    # Pokemon damage formula has 16 possible rolls (216-255 + max)
    # Actually 100 possible rolls (85-100% of max damage in some gens)
    # For Gen 9: 16 rolls from 85% to 100% of max
    dmg_range = damage_pct(battle, attacker, defender, move)
    if dmg_range is None:
        return [(0.0, 1.0)]

    lo, hi = dmg_range
    hp_frac = defender.current_hp_fraction

    # The 16 damage rolls are roughly uniformly distributed between 85% and 100%
    # of max damage. Each has probability 1/16.
    rolls = 16
    outcomes = []
    for i in range(rolls):
        roll_frac = 0.85 + (0.15 * i / (rolls - 1))  # 0.85 to 1.0
        dmg = lo + (hi - lo) * roll_frac
        dmg_hp_frac = dmg / 100.0
        outcomes.append((dmg_hp_frac, 1.0 / rolls))

    return outcomes


def enumerate_crit(move: Move) -> list[tuple[float, float]]:
    """Enumerate crit outcomes: (crit_multiplier, probability).

    For Gen 9: crit rate is 1/16 (6.25%) for most moves, 1/8 (12.5%) for high-crit-ratio moves.
    Crit does 1.5x damage (in Gen 9, was 2x in older gens).
    """
    crit_ratio = move.crit_ratio if hasattr(move, 'crit_ratio') and move.crit_ratio else 0
    if crit_ratio >= 1:
        # High crit ratio (1/8)
        return [(1.5, 1/8), (1.0, 7/8)]
    else:
        # Standard crit rate (1/16)
        return [(1.5, 1/16), (1.0, 15/16)]


def enumerate_secondary_effects(move: Move) -> list[tuple[dict[str, Any], float]]:
    """Enumerate secondary effect outcomes (status, stat drops, flinch, etc.).

    Returns list of (effect_dict, probability) where effect_dict describes
    what happens (e.g. {"status": "brn"}, {"flinch": True}, {"boosts": {"spa": -1}}).
    """
    outcomes = []

    # Always include "no secondary effect"
    no_effect_prob = 1.0

    if move.secondary:
        for sec in move.secondary:
            chance = sec.get("chance", 0) / 100.0
            if chance > 0:
                effect = {}
                if "status" in sec:
                    effect["status"] = sec["status"]
                if "volatileStatus" in sec:
                    effect["volatile"] = sec["volatileStatus"]
                if "boosts" in sec:
                    effect["boosts"] = sec["boosts"]
                if "flinch" in sec:
                    effect["flinch"] = True
                outcomes.append((effect, chance))
                no_effect_prob -= chance

    if no_effect_prob > 0:
        outcomes.append(({}, no_effect_prob))

    return outcomes


def enumerate_accuracy(move: Move, attacker: Pokemon, defender: Pokemon) -> list[tuple[bool, float]]:
    """Enumerate hit/miss outcomes: (hit, probability)."""
    if move.accuracy is True:
        return [(True, 1.0)]
    acc = move.accuracy or 100
    # Apply accuracy/evasion modifiers
    acc_stage = attacker.boosts.get('accuracy', 0)
    eva_stage = defender.boosts.get('evasion', 0)
    stage = acc_stage - eva_stage
    stage = max(-6, min(6, stage))
    stage_mult = [3/9, 3/8, 3/7, 3/6, 3/5, 3/4, 1, 4/3, 5/3, 6/3, 7/3, 8/3, 9/3][stage + 6]
    final_acc = acc * stage_mult / 100
    final_acc = max(0.0, min(1.0, final_acc))
    return [(True, final_acc), (False, 1.0 - final_acc)]


def enumerate_protect(counter: int) -> list[tuple[bool, float]]:
    """Enumerate Protect success/failure.

    Protect counter: 0 = 100%, 1 = 50%, 2 = 25%, 3 = 12.5% (Gen 9)
    """
    if counter == 0:
        return [(True, 1.0)]
    elif counter == 1:
        return [(True, 0.5), (False, 0.5)]
    elif counter == 2:
        return [(True, 0.25), (False, 0.75)]
    else:
        return [(True, 0.125), (False, 0.875)]


def build_chance_branches(
    battle: DoubleBattle,
    our_action: str,
    opp_action: str,
    max_branches: int = 32,
) -> list[ChanceOutcome]:
    """Build explicit chance event branches for a pair of actions.

    This analyzes the actions and identifies the key random events:
    - Damage rolls for each attacking move
    - Crits for each attacking move
    - Secondary effects (burn, flinch, stat drops)
    - Accuracy (hit/miss)
    - Protect success

    Returns a list of ChanceOutcome, each with a modified battle state and probability.
    The total probability sums to 1.0.
    """
    # Parse actions to identify moves
    # This is a simplified version - in practice we'd need to parse the action strings
    # and simulate the turn with the battle engine to get the exact events

    # For now, return a single branch (no explicit branching)
    # The full implementation would require integrating with the battle simulator
    # to enumerate the actual random events that occur
    return [ChanceOutcome(battle=battle, probability=1.0, description="no explicit branching")]


def chance_branch_value(
    model: Policy,
    battle: DoubleBattle,
    our_action: np.ndarray,
    opp_actions: list[str],
    opp_probs: np.ndarray,
    rating: float,
    max_branches: int = 32,
) -> float:
    """Compute expected value using explicit chance branching.

    For each opponent action, we:
    1. Build chance branches for the (our_action, opp_action) pair
    2. Evaluate each branch with the value head
    3. Sum prob * value

    This replaces the seed-averaging approach.
    """
    from ai_vgc.nn.search import Bridge, _playout_values, after
    from poke_env.environment import DoublesEnv

    # This is a placeholder - the full implementation would:
    # 1. Use the bridge to play the turn
    # 2. Instead of averaging over seeds, enumerate chance events
    # 3. For each chance event, apply it and evaluate

    # For now, fall back to the existing seed-based approach
    # by calling the bridge with multiple seeds
    # (The bridge already does this internally)

    # TODO: Implement full chance branching by:
    # - Modifying sim_bridge.js to return chance event info
    # - Or building a local chance event simulator

    return 0.0  # Placeholder


class ChanceBranchSimulator:
    """Simulator that explicitly branches on chance events instead of using seeds.

    This would replace the sim_bridge.js for chance branching mode.
    It uses the Showdown simulator but intercepts random events to branch explicitly.
    """

    def __init__(self, showdown_path: str = "pokemon-showdown"):
        self.showdown_path = showdown_path
        # Would need to modify the Showdown battle engine to support explicit branching
        # For now, this is a design sketch

    def simulate_with_branching(
        self,
        state: dict,
        our_choice: str,
        opp_choice: str,
        value_fn: Callable[[DoubleBattle], float],
    ) -> float:
        """Simulate one turn with explicit chance branching.

        Returns the true expected value.
        """
        # Placeholder - would require deep integration with Showdown's battle engine
        # to intercept RNG calls and branch explicitly
        return 0.0


# Integration point: modify the search to use chance branching
# when a flag is enabled. For now, we keep the seed-based approach
# and add this as an optional enhancement.

def enable_chance_branching() -> bool:
    """Check if chance branching should be enabled (via env var or config)."""
    import os
    return os.environ.get("NESSIE_CHANCE_BRANCHING", "0") == "1"