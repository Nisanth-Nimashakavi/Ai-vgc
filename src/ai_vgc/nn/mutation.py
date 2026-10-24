"""Mutation augmentation for Pokemon data (Nessie-style).

Nessie's key insight: by randomizing stats, abilities, typing, moves, and items
during training, the network is forced to learn the *mechanics* of Pokemon
(type charts, damage formulas, ability interactions) rather than memorizing
specific matchups. This dramatically improves sample efficiency.

Half of Nessie's training games use mutated Pokemon. This creates a much
greater diversity of positions and encourages smoothness in the network's
learned representations.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ai_vgc.nn.encode import vocab, move_table, vocab_sizes, N_MV, T, TYPES, CATEGORIES, TARGETS


@dataclass
class MutationConfig:
    """Configuration for mutation augmentation."""
    # Probability of applying each mutation type
    p_species: float = 0.1      # Change species (changes base stats, types, abilities)
    p_ability: float = 0.15     # Change ability
    p_item: float = 0.1         # Change item
    p_moves: float = 0.2        # Replace one move
    p_stats: float = 0.1        # Perturb base stats
    p_types: float = 0.05       # Change type(s)
    p_nature: float = 0.1       # Change nature
    p_evs: float = 0.15         # Change EV spread
    p_ivs: float = 0.05         # Change IVs

    # Mutation intensity
    stat_noise_std: float = 0.1  # Gaussian noise on base stats (relative)
    ev_noise_std: float = 20     # Gaussian noise on EVs
    max_mutations_per_pokemon: int = 3  # Cap total mutations per Pokemon

    # Species mutation: when changing species, what to preserve
    preserve_moves_when_species_change: bool = False
    preserve_ability_when_species_change: bool = False

    # Overall probability that a given game uses mutation
    p_game_mutated: float = 0.5  # Nessie uses 0.5


# Global config instance
DEFAULT_MUTATION_CONFIG = MutationConfig()


# Cached data for mutations
_SPECIES_LIST: list[str] | None = None
_ABILITIES_LIST: list[str] | None = None
_ITEMS_LIST: list[str] | None = None
_MOVES_LIST: list[str] | None = None
_SPECIES_DATA: dict[str, dict] | None = None


def _load_showdown_data():
    """Load Pokemon data from Showdown for mutation reference."""
    global _SPECIES_LIST, _ABILITIES_LIST, _ITEMS_LIST, _MOVES_LIST, _SPECIES_DATA

    if _SPECIES_LIST is not None:
        return

    showdown_data = Path("pokemon-showdown/data")
    from ai_vgc.nn.encode import _ts_keys
    from poke_env.data import GenData

    g = GenData.from_gen(9)

    _SPECIES_LIST = sorted(g.pokedex.keys())
    _ABILITIES_LIST = _ts_keys("abilities.ts")
    _ITEMS_LIST = _ts_keys("items.ts")
    _MOVES_LIST = sorted(g.moves.keys())

    # Build species data cache
    _SPECIES_DATA = {}
    for sp in _SPECIES_LIST:
        entry = g.pokedex[sp]
        _SPECIES_DATA[sp] = {
            "baseStats": entry.get("baseStats", {}),
            "types": entry.get("types", []),
            "abilities": entry.get("abilities", {}),
            "weight": entry.get("weightkg", 0),
        }


def _vocab_id(name: str, table: str) -> int:
    """Get vocab ID for a name, or 0 if not found."""
    v = vocab()
    return v.get(table, {}).get(name, 0)


def mutate_species(tok_cat: torch.Tensor, tok_num: torch.Tensor,
                   batch_idx: int, slot_idx: int, config: MutationConfig) -> None:
    """Mutate the species at a given token slot."""
    _load_showdown_data()

    if not _SPECIES_LIST:
        return

    # Pick a random new species
    new_species = random.choice(_SPECIES_LIST)
    data = _SPECIES_DATA.get(new_species, {})

    # Update tok_cat: species id
    tok_cat[batch_idx, slot_idx, 0] = _vocab_id(new_species, "species")

    # Update tok_num: base stats, types
    base_stats = data.get("baseStats", {})
    stats_order = ["hp", "atk", "def", "spa", "spd", "spe"]
    # base_stats are at indices 7-12 (after present, ours, not_ours, hp_frac, fainted, act_a, act_b)
    for i, stat in enumerate(stats_order):
        val = base_stats.get(stat, 0) / 255
        tok_num[batch_idx, slot_idx, 7 + i] = val

    # Update types (indices after base stats + maxhp + known stats)
    # tok_num layout: present(1), ours(1), not_ours(1), hp_frac(1), fainted(1), act_a(1), act_b(1),
    # base_stats(6), maxhp(1), known_stats(5), types(18), statuses(8), boosts(7), effects(18), flags(9)
    type_start = 1 + 1 + 1 + 1 + 1 + 1 + 1 + 6 + 1 + 5  # = 19
    new_types = data.get("types", [])
    for i, t in enumerate(TYPES):
        tok_num[batch_idx, slot_idx, type_start + i] = 1.0 if t in new_types else 0.0

    # Update abilities in tok_cat
    abilities = data.get("abilities", {})
    # Pick first ability
    first_ability = list(abilities.values())[0] if abilities else ""
    tok_cat[batch_idx, slot_idx, 2] = _vocab_id(first_ability, "abilities")

    # Update weight
    tok_num[batch_idx, slot_idx, -1] = min(data.get("weight", 0), 500) / 500


def mutate_ability(tok_cat: torch.Tensor, batch_idx: int, slot_idx: int, config: MutationConfig) -> None:
    """Mutate the ability at a given token slot."""
    _load_showdown_data()
    if not _ABILITIES_LIST:
        return

    new_ability = random.choice(_ABILITIES_LIST)
    tok_cat[batch_idx, slot_idx, 2] = _vocab_id(new_ability, "abilities")


def mutate_item(tok_cat: torch.Tensor, batch_idx: int, slot_idx: int, config: MutationConfig) -> None:
    """Mutate the item at a given token slot."""
    _load_showdown_data()
    if not _ITEMS_LIST:
        return

    new_item = random.choice(_ITEMS_LIST)
    tok_cat[batch_idx, slot_idx, 1] = _vocab_id(new_item, "items")


def mutate_moves(tok_cat: torch.Tensor, mv_cat: torch.Tensor, mv_dyn: torch.Tensor,
                 batch_idx: int, slot_idx: int, config: MutationConfig) -> None:
    """Mutate one move at a given token slot."""
    _load_showdown_data()
    if not _MOVES_LIST:
        return

    # Pick a random move slot to replace
    move_slot = random.randrange(4)
    new_move = random.choice(_MOVES_LIST)

    # Update mv_cat
    mv_cat[batch_idx, slot_idx, move_slot] = _vocab_id(new_move, "moves")

    # Update mv_dyn: new move has full PP, not used last turn
    # Note: actual PP depends on the move, but we don't have that here
    # Just mark as "not guessed" and not used last turn
    mv_dyn[batch_idx, slot_idx, move_slot] = torch.tensor([1.0, 0.0])


def mutate_stats(tok_num: torch.Tensor, batch_idx: int, slot_idx: int, config: MutationConfig) -> None:
    """Perturb base stats with Gaussian noise."""
    # Base stats are at indices 7-12 (after present, ours, not_ours, hp_frac, fainted, act_a, act_b)
    for i in range(6):
        idx = 7 + i
        noise = random.gauss(0, config.stat_noise_std)
        tok_num[batch_idx, slot_idx, idx] = torch.clamp(tok_num[batch_idx, slot_idx, idx] + noise, 0.0, 1.0)


def mutate_types(tok_num: torch.Tensor, batch_idx: int, slot_idx: int, config: MutationConfig) -> None:
    """Randomly change one type."""
    _load_showdown_data()
    if not TYPES:
        return

    type_start = 19  # See mutate_species
    # Pick one type to flip
    current_types = [i for i, t in enumerate(TYPES) if tok_num[batch_idx, slot_idx, type_start + i] > 0.5]

    # Either add a type or remove one (or both)
    if random.random() < 0.5 and current_types:
        # Remove a type
        remove_idx = random.choice(current_types)
        tok_num[batch_idx, slot_idx, type_start + remove_idx] = 0.0
    else:
        # Add a type (different from current)
        available = [i for i in range(len(TYPES)) if i not in current_types]
        if available:
            add_idx = random.choice(available)
            tok_num[batch_idx, slot_idx, type_start + add_idx] = 1.0


def mutate_nature(tok_num: torch.Tensor, batch_idx: int, slot_idx: int, config: MutationConfig) -> None:
    """Change nature (affects stat boosts implicitly via EV interpretation)."""
    # Nature is encoded implicitly in the stat distribution
    # For simplicity, we just perturb EVs which achieves similar effect
    mutate_evs(tok_num, batch_idx, slot_idx, config)


def mutate_evs(tok_num: torch.Tensor, batch_idx: int, slot_idx: int, config: MutationConfig) -> None:
    """Perturb EV spread with Gaussian noise."""
    # Known stats (visible stats after EVs/nature) are at indices:
    # present(1), ours(1), not_ours(1), hp_frac(1), fainted(1), act_a(1), act_b(1),
    # base_stats(6), maxhp(1), known_stats(5) -> known_stats start at index 14
    ev_start = 14
    for i in range(5):  # 5 known stats (atk, def, spa, spd, spe)
        idx = ev_start + i
        noise = random.gauss(0, config.ev_noise_std) / 255
        tok_num[batch_idx, slot_idx, idx] = torch.clamp(tok_num[batch_idx, slot_idx, idx] + noise, 0.0, 1.0)


def mutate_ivs(tok_num: torch.Tensor, batch_idx: int, slot_idx: int, config: MutationConfig) -> None:
    """Perturb IVs (not directly encoded, but affects stats)."""
    # IVs are not directly in tok_num (they're assumed 31 for opponent, known for us)
    # This would require modifying the damage calculation, so skip for now
    pass


MUTATION_FUNCS = [
    ("species", mutate_species, "p_species"),
    ("ability", mutate_ability, "p_ability"),
    ("item", mutate_item, "p_item"),
    ("moves", mutate_moves, "p_moves"),
    ("stats", mutate_stats, "p_stats"),
    ("types", mutate_types, "p_types"),
    ("nature", mutate_nature, "p_nature"),
    ("evs", mutate_evs, "p_evs"),
    ("ivs", mutate_ivs, "p_ivs"),
]


def apply_mutations(
    b: dict[str, torch.Tensor],
    config: MutationConfig = DEFAULT_MUTATION_CONFIG,
) -> dict[str, torch.Tensor]:
    """Apply random mutations to a batch of encoded positions.

    This is the core mutation augmentation function. It modifies the input
    tensors in-place (on a copy) to create mutated versions of the Pokemon.

    Args:
        b: Batch dict with keys tok_cat, tok_num, mv_cat, mv_dyn
        config: MutationConfig controlling mutation probabilities

    Returns:
        Mutated batch (new dict, original unchanged)
    """
    # Only mutate if game is selected for mutation
    if random.random() > config.p_game_mutated:
        return b

    # Clone to avoid modifying original
    b = {k: v.clone() for k, v in b.items()}

    B = b["tok_cat"].shape[0]
    tok_cat = b["tok_cat"]
    tok_num = b["tok_num"]
    mv_cat = b["mv_cat"]
    mv_dyn = b["mv_dyn"]

    for batch_idx in range(B):
        # Determine which Pokemon to mutate (any of the 12 tokens that are present)
        present = tok_num[batch_idx, :, 0] > 0.5
        present_indices = torch.where(present)[0].tolist()

        if not present_indices:
            continue

        for slot_idx in present_indices:
            # Count mutations for this Pokemon
            n_mutations = 0

            # Apply each mutation type with its probability
            for name, func, prob_attr in MUTATION_FUNCS:
                if n_mutations >= config.max_mutations_per_pokemon:
                    break

                prob = getattr(config, prob_attr)
                if random.random() < prob:
                    if name == "species":
                        func(tok_cat, tok_num, batch_idx, slot_idx, config)
                    elif name in ("ability", "item"):
                        func(tok_cat, batch_idx, slot_idx, config)
                    elif name == "moves":
                        func(tok_cat, mv_cat, mv_dyn, batch_idx, slot_idx, config)
                    elif name == "stats":
                        func(tok_num, batch_idx, slot_idx, config)
                    elif name == "types":
                        func(tok_num, batch_idx, slot_idx, config)
                    elif name in ("nature", "evs"):
                        func(tok_num, batch_idx, slot_idx, config)
                    elif name == "ivs":
                        func(tok_num, batch_idx, slot_idx, config)

                    n_mutations += 1

    return b


def apply_mutations_to_batch(
    batch: dict[str, torch.Tensor],
    config: MutationConfig = DEFAULT_MUTATION_CONFIG,
) -> dict[str, torch.Tensor]:
    """Apply mutations to a batch, handling the case where only some games are mutated.

    For games that are mutated, we also need to update the damage features (dmg)
    to reflect the new stats/types/abilities. This is complex because dmg depends
    on the actual battle mechanics.

    For now, we only mutate the static features (tok_cat, tok_num, mv_cat, mv_dyn)
    and leave dmg as-is. This introduces some noise but the network can learn
    to be robust to it. A full implementation would recompute dmg after mutation.
    """
    return apply_mutations(batch, config)


def create_mutation_config_from_args(args) -> MutationConfig:
    """Create MutationConfig from argparse args."""
    return MutationConfig(
        p_species=args.mut_species,
        p_ability=args.mut_ability,
        p_item=args.mut_item,
        p_moves=args.mut_moves,
        p_stats=args.mut_stats,
        p_types=args.mut_types,
        p_nature=args.mut_nature,
        p_evs=args.mut_evs,
        p_ivs=args.mut_ivs,
        stat_noise_std=args.mut_stat_noise,
        ev_noise_std=args.mut_ev_noise,
        max_mutations_per_pokemon=args.mut_max_per_pokemon,
        p_game_mutated=args.mut_game_prob,
    )