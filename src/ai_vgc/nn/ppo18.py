"""Probability and observation contracts for corrected-engine self-play."""

from __future__ import annotations

import numpy as np
import torch
from ai_vgc.nn.rl18_extensions import temperature_logits


def mask_closed_observation(cat, num) -> None:
    """Conservative CTS inputs, without reserves or private-set arithmetic.

    This base mask removes private fields. The public-history encoder then
    restores announced items/abilities and remembers seen reserves and moves.
    """
    cat[..., 4:8, 1:3] = 0
    cat[..., 6:8, :] = 0
    num[..., 6:8, :] = 0
    num[..., 4:6, 7] = 0
    num[..., 4:6, 11] = 0
    hp = num[..., 4:6, 0]
    if isinstance(hp, np.ndarray):
        num[..., 4:6, 0] = np.ceil(np.clip(hp, 0, 1) * 100 - 1e-4) / 100
    else:
        num[..., 4:6, 0] = (hp.clamp(0, 1) * 100 - 1e-4).ceil() / 100
    num[..., 8:16, 4:12] = 0
    # Weather rocks, Terrain Extender and Court Change can make exact
    # remaining durations depend on an unrevealed opponent item.
    num[..., 16, 0:2] = 0
    num[..., 16, 5:8] = num[..., 16, 5:8] > 0


def masked_action_logits(logits: torch.Tensor, legal: torch.Tensor) -> torch.Tensor:
    return logits.float().masked_fill(~legal, -1e9)


def preview_ppo_loss(output, actions, old_logps, old_values, outcomes, clip_eps, entropy_coef, temperature=1.0):
    """Undiscounted terminal game outcome credits the sampled preview decision."""
    dist = torch.distributions.Categorical(logits=temperature_logits(output["preview_plan"], temperature))
    advantage = outcomes - old_values
    ratio = (dist.log_prob(actions) - old_logps).exp()
    policy_loss = -torch.minimum(
        ratio * advantage, ratio.clamp(1 - clip_eps, 1 + clip_eps) * advantage,
    ).mean()
    value_loss = torch.nn.functional.mse_loss(output["preview_value"].float(), outcomes)
    entropy = dist.entropy().mean()
    return policy_loss + 0.5 * value_loss - entropy_coef * entropy, policy_loss, value_loss, entropy


def conditioned_b_mask(legal: torch.Tensor, action_a: torch.Tensor) -> torch.Tensor:
    result = legal.clone()
    switches = (action_a == 16) | (action_a == 17)
    rows = torch.arange(len(action_a), device=action_a.device)
    result[rows[switches], action_a[switches]] = False
    if result.shape[-1] == 35:
        result[action_a >= 19, 19:] = False
    # Two empty slots and one remaining reserve allow either slot to pass.
    # They cannot both pass: if A waits, B must use the remaining reserve.
    replacement_b = result[:, 16:18].any(-1) & ~result[:, :16].any(-1)
    result[replacement_b & (action_a == 18), 18] = False
    empty = ~result.any(dim=-1)
    if empty.any():
        pass_legal = legal[:, 18]
        use_pass = empty & pass_legal
        result[use_pass, 18] = True
        restore_legal = empty & ~pass_legal
        result[restore_legal] = legal[restore_legal]
    return result


def generalized_advantages(rewards, values, dones, bootstrap, gamma, gae_lambda, elapsed_turns=None):
    advantages = torch.empty_like(values)
    last = torch.zeros_like(bootstrap)
    for t in reversed(range(len(rewards))):
        next_value = bootstrap if t == len(rewards) - 1 else values[t + 1]
        nonterminal = 1 - dones[t]
        discount = gamma if elapsed_turns is None else gamma ** elapsed_turns[t]
        trace = gae_lambda if elapsed_turns is None else gae_lambda ** elapsed_turns[t]
        delta = rewards[t] + discount * next_value * nonterminal - values[t]
        last = delta + discount * trace * nonterminal * last
        advantages[t] = last
    return advantages, advantages + values


def compute_aux_losses(
    output: dict[str, torch.Tensor],
    target_weather: torch.Tensor,
    target_terrain: torch.Tensor,
    target_survives: torch.Tensor,
    target_hp: torch.Tensor,
    target_damage: torch.Tensor,
    combat_mask: torch.Tensor,
    active_alive_mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute dense auxiliary supervision losses from ground-truth transition targets.

    Predicts:
    1. Resulting weather (next turn): 5 classes (0: None, 1: Rain, 2: Sun, 3: Sand, 4: Snow)
    2. Resulting terrain (next turn): 5 classes (0: None, 1: Electric, 2: Grassy, 3: Psychic, 4: Misty)
    3. Who survives: 4 active slots binary classification (1: alive, 0: fainted)
    4. Resulting HP fraction: 4 active slots in [0, 1]
    5. Damage dealt (HP delta): 4 active slots in [0, 1]

    Args:
        output: Model output dict containing aux predictions.
        target_weather: [B] int64 in 0..4 (weather at t+1)
        target_terrain: [B] int64 in 0..4 (terrain at t+1)
        target_survives: [B, 4] float32 in {0.0, 1.0} (alive flag at t+1)
        target_hp: [B, 4] float32 in [0.0, 1.0] (HP fraction at t+1)
        target_damage: [B, 4] float32 in [0.0, 1.0] (damage taken / HP delta)
        combat_mask: [B] bool (True if transition is within same battle and elapsed_turns > 0)
        active_alive_mask: [B, 4] bool (True if active slot was alive at turn t)

    Returns:
        total_aux_loss: scalar tensor with gradient
        diag: dict of scalar loss components
    """
    dev = target_weather.device
    zero = torch.tensor(0.0, device=dev)

    # 1. Resulting weather (next turn)
    if combat_mask.any() and "aux_weather" in output:
        loss_weather = torch.nn.functional.cross_entropy(
            output["aux_weather"][combat_mask],
            target_weather[combat_mask],
        )
    else:
        loss_weather = zero

    # 2. Resulting terrain (next turn)
    if combat_mask.any() and "aux_terrain" in output:
        loss_terrain = torch.nn.functional.cross_entropy(
            output["aux_terrain"][combat_mask],
            target_terrain[combat_mask],
        )
    else:
        loss_terrain = zero

    # 3. Active Pokémon survival & HP/damage targets
    valid_slots = combat_mask.unsqueeze(-1) & active_alive_mask  # [B, 4]

    if valid_slots.any() and "aux_survives" in output:
        loss_survival = torch.nn.functional.binary_cross_entropy_with_logits(
            output["aux_survives"][valid_slots],
            target_survives[valid_slots].float(),
        )
    else:
        loss_survival = zero

    if valid_slots.any() and "aux_hp" in output:
        loss_hp = torch.nn.functional.smooth_l1_loss(
            output["aux_hp"][valid_slots],
            target_hp[valid_slots].float(),
        )
    else:
        loss_hp = zero

    if valid_slots.any() and "aux_damage" in output:
        loss_damage = torch.nn.functional.smooth_l1_loss(
            output["aux_damage"][valid_slots],
            target_damage[valid_slots].float(),
        )
    else:
        loss_damage = zero

    total_aux = (
        1.0 * loss_weather
        + 1.0 * loss_terrain
        + 0.5 * loss_survival
        + 1.0 * loss_hp
        + 1.0 * loss_damage
    )

    diag = {
        "aux_wea": loss_weather.detach(),
        "aux_ter": loss_terrain.detach(),
        "aux_surv": loss_survival.detach(),
        "aux_hp": loss_hp.detach(),
        "aux_dmg": loss_damage.detach(),
    }
    return total_aux, diag
