"""Sampling, schedule and supervision helpers for the human-warmstarted league."""
from __future__ import annotations

import math
import torch
from torch.nn import functional as F


def initialize_ema(state, saved=None):
    """Keep floating-point averages in float32 and preserve integer buffers."""
    source = state if saved is None else saved
    if source.keys() != state.keys():
        raise ValueError("EMA checkpoint keys do not match the model")
    result = {}
    for key, value in state.items():
        if source[key].shape != value.shape:
            raise ValueError(f"EMA checkpoint shape mismatch: {key}")
        dtype = torch.float32 if value.is_floating_point() else value.dtype
        result[key] = source[key].detach().to(device=value.device, dtype=dtype).clone()
    return result


@torch.no_grad()
def update_ema(ema, state, decay):
    for key, value in state.items():
        if ema[key].is_floating_point():
            ema[key].mul_(decay).add_(value.detach().float(), alpha=1 - decay)
        else:
            ema[key].copy_(value.detach())


def policy_temperature(model):
    return float(getattr(getattr(model, "module", model), "sampling_temperature", 1.0))


def temperature_logits(logits, temperature=1.0):
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("sampling temperature must be finite and positive")
    return logits.float() / temperature


def scheduled_lr(base_lr, steps, half_every, minimum):
    if half_every < 0 or minimum < 0 or base_lr <= 0 or minimum > base_lr:
        raise ValueError("invalid LR schedule")
    halvings = steps // half_every if half_every else 0
    return max(minimum, math.ldexp(base_lr, -min(halvings, 1024)))


def hidden_belief_targets(cat, num):
    """Training-only labels from the opponent's OWN view, before CTS masking.

    Never return these tensors in model history/observations. They are consumed
    only by the PPO loss, like reward and value targets.
    """
    own = cat[1, :, :2]
    return {
        "item": own[..., 1].copy(),
        "ability": own[..., 2].copy(),
        "moves": cat[1, :, 8:16, 0].reshape(-1, 2, 4).copy(),
        "valid": ((own[..., 0] != 0) & (own[..., 0] != 65535) & (num[1, :, :2, 0] > 0)).copy(),
    }


def belief_prediction_loss(output, targets):
    valid = targets["valid"].bool()
    if not valid.any():
        return output["belief_item_logits"].sum() * 0
    losses = []
    for name in ("item", "ability"):
        logits = output[f"belief_{name}_logits"].float()[valid]
        target = targets[name].long()[valid]
        # The final class represents no held item / absent ability.
        target = torch.where((target >= 0) & (target < logits.shape[-1] - 1), target, logits.shape[-1] - 1)
        losses.append(F.cross_entropy(logits, target))
    logits = output["belief_move_logits"].float()[valid]
    ids = targets["moves"].long()[valid]
    known = (ids > 0) & (ids < logits.shape[-1])
    multi_hot = torch.zeros_like(logits)
    multi_hot.scatter_add_(1, ids.clamp(0, logits.shape[-1] - 1), known.float())
    multi_hot.clamp_(max=1)
    # Balance positives/negatives rather than reward predicting every move absent.
    positive = F.softplus(-logits) * multi_hot
    negative = F.softplus(logits) * (1 - multi_hot)
    losses.append((positive.sum(-1) / multi_hot.sum(-1).clamp_min(1)
                   + negative.sum(-1) / (1 - multi_hot).sum(-1).clamp_min(1)).mean())
    return sum(losses) / len(losses)
