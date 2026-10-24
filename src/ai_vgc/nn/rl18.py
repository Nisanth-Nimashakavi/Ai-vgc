"""Fresh random-initialized PPO self-play using the corrected Champions engine.

No old model, heuristic dataset, imitation anchor or historical league is read.
The battle policy and a full-six preview policy start from fresh parameters.
Bring-four, ordered leads, normal/Mega moves and knockout replacements all
learn from outcomes. Public reveal memory is separate from private simulator sets.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import random
import threading
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
import vgc_engine

from ai_vgc.nn.model18 import Transformer18
from ai_vgc.nn.rl18_extensions import (policy_temperature, temperature_logits, scheduled_lr,
                                      belief_prediction_loss, initialize_ema, update_ema)
from ai_vgc.nn.training_control18 import DampingSchedule, SkillCheckState, validate_skill_check
from ai_vgc.nn.preview18 import TeamPreviewPool, PreviewDecision, PreviewResult, PREVIEW_PLANS
from ai_vgc.nn.history18 import known_history, apply_known_active_sets
from ai_vgc.nn.ppo18 import (
    conditioned_b_mask, generalized_advantages, mask_closed_observation,
    masked_action_logits, preview_ppo_loss, compute_aux_losses,
)
from ai_vgc.nn.rl17_selfplay import load_pool_teams
from ai_vgc.nn.batch_collector18 import BatchCollector

ACTION_CONTRACT = "model18-midturn-pivot-recharge-v3"


class FreshBattleSlot:
    def __init__(self):
        self.battle = None
        self.done = True
        self.preview_decision = None

    def start(self, ours, theirs, learner_side):
        self.learner_side = learner_side
        self.opp_side = 1 - self.learner_side
        self.seed = random.getrandbits(64)
        teams = (ours, theirs) if learner_side == 0 else (theirs, ours)
        self.battle = vgc_engine.Battle.from_teams(
            *teams,
            format="gen9championsvgc2026regmc", stat_points=True,
            seed=self.seed, decision_phases=True,
        )
        self.turn, self.done = 1, False


def sample_actions(model, cat, num, masks, history=None):
    history = history or {}
    scorer = getattr(model, "score_slot_b", None) or getattr(getattr(model, "module", None), "score_slot_b", None)
    if scorer is not None:
        out = model(cat, num, return_hidden=True, **history)
    else:
        out = model(cat, num, **history)
    ma = masks[:, 0].clone()
    empty_a = ~ma.any(dim=-1)
    if empty_a.any():
        ma[empty_a, 18] = True
    logits_a = temperature_logits(out["policy_a"], policy_temperature(model))
    da = torch.distributions.Categorical(logits=masked_action_logits(logits_a, ma))
    a = da.sample()
    mb = conditioned_b_mask(masks[:, 1], a)
    if scorer is not None and "_hidden" in out:
        policy_b = scorer(out["_hidden"], a)
    else:
        policy_b = model(cat, num, action_a=a, **history)["policy_b"]
    logits_b = temperature_logits(policy_b, policy_temperature(model))
    db = torch.distributions.Categorical(logits=masked_action_logits(logits_b, mb))
    b = db.sample()
    return a, b, da.log_prob(a) + db.log_prob(b), out["value"].squeeze(-1).float(), mb, logits_a, logits_b


def sample_actions_pair(policy, opponent, cat, num, masks, history, is_mirror=False):
    """Samples actions for learner (cat[0]) and opponent (cat[1]).
    When is_mirror is True (or opponent is policy), batches both perspectives
    into a single transformer pass + score_slot_b, reducing forward passes per turn from 4 to 1."""
    if is_mirror or opponent is policy:
        B = cat.shape[1]
        cat_both = torch.cat([cat[0], cat[1]], dim=0)
        num_both = torch.cat([num[0], num[1]], dim=0)
        masks_both = torch.cat([masks[0], masks[1]], dim=0)
        history_both = {k: torch.cat([v[0], v[1]], dim=0) for k, v in history.items()}
        a_both, b_both, logps_both, values_both, mb_both, logits_a, logits_b = sample_actions(
            policy, cat_both, num_both, masks_both, history_both
        )
        ours = (a_both[:B], b_both[:B], logps_both[:B], values_both[:B], mb_both[:B], logits_a[:B], logits_b[:B])
        theirs = (a_both[B:], b_both[B:], logps_both[B:], values_both[B:], mb_both[B:], logits_a[B:], logits_b[B:])
        return ours, theirs
    else:
        ours = sample_actions(policy, cat[0], num[0], masks[0], {k: v[0] for k, v in history.items()})
        theirs = sample_actions(opponent, cat[1], num[1], masks[1], {k: v[1] for k, v in history.items()})
        return ours, theirs


def sample_pivots_pair(ours, theirs, masks):
    """Commit a bench choice for each selected pivot move before the turn."""
    sampled = []
    for perspective, decisions in enumerate((ours, theirs)):
        da = torch.distributions.Categorical(logits=masked_action_logits(decisions[5], masks[perspective, :, 0]))
        a = da.sample()
        mask_b = masks[perspective, :, 1].clone()
        both_pivot = mask_b[:, 16:18].any(dim=-1) & (a >= 16) & (a <= 17)
        if both_pivot.any():
            rows = torch.where(both_pivot)[0]
            mask_b[rows, a[rows]] = False
            mask_b[rows, 18] = ~mask_b[rows, 16:18].any(dim=-1)
        db = torch.distributions.Categorical(logits=masked_action_logits(decisions[6], mask_b))
        b = db.sample()
        sampled.append((a, b, da.log_prob(a) + db.log_prob(b), mask_b))
    return sampled[0], sampled[1]


class Collector:
    def __init__(self, pool, count, device, max_turns, belief_bank=None):
        self.pool = pool
        self.belief_bank = belief_bank
        self.beliefs = [[None, None] for _ in range(count)]
        self.belief_features = (torch.empty((2, count, 17, 16), dtype=torch.float32,
                               pin_memory=device.type == "cuda") if belief_bank is not None else None)
        self.slots = [FreshBattleSlot() for _ in range(count)]
        self.preview_results = []
        self.device, self.max_turns = device, max_turns
        pin = device.type == "cuda"
        self.cat = torch.empty((2, count, 17, 4), dtype=torch.long, pin_memory=pin)
        self.num = torch.empty((2, count, 17, 16), dtype=torch.float32, pin_memory=pin)
        self.mask = torch.empty((2, count, 2, 35), dtype=torch.bool, pin_memory=pin)
        self.history = {
            "history_cat": torch.empty((2, count, 12, 3), dtype=torch.long, pin_memory=pin),
            "history_num": torch.empty((2, count, 12, 6), pin_memory=pin),
            "history_moves": torch.empty((2, count, 12, 4), dtype=torch.long, pin_memory=pin),
            "history_flags": torch.empty((2, count, 12, 6), pin_memory=pin),
        }
        self.full_teams = np.zeros((count, 2), dtype=np.int64)
        self.native_picks = np.zeros((count, 2, 4), dtype=np.int64)
        self.learner_sides = np.zeros(count, dtype=np.int64)
        self.elapsed_turns = np.zeros(count, dtype=np.float32)
        self.replacement_decisions = self.mega_declarations = 0

    @torch.no_grad()
    def reset_done(self, model, opponent, amp, *, balanced=False):
        indices = [i for i, slot in enumerate(self.slots) if slot.done]
        if not indices:
            return
        ours = [random.randrange(len(self.pool)) for _ in indices]
        theirs = [random.randrange(len(self.pool)) for _ in indices]
        sampled = []
        for policy, a, b in ((model, ours, theirs), (opponent, theirs, ours)):
            cat, num, moves = [torch.from_numpy(x).to(self.device) for x in self.pool.observations(a, b)]
            with amp():
                output = policy(preview_cat=cat, preview_num=num, preview_moves=moves, preview_only=True)
                dist = torch.distributions.Categorical(logits=temperature_logits(output["preview_plan"], policy_temperature(policy)))
                plans = dist.sample()
            sampled.append((plans.cpu().tolist(), dist.log_prob(plans).cpu().tolist(),
                            output["preview_value"].float().cpu().tolist()))
        for j, i in enumerate(indices):
            slot = self.slots[i]
            side = i % 2 if balanced else random.randrange(2)
            slot.start(self.pool.selected_json(ours[j], sampled[0][0][j]),
                       self.pool.selected_json(theirs[j], sampled[1][0][j]), side)
            if self.belief_bank is not None:
                slot.battle.enable_belief_observations19()
                self.beliefs[i] = [self.belief_bank.tracker(), self.belief_bank.tracker()]
            slot.preview_decision = PreviewDecision(
                ours[j], theirs[j], sampled[0][0][j], sampled[0][1][j], sampled[0][2][j],
            )
            self.full_teams[i] = (ours[j], theirs[j])
            self.learner_sides[i] = side
            picks = (PREVIEW_PLANS[sampled[0][0][j]], PREVIEW_PLANS[sampled[1][0][j]])
            self.native_picks[i] = picks if side == 0 else picks[::-1]

    def encode(self):
        cat, num, mask = self.cat.numpy(), self.num.numpy(), self.mask.numpy()
        public = []
        for i, slot in enumerate(self.slots):
            for k, side in enumerate((slot.learner_side, slot.opp_side)):
                if self.belief_bank is not None:
                    features = self.beliefs[i][k].features(slot.battle.public_view19(side))
                    self.belief_features[k, i].numpy()[:] = np.frombuffer(features, dtype="<f4").reshape(17, 16)
                enc = slot.battle.encode_17(side)
                cat[k, i] = np.frombuffer(enc["tok_cat"], np.uint32).reshape(17, 4)
                num[k, i] = np.frombuffer(enc["tok_num"], np.float32).reshape(17, 16)
                if self.belief_bank is not None:
                    # Transform copies hidden foe Speed; it is not owner knowledge.
                    for actor in range(2):
                        if self.belief_features[k, i, actor, 15] != 0:
                            num[k, i, actor, 7] = 0
                for actor in range(2):
                    mask[k, i, actor] = slot.battle.legal_action_mask_18(side, actor)
            public.append(np.frombuffer(slot.battle.public_history_18(), dtype="<u4").reshape(2, 2, 6, 20))
        mask_closed_observation(cat, num)
        facts = np.stack(public, axis=1)
        apply_known_active_sets(cat, facts, self.learner_sides)
        encoded_history = known_history(self.pool, self.full_teams, self.native_picks, self.learner_sides, facts)
        for target, data in zip(self.history.values(), encoded_history):
            target.numpy()[:] = data
        history = {k: t.to(self.device, non_blocking=True) for k, t in self.history.items()}
        if self.belief_features is not None:
            history["belief_features"] = self.belief_features.to(self.device, non_blocking=True)
        return (*[t.to(self.device, non_blocking=True) for t in (self.cat, self.num, self.mask)], history)

    def pivot_masks(self, actions):
        masks = torch.zeros((2, len(self.slots), 2, 35), dtype=torch.bool, device=self.device)
        masks[..., 18] = True
        return masks

    def step(self, actions, pivots=None):
        rewards = np.zeros(len(self.slots), np.float32)
        dones = np.zeros(len(self.slots), np.float32)
        games = capped = 0
        if pivots is None:
            pivots = np.full((len(self.slots), 4), 18, dtype=np.uint8)
        for i, (slot, (la, lb, oa, ob)) in enumerate(zip(self.slots, actions)):
            replacement = slot.battle.needs_replacements()
            before = slot.battle.turn
            pa, pb, qa, qb = [int(x) for x in pivots[i]]
            if hasattr(slot.battle, "actions_18_to_choices_with_pivots"):
                ours = slot.battle.actions_18_to_choices_with_pivots(slot.learner_side, int(la), int(lb), pa, pb)
                theirs = slot.battle.actions_18_to_choices_with_pivots(slot.opp_side, int(oa), int(ob), qa, qb)
            else:
                ours = slot.battle.actions_18_to_choices(slot.learner_side, int(la), int(lb))
                theirs = slot.battle.actions_18_to_choices(slot.opp_side, int(oa), int(ob))
            self.replacement_decisions += int(replacement and any(c[0] == "switch" for c in ours))
            self.mega_declarations += sum(c[0] == "mega" for c in ours)
            choices = (ours, theirs) if slot.learner_side == 0 else (theirs, ours)
            result = slot.battle.step_move(*choices)
            self.elapsed_turns[i] = slot.battle.turn - before
            slot.turn = slot.battle.turn
            terminal = result != "continue"
            truncated = not terminal and slot.turn >= self.max_turns
            if terminal or truncated:
                if result in ("p1_win", "p2_win"):
                    winner = 0 if result == "p1_win" else 1
                    rewards[i] = 1 if winner == slot.learner_side else -1
                dones[i] = 1
                slot.done = True
                games += 1
                capped += int(truncated)
                # Capped games have no known terminal outcome for preview.
                if terminal:
                    self.preview_results.append(PreviewResult(slot.preview_decision, float(rewards[i])))
        return torch.from_numpy(rewards).to(self.device), torch.from_numpy(dones).to(self.device), games, capped


@torch.no_grad()
def evaluate(model, opponent, teams, count, device, amp, max_turns, belief_bank=None):
    collector = Collector(teams, count, device, max_turns, belief_bank)
    collector.reset_done(model, opponent, amp, balanced=True)
    scores = torch.zeros(count, device=device)
    active = torch.ones(count, dtype=torch.bool, device=device)
    for _ in range(3 * max_turns):
        cat, num, masks, history = collector.encode()
        with amp():
            ours, theirs = sample_actions_pair(model, opponent, cat, num, masks, history, is_mirror=(model is opponent))
        actions = torch.stack((ours[0], ours[1], theirs[0], theirs[1]), dim=1).cpu().numpy()
        pivot_masks = collector.pivot_masks(actions)
        ours_pivot, theirs_pivot = sample_pivots_pair(ours, theirs, pivot_masks)
        pivots = torch.stack((ours_pivot[0], ours_pivot[1], theirs_pivot[0], theirs_pivot[1]), dim=1).cpu().numpy()
        # Finished slots are not stepped again, and their first result is retained.
        for i, slot in enumerate(collector.slots):
            if not active[i]:
                continue
            sub = Collector.__new__(Collector)
            sub.slots, sub.device, sub.max_turns = [slot], device, max_turns
            sub.preview_results = []
            sub.elapsed_turns = np.zeros(1, dtype=np.float32)
            sub.replacement_decisions = sub.mega_declarations = 0
            reward, done, _, _ = sub.step(actions[i:i + 1], pivots[i:i + 1])
            if done[0]:
                scores[i] = reward[0]
                active[i] = False
        if not active.any():
            break
    return {"games": count, "wins": int((scores > 0).sum()), "losses": int((scores < 0).sum()), "draws_or_capped": int((scores == 0).sum())}


@torch.no_grad()
def evaluate_skill(model, opponent, teams, count, batch_size, device, amp, max_turns,
                   belief_bank=None, seed=20261022):
    """Fixed-seed, side-balanced batches; preserve training RNG and count every game."""
    validate_skill_check(1, count, batch_size, 0.51)
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    modes = model.training, opponent.training
    result = dict(games=count, wins=0, losses=0, draws_or_capped=0)
    try:
        model.eval()
        opponent.eval()
        with torch.random.fork_rng(devices=devices), (torch.cuda.device(device) if devices else nullcontext()):
            torch.random.default_generator.manual_seed(seed)
            if devices:
                torch.cuda.manual_seed(seed)
            for start in range(0, count, batch_size):
                size = min(batch_size, count - start)
                collector = BatchCollector(teams, size, device, max_turns, belief_bank, seed=seed + start)
                collector.reset_done(model, opponent, amp, balanced=True)
                active = torch.ones(size, dtype=torch.bool, device=device)
                scores = torch.zeros(size, device=device)
                for _ in range(3 * max_turns):
                    # The native batch API steps every slot. Replace finished
                    # battles before another step; active/scores below retain
                    # only the first result from each original evaluation slot.
                    if collector.done.any():
                        collector.reset_done(model, opponent, amp, balanced=True)
                    cat, num, masks, history = collector.encode()
                    with amp():
                        ours, theirs = sample_actions_pair(model, opponent, cat, num, masks, history,
                                                          is_mirror=(model is opponent))
                    actions = torch.stack((ours[0], ours[1], theirs[0], theirs[1]), dim=1).cpu().numpy()
                    pivot_masks = collector.pivot_masks(actions)
                    p1, p2 = sample_pivots_pair(ours, theirs, pivot_masks)
                    pivots = torch.stack((p1[0], p1[1], p2[0], p2[1]), dim=1).cpu().numpy()
                    reward, done, _, _ = collector.step(actions, pivots)
                    finished = active & done.bool()
                    for slot in torch.where(finished & (reward == 0))[0].tolist():
                        code = int(collector.codes_buf[slot])
                        turn = int(collector.after_buf[slot]) if hasattr(collector, "after_buf") else None
                        print("EVAL_UNRESOLVED " + json.dumps(dict(
                            seed=seed + start, batch_slot=slot,
                            reason="turn_cap" if code == 1 else "terminal_without_winner",
                            engine_code=code, turn=turn, max_turns=max_turns)), flush=True)
                    scores[finished] = reward[finished]
                    active &= ~finished
                    collector.preview_results.clear()
                    if not active.any():
                        break
                for slot in torch.where(active)[0].tolist():
                    print("EVAL_UNRESOLVED " + json.dumps(dict(
                        seed=seed + start, batch_slot=slot,
                        reason="decision_limit", limit=3 * max_turns)), flush=True)
                result["wins"] += int((scores > 0).sum())
                result["losses"] += int((scores < 0).sum())
                result["draws_or_capped"] += int((scores == 0).sum())
    finally:
        model.train(modes[0])
        opponent.train(modes[1])
    return result


class RolloutThread(threading.Thread):
    def __init__(self, target, args=(), kwargs=None):
        super().__init__()
        self._target = target
        self._args = args
        self._kwargs = kwargs or {}
        self.exception = None

    def run(self):
        try:
            self._target(*self._args, **self._kwargs)
        except Exception as e:
            self.exception = e

    def join_and_raise(self):
        self.join()
        if self.exception is not None:
            raise self.exception


class RolloutBuffer:
    def __init__(self, rollout_len: int, num_envs: int, device: torch.device, has_beliefs: bool = False, has_belief_head: bool = False):
        self.rollout_len = rollout_len
        self.num_envs = num_envs
        self.device = device
        shape = (rollout_len, num_envs)
        alloc = lambda *tail, dtype=torch.float32: torch.empty((*shape, *tail), dtype=dtype, device=device)
        self.belief_targets = ({"item": alloc(2, dtype=torch.long),
                                "ability": alloc(2, dtype=torch.long),
                                "moves": alloc(2, 4, dtype=torch.long),
                                "valid": alloc(2, dtype=torch.bool)} if has_belief_head else {})
        self.cats = alloc(17, 4, dtype=torch.long)
        self.nums = alloc(17, 16)
        self.masks_a = alloc(35, dtype=torch.bool)
        self.masks_b = alloc(35, dtype=torch.bool)
        self.pivot_masks_a = alloc(35, dtype=torch.bool)
        self.pivot_masks_b = alloc(35, dtype=torch.bool)
        self.history_buffers = {
            "history_cat": alloc(12, 3, dtype=torch.long),
            "history_num": alloc(12, 6),
            "history_moves": alloc(12, 4, dtype=torch.long),
            "history_flags": alloc(12, 6),
        }
        if has_beliefs:
            self.history_buffers["belief_features"] = alloc(17, 16)
        self.acts_a = alloc(dtype=torch.long)
        self.acts_b = alloc(dtype=torch.long)
        self.pivot_acts_a = alloc(dtype=torch.long)
        self.pivot_acts_b = alloc(dtype=torch.long)
        self.logps = alloc()
        self.values = alloc()
        self.rewards = alloc()
        self.dones = alloc()
        self.elapsed_turns = alloc()

        self.last_cat = torch.empty((num_envs, 17, 4), dtype=torch.long, device=device)
        self.last_num = torch.empty((num_envs, 17, 16), dtype=torch.float32, device=device)
        self.last_history = {
            "history_cat": torch.empty((num_envs, 12, 3), dtype=torch.long, device=device),
            "history_num": torch.empty((num_envs, 12, 6), dtype=torch.float32, device=device),
            "history_moves": torch.empty((num_envs, 12, 4), dtype=torch.long, device=device),
            "history_flags": torch.empty((num_envs, 12, 6), dtype=torch.float32, device=device),
        }
        if has_beliefs:
            self.last_history["belief_features"] = torch.empty((num_envs, 17, 16), dtype=torch.float32, device=device)

        self.advantages = alloc()
        self.returns = alloc()

        self.preview_results = []
        self.local_games = 0
        self.local_capped = 0
        self.replacement_decisions = 0
        self.mega_declarations = 0
        self.aux_targets = {}


def collect_rollout(
    collector,
    policy,
    opponent,
    buf: RolloutBuffer,
    amp,
    gamma: float,
    gae_lambda: float,
    is_mirror: bool = False,
    stream: torch.cuda.Stream | None = None,
    aux_extended: bool = False,
):
    # CUDA's current device is thread-local. Bind the rollout worker before
    # entering a stream context so cleanup cannot initialize default GPU 0.
    if collector.device.type == "cuda":
        torch.cuda.set_device(collector.device)
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    ctx = torch.cuda.stream(stream) if stream is not None else nullcontext()
    with ctx, torch.no_grad():
        local_games = local_capped = 0
        collector.preview_results = []
        collector.replacement_decisions = collector.mega_declarations = 0

        for t in range(buf.rollout_len):
            collector.reset_done(policy, opponent, amp)
            cat, num, masks, history = collector.encode()
            for key in buf.belief_targets:
                buf.belief_targets[key][t].copy_(torch.from_numpy(collector.belief_targets[key]).to(buf.device))
            with amp():
                ours, theirs = sample_actions_pair(
                    policy, opponent, cat, num, masks, history,
                    is_mirror=is_mirror,
                )
            actions = torch.stack((ours[0], ours[1], theirs[0], theirs[1]), dim=1).cpu().numpy()
            pivot_masks = collector.pivot_masks(actions)
            ours_pivot, theirs_pivot = sample_pivots_pair(ours, theirs, pivot_masks)
            pivots = torch.stack((ours_pivot[0], ours_pivot[1], theirs_pivot[0], theirs_pivot[1]), dim=1).cpu().numpy()
            reward, done, ended, truncated = collector.step(actions, pivots)
            local_games += ended
            local_capped += truncated
            buf.cats[t], buf.nums[t] = cat[0], num[0]
            buf.masks_a[t], buf.masks_b[t] = masks[0, :, 0], ours[4]
            buf.pivot_masks_a[t], buf.pivot_masks_b[t] = pivot_masks[0, :, 0], ours_pivot[3]
            buf.pivot_acts_a[t], buf.pivot_acts_b[t] = ours_pivot[:2]
            buf.acts_a[t], buf.acts_b[t] = ours[:2]
            buf.logps[t] = ours[2] + ours_pivot[2]
            buf.values[t] = ours[3]
            buf.rewards[t], buf.dones[t] = reward, done
            buf.elapsed_turns[t] = torch.from_numpy(collector.elapsed_turns).to(buf.device)
            for key in buf.history_buffers:
                buf.history_buffers[key][t] = history[key][0]

        cat, num, _, history = collector.encode()
        buf.last_cat.copy_(cat[0])
        buf.last_num.copy_(num[0])
        for k in buf.last_history:
            buf.last_history[k].copy_(history[k][0])

        with amp():
            bootstrap = policy(cat[0], num[0], **{k: v[0] for k, v in history.items()})["value"].squeeze(-1).float()
        advantages, returns = generalized_advantages(
            buf.rewards, buf.values, buf.dones, bootstrap, gamma, gae_lambda, buf.elapsed_turns
        )
        buf.advantages.copy_(advantages)
        buf.returns.copy_(returns)

        buf.preview_results = list(collector.preview_results)
        buf.local_games = local_games
        buf.local_capped = local_capped
        buf.replacement_decisions = collector.replacement_decisions
        buf.mega_declarations = collector.mega_declarations

        if aux_extended:
            full_cats = torch.cat([buf.cats, buf.last_cat.unsqueeze(0)], dim=0)
            full_nums = torch.cat([buf.nums, buf.last_num.unsqueeze(0)], dim=0)
            target_weathers = full_cats[1:, :, 16, 0].long().clamp(0, 4)
            target_terrains = full_cats[1:, :, 16, 1].long().clamp(0, 4)
            aux_slots = [0, 1, 4, 5]
            cur_hp = full_nums[:-1, :, aux_slots, 0]
            next_hp = full_nums[1:, :, aux_slots, 0]
            target_hps = next_hp.clamp(0.0, 1.0)
            target_damages = (cur_hp - next_hp).clamp(0.0, 1.0)
            target_survives = full_nums[1:, :, aux_slots, 9].clamp(0.0, 1.0)
            combat_masks = (buf.dones == 0) & (buf.elapsed_turns > 0)
            active_alive_masks = (full_nums[:-1, :, aux_slots, 9] > 0)

            buf.aux_targets = {
                "flat_aux_wea": target_weathers.flatten(0, 1),
                "flat_aux_ter": target_terrains.flatten(0, 1),
                "flat_aux_hp": target_hps.flatten(0, 1),
                "flat_aux_dmg": target_damages.flatten(0, 1),
                "flat_aux_surv": target_survives.flatten(0, 1),
                "flat_combat_mask": combat_masks.flatten(0, 1),
                "flat_active_alive_mask": active_alive_masks.flatten(0, 1),
            }


def run_ppo_update(
    model,
    optimizer,
    scaler,
    buf: RolloutBuffer,
    train_preview: TeamPreviewPool,
    args,
    device: torch.device,
    world: int,
    distributed: bool,
    amp,
    anchor=None,
    ema=None,
):
    temperature = getattr(args, "temperature", 1.0)
    belief_coefficient = getattr(args, "belief_loss_coef", 0.1)
    completed = buf.preview_results
    local_previews = len(completed)
    preview_count = torch.tensor(local_previews, dtype=torch.long, device=device)
    if distributed:
        dist.all_reduce(preview_count)
    global_previews = int(preview_count)

    if completed:
        decisions = [r.decision for r in completed]
        preview_inputs = [torch.from_numpy(x).to(device) for x in train_preview.observations(
            [d.team for d in decisions], [d.opponent for d in decisions],
        )]
        preview_data = preview_inputs + [
            torch.tensor([d.action for d in decisions], device=device, dtype=torch.long),
            torch.tensor([d.logp for d in decisions], device=device),
            torch.tensor([d.value for d in decisions], device=device),
            torch.tensor([r.outcome for r in completed], device=device),
        ]
    else:
        preview_data = [torch.from_numpy(x).to(device) for x in train_preview.observations([0], [0])]
        preview_data += [torch.zeros(1, dtype=torch.long, device=device)]
        preview_data += [torch.zeros(1, device=device) for _ in range(3)]

    count = args.num_envs * args.rollout_len
    flat = [x.flatten(0, 1) for x in (
        buf.cats, buf.nums, buf.masks_a, buf.masks_b, buf.acts_a, buf.acts_b,
        buf.logps, buf.advantages, buf.returns,
        buf.pivot_masks_a, buf.pivot_masks_b, buf.pivot_acts_a, buf.pivot_acts_b
    )]
    flat_history = {k: v.flatten(0, 1) for k, v in buf.history_buffers.items()}
    flat_belief_targets = {k: v.flatten(0, 1) for k, v in buf.belief_targets.items()}
    flat[7] = (flat[7] - flat[7].mean()) / (flat[7].std(unbiased=False) + 1e-8)

    losses = []
    preview_losses = []
    aux_losses = []
    minibatches = math.ceil(count / args.batch_size)
    accum = max(1, args.accum)

    policy = model.module if distributed else model
    has_aux = args.aux_coef > 0 and getattr(policy, "aux_extended", False)
    if has_aux:
        aux_t = buf.aux_targets
        flat_aux_wea = aux_t["flat_aux_wea"]
        flat_aux_ter = aux_t["flat_aux_ter"]
        flat_aux_hp = aux_t["flat_aux_hp"]
        flat_aux_dmg = aux_t["flat_aux_dmg"]
        flat_aux_surv = aux_t["flat_aux_surv"]
        flat_combat_mask = aux_t["flat_combat_mask"]
        flat_active_alive_mask = aux_t["flat_active_alive_mask"]

    flat_anc_logits_a = None
    flat_anc_logits_b = None
    if anchor is not None and args.kl_coef > 0:
        flat_anc_logits_a = torch.empty((count, 35), dtype=torch.float32, device=device)
        flat_anc_logits_b = torch.empty((count, 35), dtype=torch.float32, device=device)
        with torch.no_grad(), amp():
            for i in range(0, count, args.batch_size):
                j = min(i + args.batch_size, count)
                c_chunk = flat[0][i:j]
                n_chunk = flat[1][i:j]
                a_chunk = flat[4][i:j]
                hist_chunk = {k: v[i:j] for k, v in flat_history.items()}
                aout = anchor(c_chunk, n_chunk, action_a=a_chunk, **hist_chunk)
                flat_anc_logits_a[i:j] = aout["policy_a"].float()
                flat_anc_logits_b[i:j] = aout["policy_b"].float()

    for _ in range(args.ppo_epochs):
        preview_batches = torch.tensor_split(torch.randperm(local_previews, device=device), minibatches)
        batch_splits = list(torch.randperm(count, device=device).split(args.batch_size))
        total_batches = len(batch_splits)
        optimizer.zero_grad(set_to_none=True)
        for batch_index, ids in enumerate(batch_splits):
            c, n, ma, mb, a, b, old_lp, adv, target = [x[ids] for x in flat[:9]]
            pma, pmb, pa_action, pb_action = [x[ids] for x in flat[9:]]
            pids = preview_batches[batch_index]
            preview_weight = len(pids) * world * minibatches / max(global_previews, 1)
            if not len(pids):
                pids = torch.zeros(1, dtype=torch.long, device=device)
            pc, pn, pm, pa, plp, pv, ptarget = [x[pids] for x in preview_data]
            with amp():
                output = model(
                    c, n, action_a=a,
                    preview_cat=pc, preview_num=pn, preview_moves=pm,
                    **{k: v[ids] for k, v in flat_history.items()},
                )
                da = torch.distributions.Categorical(logits=masked_action_logits(temperature_logits(output["policy_a"], temperature), ma))
                db = torch.distributions.Categorical(logits=masked_action_logits(temperature_logits(output["policy_b"], temperature), mb))
                dp_a = torch.distributions.Categorical(logits=masked_action_logits(temperature_logits(output["policy_a"], temperature), pma))
                dp_b = torch.distributions.Categorical(logits=masked_action_logits(temperature_logits(output["policy_b"], temperature), pmb))
                ratio = (da.log_prob(a) + db.log_prob(b) + dp_a.log_prob(pa_action) + dp_b.log_prob(pb_action) - old_lp).exp()
                loss_p = -torch.minimum(ratio * adv, ratio.clamp(1 - args.clip_eps, 1 + args.clip_eps) * adv).mean()
                loss_v = F.mse_loss(output["value"].squeeze(-1).float(), target)
                belief_loss = (belief_prediction_loss(output, {k: v[ids] for k, v in flat_belief_targets.items()})
                           if flat_belief_targets else loss_p * 0)
                loss = belief_coefficient * belief_loss + loss_p + 0.5 * loss_v - args.entropy * (da.entropy() + db.entropy() + dp_a.entropy() + dp_b.entropy()).mean()
                preview_loss, preview_p, preview_v, preview_entropy = preview_ppo_loss(
                    output, pa, plp, pv, ptarget, args.clip_eps, args.entropy, temperature=temperature,
                )
                loss = loss + preview_weight * preview_loss
                if has_aux:
                    aux_loss, aux_diag = compute_aux_losses(
                        output,
                        flat_aux_wea[ids],
                        flat_aux_ter[ids],
                        flat_aux_surv[ids],
                        flat_aux_hp[ids],
                        flat_aux_dmg[ids],
                        flat_combat_mask[ids],
                        flat_active_alive_mask[ids],
                    )
                    loss = loss + args.aux_coef * aux_loss
                    aux_losses.append(torch.stack([
                        aux_loss.detach(),
                        aux_diag["aux_wea"],
                        aux_diag["aux_ter"],
                        aux_diag["aux_surv"],
                        aux_diag["aux_hp"],
                        aux_diag["aux_dmg"],
                    ]))
                if flat_anc_logits_a is not None:
                    anc_a = torch.distributions.Categorical(logits=masked_action_logits(temperature_logits(flat_anc_logits_a[ids], temperature), ma))
                    anc_b = torch.distributions.Categorical(logits=masked_action_logits(temperature_logits(flat_anc_logits_b[ids], temperature), mb))
                    kl = (torch.distributions.kl_divergence(da, anc_a).mean()
                          + torch.distributions.kl_divergence(db, anc_b).mean())
                    loss = loss + args.kl_coef * kl
                    if anchor is not None and "preview_plan" in output:
                        with torch.no_grad():
                            anc_p = anchor(preview_cat=pc, preview_num=pn, preview_moves=pm, preview_only=True)
                        dist_anc_p = torch.distributions.Categorical(logits=temperature_logits(anc_p["preview_plan"], temperature))
                        dist_curr_p = torch.distributions.Categorical(logits=temperature_logits(output["preview_plan"], temperature))
                        kl_prev = torch.distributions.kl_divergence(dist_curr_p, dist_anc_p).mean()
                        loss = loss + preview_weight * args.kl_coef * kl_prev
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite PPO loss")
            is_accum_step = ((batch_index + 1) % accum != 0) and ((batch_index + 1) != total_batches)
            scaled_loss = loss / accum
            sync_context = model.no_sync() if (distributed and is_accum_step) else nullcontext()
            with sync_context:
                scaler.scale(scaled_loss).backward()
            if not is_accum_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if ema is not None and scaler.get_scale() >= previous_scale:
                    update_ema(ema, getattr(policy, "_orig_mod", policy).state_dict(), args.ema_decay)
                optimizer.zero_grad(set_to_none=True)
            losses.append(torch.stack((loss_p.detach(), loss_v.detach(), belief_loss.detach())))
            preview_losses.append(torch.stack((preview_p.detach(), preview_v.detach(), preview_entropy.detach())) * preview_weight)

    return losses, preview_losses, aux_losses, global_previews
def copy_weights(dst, src_state_dict):
    target = getattr(dst, "_orig_mod", dst)
    target.load_state_dict(src_state_dict)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-pool", required=True)
    p.add_argument("--holdout-pool", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--timesteps", type=int, default=30_000_000)
    p.add_argument("--num-envs", type=int, default=256)
    p.add_argument("--rollout-len", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--accum", type=int, default=1,
                   help="Gradient accumulation steps per optimizer update (multiplies effective minibatch size by accum)")
    p.add_argument("--ppo-epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--temperature", type=float, default=1.0, help="Positive mixed-action sampling temperature; not a Nash solver")
    p.add_argument("--lr-half-every", type=int, default=0, help="Halve LR every N global environment steps; 0 disables")
    p.add_argument("--lr-min", type=float, default=1e-5)
    p.add_argument("--dynamic-damping", action="store_true",
                   help="Anneal entropy and LR together; skill-check halvings multiply the LR curve")
    p.add_argument("--damping-steps", type=int, default=0,
                   help="Annealing duration when first enabled; 0 uses remaining target steps")
    p.add_argument("--entropy-final", type=float, default=0.002)
    p.add_argument("--skill-check-seconds", type=float, default=0,
                   help="Active-run seconds between checks against the start-of-window network; 0 disables")
    p.add_argument("--skill-check-games", type=int, default=2000)
    p.add_argument("--skill-check-batch-size", type=int, default=256)
    p.add_argument("--skill-check-threshold", type=float, default=0.51)
    p.add_argument("--learned-belief-head", action="store_true", help="Predict opponent active items, abilities and moves from public inputs")
    p.add_argument("--belief-loss-coef", type=float, default=0.1)
    p.add_argument("--entropy", type=float, default=0.02)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--save-interval", type=int, default=1_000_000)
    p.add_argument("--eval-interval", type=int, default=1_000_000)
    p.add_argument("--eval-games", type=int, default=128)
    p.add_argument("--max-turns", type=int, default=60)
    p.add_argument("--target-games", type=int, default=0,
                   help="Stop once this many completed games are reached (0 = governed only by --timesteps)")
    p.add_argument("--seed", type=int, default=20261017)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--belief-priors", help="Frozen public MunchStats snapshot; enables Transformer19")
    p.add_argument("--belief-particles", type=int, default=16)
    p.add_argument("--init-policy", help="Corrected model18 PPO checkpoint, accepted only with public beliefs")
    p.add_argument("--resume-from", help="Saved Model 18/19 checkpoint to resume training directly from")
    p.add_argument("--warmstart-from", help="Model checkpoint to initialize weights from without restoring step counter/optimizer")
    p.add_argument("--specialist-team", default=None,
                   help="Path to team (.txt or .json) for specialist training: learner always plays this team vs the entire train-pool")
    p.add_argument("--batched", action="store_true",
                   help="Use the Rust BattleBatch collector (encode18_batch/step18); scales to thousands of envs")
    # Tier 1 — exploiter league (AlphaStar/Stratego-style). Exploiters are produced by a
    # second invocation with --exploit-target pinned to a main checkpoint; they drop their
    # checkpoints into --exploiter-dir, which the main run mixes into its opponent pool.
    p.add_argument("--exploiter-dir", default=None,
                   help="Directory of exploiter checkpoints to mix into the opponent pool (rglob *.pt)")
    p.add_argument("--exploiter-prob", type=float, default=0.3,
                   help="Probability the opponent refresh draws from --exploiter-dir when it is non-empty")
    p.add_argument("--exploit-target", default=None,
                   help="Train THIS run as an exploiter: pin the opponent to this frozen checkpoint "
                        "(no mirror/past-self), so wins against it are the only reward")
    # Tier 2 — plateau LR halving + EMA weights.
    p.add_argument("--ema-decay", type=float, default=0.0,
                   help="If >0, keep an EMA copy of the weights and evaluate/save it alongside (e.g. 0.999)")
    p.add_argument("--plateau-interval", type=int, default=0,
                   help="Every N steps, play current vs the previous-window reference; halve LR if "
                        "win rate < --plateau-threshold (0 disables)")
    p.add_argument("--plateau-threshold", type=float, default=0.51)
    p.add_argument("--aux-coef", type=float, default=0.25,
                   help="Weight for dense auxiliary supervision loss (weather, terrain, survival, hp, damage). 0 disables.")
    p.add_argument("--plateau-min-steps", type=int, default=10_000_000,
                   help="Minimum steps before plateau LR halving can trigger (protects early exploration / escape phase)")
    p.add_argument("--plateau-min-lr", type=float, default=1e-5,
                   help="Floor learning rate for plateau LR halving")
    # Wolfe's anti-collapse leash: a KL penalty that first pulls the policy toward the imitation
    # warmstart, then (optionally) toward its own slow EMA. rl18 had no KL term, so the policy
    # drifted into the weak over-confident equilibrium we measured.
    p.add_argument("--kl-anchor", default=None,
                   help="Checkpoint to anchor toward via a KL penalty (use the imitation warmstart)")
    p.add_argument("--kl-coef", type=float, default=0.0, help="KL penalty strength (0 disables)")
    p.add_argument("--kl-ema-after", type=int, default=0,
                   help="After this many steps, re-anchor the KL target to the latest EMA checkpoint "
                        "instead of the fixed imitation net (0 keeps the imitation anchor; needs --ema-decay)")
    p.add_argument("--double-buffer", action=argparse.BooleanOptionalAction, default=True,
                   help="Overlap rollout collection in a background thread with GPU PPO update (Phase 4)")
    p.add_argument("--torch-compile", action=argparse.BooleanOptionalAction, default=False,
                   help="Wrap policy model in torch.compile for high-throughput CUDA execution")
    p.add_argument("--compile-mode", default="default",
                   choices=["default", "reduce-overhead", "max-autotune"],
                   help="torch.compile optimization mode")
    args = p.parse_args()
    temperature_logits(torch.zeros(1), args.temperature)
    if not 0 <= args.ema_decay < 1:
        p.error("--ema-decay must be in [0, 1)")
    scheduled_lr(args.lr, 0, args.lr_half_every, args.lr_min)
    validate_skill_check(args.skill_check_seconds, args.skill_check_games,
                         args.skill_check_batch_size, args.skill_check_threshold)
    if args.damping_steps < 0 or (args.dynamic_damping and not 0 <= args.entropy_final <= args.entropy):
        p.error("invalid damping duration or entropy endpoints")
    if args.dynamic_damping and (args.lr_half_every or args.plateau_interval):
        p.error("dynamic damping replaces fixed-step and legacy plateau LR schedules")
    if args.skill_check_seconds and (args.lr_half_every or args.plateau_interval):
        p.error("time-window skill checks replace fixed-step and legacy plateau halving")
    if args.lr_half_every and args.plateau_interval:
        p.error("Choose scheduled LR halving or plateau halving, not both")
    if args.learned_belief_head and not (args.belief_priors and args.batched):
        p.error("--learned-belief-head requires --belief-priors and --batched")
    if args.belief_loss_coef < 0:
        p.error("--belief-loss-coef must be nonnegative")
    if args.init_policy and not args.belief_priors:
        p.error("--init-policy requires --belief-priors")
    if min(args.num_envs, args.rollout_len, args.batch_size, args.ppo_epochs, args.timesteps, args.save_interval, args.eval_interval, args.max_turns) < 1 or args.eval_games < 0:
        p.error("Counts/intervals must be positive; eval-games may be zero")
    distributed = int(os.environ.get("WORLD_SIZE", 1)) > 1
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    device = torch.device(f"cuda:{os.environ['LOCAL_RANK']}" if distributed else args.device)
    if device.type == "cuda":
        if device.index is None:
            device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
        torch.cuda.set_device(device)
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if distributed:
        dist.init_process_group("nccl")
    outdir = Path(args.out_dir)
    if not args.resume_from and not args.warmstart_from and any(outdir.glob("*.pt")):
        raise ValueError("Fresh initialization requires an output directory without checkpoints (use --resume-from to resume)")
    outdir.mkdir(parents=True, exist_ok=True)
    for method in ("legal_action_mask_18", "actions_18_to_choices", "public_history_18"):
        if not hasattr(vgc_engine.Battle, method):
            raise RuntimeError(f"Rebuild the native extension: missing {method}")
    torch.manual_seed(args.seed)
    belief_bank = None
    architecture = Transformer18
    parent = None
    if args.belief_priors:
        from ai_vgc.nn.belief19 import PublicBeliefBank
        from ai_vgc.nn.model19 import Transformer19, Transformer19Belief, initialize_from_model18
        belief_bank = PublicBeliefBank(args.belief_priors, particles=args.belief_particles)
        architecture = Transformer19Belief if args.learned_belief_head else Transformer19
    model = architecture(dropout=0.0).to(device)
    resume_ckpt = None
    if args.resume_from:
        resume_ckpt = torch.load(args.resume_from, map_location=device, weights_only=False)
        prior_contract = resume_ckpt.get("config", {}).get("action_contract")
        if prior_contract != ACTION_CONTRACT:
            raise ValueError(
                f"Cannot resume optimizer from action contract {prior_contract!r}; "
                f"expected {ACTION_CONTRACT!r}. Use --warmstart-from to retain weights "
                "and start a new rollout/optimizer history."
            )
        model.load_state_dict(resume_ckpt["model_state_dict"], strict=False)
    elif args.warmstart_from:
        w_ckpt = torch.load(args.warmstart_from, map_location=device, weights_only=False)
        w_state = w_ckpt.get("model_state_dict", w_ckpt.get("model", w_ckpt))
        incompatible = model.load_state_dict(
            {k.removeprefix("module."): v for k, v in w_state.items()}, strict=False)
        allowed_new = ("belief_projection.", "opponent_belief.")
        unsupported = [key for key in incompatible.missing_keys if not key.startswith(allowed_new)]
        if incompatible.unexpected_keys or unsupported:
            raise ValueError(f"Incompatible warmstart: missing={unsupported}, unexpected={incompatible.unexpected_keys}")
        if rank == 0 and incompatible.missing_keys:
            print(f"New belief parameters initialized: {incompatible.missing_keys}", flush=True)
        if rank == 0:
            print(f"Warmstarted model weights from {args.warmstart_from}", flush=True)
    elif args.init_policy:
        parent = initialize_from_model18(model, args.init_policy)
    if args.torch_compile:
        model = torch.compile(model, mode=args.compile_mode)
    if distributed:
        model = DDP(model, device_ids=[device.index], find_unused_parameters=True)
    policy = model.module if distributed else model
    raw_policy = getattr(policy, "_orig_mod", policy)
    initial_preview_head = [p.detach().clone() for p in raw_policy.preview_head.parameters()]
    opponent = architecture(dropout=0.0).to(device)
    copy_weights(opponent, raw_policy.state_dict())
    opponent.eval().requires_grad_(False)
    # A fixed random network is a consistent initial evaluation opponent.
    baseline = architecture(dropout=0.0).to(device) if rank == 0 and args.eval_games else None
    if baseline is not None:
        baseline_path = outdir / "eval-baseline.pt"
        if baseline_path.exists():
            copy_weights(baseline, torch.load(baseline_path, map_location=device, weights_only=False))
        elif resume_ckpt and resume_ckpt.get("config", {}).get("warmstart_from"):
            # Preserve the original opponent when resuming a pre-persistence run.
            initial = torch.load(resume_ckpt["config"]["warmstart_from"], map_location=device, weights_only=False)
            state = initial.get("model_state_dict", initial.get("model", initial))
            missing = baseline.load_state_dict(
                {k.removeprefix("module."): v for k, v in state.items()}, strict=False)
            if missing.unexpected_keys or any(not k.startswith(("belief_projection.", "opponent_belief.")) for k in missing.missing_keys):
                raise ValueError(f"Incompatible initial evaluation checkpoint: {missing}")
        else:
            copy_weights(baseline, raw_policy.state_dict())
        if not baseline_path.exists():
            baseline_temp = baseline_path.with_suffix(".tmp")
            torch.save(baseline.state_dict(), baseline_temp)
            baseline_temp.replace(baseline_path)
        baseline.eval().requires_grad_(False)
    # Exploiter mode: pin the opponent to the frozen target for the whole run.
    if args.exploit_target:
        copy_weights(opponent, torch.load(args.exploit_target, map_location=device, weights_only=False)["model_state_dict"])
        opponent.eval().requires_grad_(False)
    # EMA shadow weights (rank 0): a slow-moving average used for eval and a *_ema.pt checkpoint.
    ema = None
    if rank == 0 and args.ema_decay > 0:
        saved_ema = resume_ckpt.get("ema_state_dict") if resume_ckpt else None
        if saved_ema is None and args.resume_from:
            resume_path = Path(args.resume_from)
            companion = resume_path.with_name(resume_path.stem + "_ema.pt")
            if companion.exists():
                prior_ema = torch.load(companion, map_location=device, weights_only=False)
                if prior_ema.get("total_steps") != resume_ckpt.get("total_steps"):
                    raise ValueError("EMA companion belongs to a different training step")
                saved_ema = prior_ema["model_state_dict"]
        ema = initialize_ema(raw_policy.state_dict(), saved_ema)
        print(f"EMA decay={args.ema_decay}; update unit=optimizer step; "
              f"{'restored saved average' if saved_ema is not None else 'initialized from current weights'}", flush=True)
    ema_net = architecture(dropout=0.0).to(device) if ema is not None else None
    if ema_net is not None:
        ema_net.eval().requires_grad_(False)
    # Plateau reference (rank 0): the policy snapshot from the start of the current LR window.
    reference = architecture(dropout=0.0).to(device) if (rank == 0 and args.plateau_interval and args.eval_games) else None
    if reference is not None:
        copy_weights(reference, raw_policy.state_dict())
        reference.eval().requires_grad_(False)
    # KL anchor (all ranks): the frozen target the policy is penalized for drifting from.
    anchor = None
    if args.kl_anchor and args.kl_coef > 0:
        anchor = architecture(dropout=0.0).to(device)
        anchor.load_state_dict(torch.load(args.kl_anchor, map_location=device, weights_only=False)["model_state_dict"], strict=False)
        if (resume_ckpt is not None and args.kl_ema_after > 0
                and resume_ckpt.get("total_steps", 0) >= args.kl_ema_after):
            anchor_ema = resume_ckpt.get("ema_state_dict")
            if anchor_ema is None:
                resume_path = Path(args.resume_from)
                companion = resume_path.with_name(resume_path.stem + "_ema.pt")
                saved_anchor = torch.load(companion, map_location=device, weights_only=False)
                if saved_anchor.get("total_steps") != resume_ckpt.get("total_steps"):
                    raise ValueError("KL EMA companion does not match resumed checkpoint")
                anchor_ema = saved_anchor["model_state_dict"]
            anchor.load_state_dict(anchor_ema, strict=True)
            if rank == 0:
                print(f"KL anchor restored from EMA @ {resume_ckpt['total_steps']}; coefficient={args.kl_coef}", flush=True)
        anchor.eval().requires_grad_(False)
        if args.torch_compile:
            anchor = torch.compile(anchor, mode=args.compile_mode)
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    def pool(path):
        teams = load_pool_teams(path)
        # Until the native engine tracks public Illusion disguises, omit
        # those teams instead of exposing a disguised opponent's true species.
        eligible = [t for t in teams if len(t) == 6 and not any(str(m.get("ability", "")).lower() == "illusion" for m in t)]
        if not eligible:
            raise ValueError(f"No eligible teams in {path}")
        return eligible, len(teams) - len(eligible)
    train, omitted_train = pool(args.train_pool)
    holdout, omitted_holdout = pool(args.holdout_pool)

    specialist_idx = None
    if args.specialist_team:
        p_spec = Path(args.specialist_team)
        if p_spec.suffix == ".txt":
            from poke_env.teambuilder import Teambuilder
            parsed = Teambuilder.parse_showdown_team(p_spec.read_text("utf-8"))
            spec_team = [{
                "species": m.species or m.nickname,
                "ability": m.ability or "",
                "item": m.item or "",
                "nature": m.nature or "Serious",
                "moves": m.moves,
                "evs": m.evs or {},
                "ivs": m.ivs or {},
            } for m in parsed]
        else:
            data = json.loads(p_spec.read_text("utf-8"))
            spec_team = data.get("pokemon", data.get("team", data)) if isinstance(data, dict) else data

        train = [spec_team] + train
        specialist_idx = 0
        if rank == 0:
            print(f"★ SPECIALIST MODE ENABLED: Learner ALWAYS plays {p_spec.name} vs all {len(train)} teams in the pool!", flush=True)

    train_preview = TeamPreviewPool(train)
    eval_teams = TeamPreviewPool(holdout) if rank == 0 and (args.eval_games or args.skill_check_seconds) else None
    if args.batched:
        collector = BatchCollector(train_preview, args.num_envs, device, args.max_turns, belief_bank,
                                   fmt="gen9championsvgc2026regmc", tera_allowed=False, seed=args.seed + rank,
                                   specialist_team=specialist_idx, track_belief_targets=args.learned_belief_head)
    else:
        collector = Collector(train_preview, args.num_envs, device, args.max_turns, belief_bank)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    if resume_ckpt and "optimizer_state_dict" in resume_ckpt:
        optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    amp = (lambda: torch.autocast("cuda", dtype=dtype)) if device.type == "cuda" else nullcontext
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and dtype == torch.float16))
    config = {**vars(args), "initialization": args.warmstart_from or args.resume_from or args.init_policy or "random", "architecture": architecture.__name__, "parameters": sum(p.numel() for p in policy.parameters()), "input_contract": "conservative-cts-v1 + public-six-preview-v1", "legacy_checkpoint_or_dataset": None, "train_teams": len(train), "holdout_teams": len(holdout), "omitted_ineligible_teams": [omitted_train, omitted_holdout], "preview": "PPO-trained bring-four and ordered leads: 180 plans, terminal outcome credit", "preview_information": "own six complete sets and stats; opponent six species only", "replacements": "sampled legal replacements, explicit decision phases", "battle_seeds": "independent 64-bit seed per reset", "torch": torch.__version__, "engine_module": vgc_engine.__file__, "world_size": world}
    config.update(
        action_contract=ACTION_CONTRACT,
        ema_update_unit="optimizer_step",
        input_contract="public-history-18-v1; 35 explicit normal/Mega actions",
        replacements="neural policy decisions recorded in PPO; zero elapsed battle turns",
        mega="explicit policy choice of normal/Mega move and holder; no automatic declaration",
        history="persistent public species, item/ability announcements and move uses; unrevealed sets masked",
        timesteps_meaning="policy decision phases, including forced replacements",
    )
    if belief_bank is not None:
        config.update(input_contract="public-history-18-v1 + public-beliefs-19-v1",
                      architecture=architecture.__name__,
                      beliefs=belief_bank.metadata,
                      belief_features="sampled-set damage ranges and approximate noncritical KO probability; same-priority move-order probabilities; posterior entropy",
                      belief_limits="public usage marginals, not joint sets; approximate conditional damage; no opponent private sets/stats",
                      initialization=args.warmstart_from or args.resume_from or args.init_policy or "random")
        if parent is not None:
            config["parent"] = {"checkpoint": str(Path(args.init_policy).resolve()),
                                "sha256": hashlib.sha256(Path(args.init_policy).read_bytes()).hexdigest(),
                                "total_steps": parent["total_steps"], "total_games": parent["total_games"],
                                "input_contract": parent["config"]["input_contract"]}
    config["preview_initial_head_sha256"] = hashlib.sha256(b"".join(p.cpu().numpy().tobytes() for p in initial_preview_head)).hexdigest()
    if rank == 0:
        (outdir / "training_config.json").write_text(json.dumps(config, indent=2) + "\n")
        print(json.dumps(config), flush=True)
    buf_curr = RolloutBuffer(args.rollout_len, args.num_envs, device, has_beliefs=(belief_bank is not None), has_belief_head=args.learned_belief_head)
    buf_next = (
        RolloutBuffer(args.rollout_len, args.num_envs, device, has_beliefs=(belief_bank is not None), has_belief_head=args.learned_belief_head)
        if args.double_buffer else None
    )
    inference_policy = architecture(dropout=0.0).to(device) if args.double_buffer else None
    if inference_policy is not None:
        copy_weights(inference_policy, raw_policy.state_dict())
        inference_policy.eval().requires_grad_(False)
    rollout_stream = torch.cuda.Stream(device=device) if (args.double_buffer and device.type == "cuda") else None

    for network in (raw_policy, opponent, baseline, anchor, ema_net, reference, inference_policy):
        if network is not None:
            network.sampling_temperature = args.temperature

    if resume_ckpt:
        steps = resume_ckpt.get("total_steps", 0)
        games = resume_ckpt.get("total_games", 0)
        capped = resume_ckpt.get("capped_games", 0)
        update = resume_ckpt.get("update", 0)
        previews = resume_ckpt.get("preview_examples", 0)
    else:
        steps = games = capped = update = previews = 0
    saved_control = (resume_ckpt or {}).get("training_control") or {}
    saved_skill = saved_control.get("skill")
    skill = SkillCheckState(**saved_skill) if saved_skill else SkillCheckState(reference_steps=steps)
    skill_base_lr = saved_control.get("base_lr", optimizer.param_groups[0]["lr"] / skill.lr_multiplier)
    damping = None
    if args.dynamic_damping:
        saved_damping = saved_control.get("damping")
        damping = DampingSchedule(**saved_damping) if saved_damping else DampingSchedule(
            start_steps=steps, duration_steps=args.damping_steps or max(1, args.timesteps - steps),
            initial_lr=skill_base_lr, final_lr=args.lr_min,
            initial_entropy=args.entropy, final_entropy=args.entropy_final,
        )
    skill_reference = None
    if rank == 0 and args.skill_check_seconds:
        skill_reference = architecture(dropout=0.0).to(device)
        saved_reference = (resume_ckpt or {}).get("skill_reference_state_dict")
        if saved_skill and saved_reference is None:
            raise ValueError("Missing skill-check reference; refusing to reset the comparison window")
        copy_weights(skill_reference, saved_reference if saved_reference is not None else raw_policy.state_dict())
        skill_reference.eval().requires_grad_(False)
        skill_reference.sampling_temperature = args.temperature
        print(f"Skill checks: {args.skill_check_games} games every {args.skill_check_seconds:g}s; "
              f"reference @ {skill.reference_steps}; elapsed={skill.elapsed_seconds:.1f}s; "
              f"halve below {args.skill_check_threshold:.1%}", flush=True)
    skill_window_started = time.monotonic()

    def training_control_state():
        return dict(base_lr=skill_base_lr, damping=damping.state_dict() if damping else None,
                    skill=skill.state_dict(time.monotonic() - skill_window_started)
                    if args.skill_check_seconds else None)
    league = []
    opponent_name = f"exploit-target:{Path(args.exploit_target).stem}" if args.exploit_target else "mirror"

    # Warmup iteration when double-buffering: synchronously collect buf_curr
    if args.double_buffer:
        collect_rollout(
            collector,
            inference_policy,
            opponent,
            buf_curr,
            amp,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            is_mirror=(opponent_name == "mirror"),
            stream=rollout_stream,
            aux_extended=getattr(policy, "aux_extended", False) and args.aux_coef > 0,
        )
        if rollout_stream is not None:
            rollout_stream.synchronize()

    while (games < args.target_games if args.target_games else steps < args.timesteps):
        started = time.monotonic()
        update += 1
        if damping is not None or args.skill_check_seconds:
            if damping is not None:
                rate, args.entropy = damping.values(steps, skill.lr_multiplier)
            else:
                rate = max(args.lr_min, skill_base_lr * skill.lr_multiplier)
            for group in optimizer.param_groups:
                group["lr"] = rate
        elif args.lr_half_every:
            rate = scheduled_lr(args.lr, steps, args.lr_half_every, args.lr_min)
            if optimizer.param_groups[0]["lr"] != rate and rank == 0:
                print(f"Scheduled LR at {steps} steps: {rate:.3e}", flush=True)
            for group in optimizer.param_groups:
                group["lr"] = rate

        if args.double_buffer:
            rollout_thread = RolloutThread(
                target=collect_rollout,
                args=(
                    collector,
                    inference_policy,
                    opponent,
                    buf_next,
                    amp,
                    args.gamma,
                    args.gae_lambda,
                    (opponent_name == "mirror"),
                    rollout_stream,
                    getattr(policy, "aux_extended", False) and args.aux_coef > 0,
                ),
            )
            rollout_thread.start()
        else:
            model.eval()
            collect_rollout(
                collector,
                policy,
                opponent,
                buf_curr,
                amp,
                args.gamma,
                args.gae_lambda,
                is_mirror=(opponent_name == "mirror"),
                stream=None,
                aux_extended=getattr(policy, "aux_extended", False) and args.aux_coef > 0,
            )

        model.train()
        losses, preview_losses, aux_losses, global_previews = run_ppo_update(
            model,
            optimizer,
            scaler,
            buf_curr,
            train_preview,
            args,
            device,
            world,
            distributed,
            amp,
            anchor=anchor,
            ema=ema,
        )

        if args.double_buffer:
            rollout_thread.join_and_raise()
            if rollout_stream is not None:
                rollout_stream.synchronize()

        counts = torch.tensor([
            buf_curr.local_games,
            buf_curr.local_capped,
            buf_curr.replacement_decisions,
            buf_curr.mega_declarations,
        ], device=device, dtype=torch.long)
        elapsed = torch.tensor(time.monotonic() - started, device=device)
        if distributed:
            dist.all_reduce(counts)
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)

        count = args.num_envs * args.rollout_len
        increment = count * world
        steps += increment
        games += int(counts[0])
        capped += int(counts[1])
        previews += global_previews
        duration = float(elapsed)
        if rank == 0:
            lp, lv, belief_ce = torch.stack(losses).mean(dim=0).tolist()
            pp, pval, pent = torch.stack(preview_losses).mean(dim=0).tolist()
            aux_str = ""
            if aux_losses:
                l_aux, l_wea, l_ter, l_surv, l_hp, l_dmg = torch.stack(aux_losses).mean(dim=0).tolist()
                aux_str = f" | Aux: {l_aux:.4f} (W:{l_wea:.3f} T:{l_ter:.3f} S:{l_surv:.3f} HP:{l_hp:.3f} D:{l_dmg:.3f})"
            head_delta = sum((p.detach() - initial).square().sum() for p, initial in zip(policy.preview_head.parameters(), initial_preview_head)).sqrt().item()
            print(f"Update {update:4d} | Steps: {steps}/{args.timesteps} ({increment/duration:.0f} st/s) | Games: {games} ({int(counts[0])/duration:.1f} g/s, {int(counts[0])/duration/world:.1f} g/gpu) | Capped: {capped} | GPUs: {world} | Opp: {opponent_name} | Pol: {lp:.4f} VF: {lv:.4f} BeliefCE: {belief_ce:.4f}{aux_str} | Preview: {global_previews} ({previews} total) PPol: {pp:.4f} PVF: {pval:.4f} PEnt: {pent:.3f} PHeadDelta: {head_delta:.6f} | Replacements: {int(counts[2])} Mega: {int(counts[3])}", flush=True)
        if rank == 0 and belief_bank is not None:
            if hasattr(collector, "belief_evidence"):
                order_ev, damage_ev = collector.belief_evidence()
            else:
                evidence = [json.loads(t.diagnostics()) for pair in collector.beliefs for t in pair if t is not None]
                order_ev = sum(x["order_evidence"] for x in evidence)
                damage_ev = sum(x["damage_evidence"] for x in evidence)
            print(f"Belief evidence (active games): order={order_ev} damage={damage_ev}; projection_norm={policy.belief_projection.weight.detach().norm().item():.6f}", flush=True)
        skill_ran = False
        if args.skill_check_seconds:
            due = torch.tensor(int(rank == 0 and skill.due(
                args.skill_check_seconds, time.monotonic() - skill_window_started)), device=device)
            if distributed:
                dist.broadcast(due, src=0)
            if due.item():
                skill_ran = True
                sync_state = torch.zeros(4, dtype=torch.float64, device=device)
                if rank == 0:
                    reference_steps = skill.reference_steps
                    window_seconds = skill.elapsed_seconds + time.monotonic() - skill_window_started
                    try:
                        result = evaluate_skill(raw_policy, skill_reference, eval_teams,
                                                args.skill_check_games, args.skill_check_batch_size,
                                                device, amp, args.max_turns, belief_bank,
                                                seed=args.seed + 424242)
                        score, halve = skill.finish(result, expected_games=args.skill_check_games,
                                                    threshold=args.skill_check_threshold, steps=steps)
                        copy_weights(skill_reference, raw_policy.state_dict())
                        sync_state[:] = torch.tensor([1, skill.lr_multiplier, skill.reference_steps,
                                                      skill.checks], dtype=torch.float64, device=device)
                        new_lr = (damping.values(steps, skill.lr_multiplier)[0] if damping else
                                  max(args.lr_min, skill_base_lr * skill.lr_multiplier))
                        print("Skill check " + json.dumps(dict(
                            steps=steps, reference_steps=reference_steps, window_seconds=window_seconds,
                            **result, win_rate=score, threshold=args.skill_check_threshold,
                            halving_triggered=halve, previous_lr=optimizer.param_groups[0]["lr"],
                            next_lr=new_lr, lr_floor=args.lr_min)), flush=True)
                    except Exception as error:
                        print(f"Skill check failed without changing LR: {error}", flush=True)
                if distributed:
                    dist.broadcast(sync_state, src=0)
                if not sync_state[0].item():
                    raise RuntimeError("Skill check failed; inspect rank-zero log")
                skill.lr_multiplier = float(sync_state[1])
                skill.reference_steps, skill.checks = int(sync_state[2]), int(sync_state[3])
                skill.elapsed_seconds = 0.0
                skill_window_started = time.monotonic()
                new_lr = (damping.values(steps, skill.lr_multiplier)[0] if damping else
                          max(args.lr_min, skill_base_lr * skill.lr_multiplier))
                for group in optimizer.param_groups:
                    group["lr"] = new_lr
        reached_target = args.target_games != 0 and games >= args.target_games
        save = skill_ran or update == 1 or steps % args.save_interval < increment or (not args.target_games and steps >= args.timesteps) or reached_target
        if save:
            checkpoint = outdir / f"model{19 if belief_bank else 18}_sp_{steps//1000}k.pt"
            if rank == 0:
                temp = checkpoint.with_suffix(".pt.tmp")
                torch.save({"model_state_dict": raw_policy.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "ema_state_dict": ema,
                            "training_control": training_control_state(),
                            "skill_reference_state_dict": skill_reference.state_dict() if skill_reference is not None else None,
                            "total_steps": steps, "total_games": games, "capped_games": capped, "preview_examples": previews,
                            "model_type": architecture.__name__, "update": update,
                            "config": {**config, "ema_update_unit": "optimizer_step"}}, temp)
                temp.replace(checkpoint)
                if damping is not None or args.skill_check_seconds:
                    print(f"Training control @ {steps}: lr={optimizer.param_groups[0]['lr']:.3e}; "
                          f"entropy_coef={args.entropy:.6f}; skill_checks={skill.checks}", flush=True)
                if ema is not None:
                    ema_ckpt = checkpoint.with_name(checkpoint.stem + "_ema.pt")
                    ema_temp = ema_ckpt.with_suffix(".pt.tmp")
                    torch.save({"model_state_dict": {k: v.to(w.dtype) for (k, v), w in zip(ema.items(), raw_policy.state_dict().values())}, "total_steps": steps, "total_games": games, "model_type": architecture.__name__, "update": update, "config": {**config, "ema_decay": args.ema_decay}}, ema_temp)
                    ema_temp.replace(ema_ckpt)
            if distributed:
                dist.barrier()
            if anchor is not None and args.kl_ema_after and steps >= args.kl_ema_after:
                ema_ckpt = checkpoint.with_name(checkpoint.stem + "_ema.pt")
                if ema_ckpt.exists():  # all ranks read the shared file written above
                    copy_weights(anchor, torch.load(ema_ckpt, map_location=device, weights_only=False)["model_state_dict"])
            league.append(checkpoint)
        if args.eval_games and (update == 1 or steps % args.eval_interval < increment):
            if rank == 0:
                policy.eval()
                result = evaluate(policy, baseline, eval_teams, args.eval_games, device, amp, args.max_turns, belief_bank)
                print(f"Holdout vs frozen initial network @ {steps}: {json.dumps(result)}", flush=True)
                if ema_net is not None:
                    copy_weights(ema_net, {k: v.to(w.dtype) for (k, v), w in zip(ema.items(), raw_policy.state_dict().values())})
                    ema_result = evaluate(ema_net, baseline, eval_teams, args.eval_games, device, amp, args.max_turns, belief_bank)
                    print(f"EMA vs frozen initial network @ {steps}: {json.dumps(ema_result)}", flush=True)
            if distributed:
                dist.barrier()
        # Tier 2 — plateau LR halving: play current vs the previous-window reference; if it
        # cannot clear --plateau-threshold, the run has stalled, so halve the learning rate.
        if args.plateau_interval and args.eval_games and update > 1 and steps >= args.plateau_min_steps and (steps % args.plateau_interval < increment):
            halve = torch.zeros(1, device=device)
            if reference is not None:
                policy.eval()
                res = evaluate(policy, reference, eval_teams, args.eval_games, device, amp, args.max_turns, belief_bank)
                wr = res["wins"] / max(res["games"], 1)
                print(f"Plateau check @ {steps}: current vs reference win rate {wr:.3f} {json.dumps(res)}", flush=True)
                if wr < args.plateau_threshold:
                    halve[0] = 1.0
                copy_weights(reference, raw_policy.state_dict())  # start a new window from here
            if distributed:
                dist.broadcast(halve, src=0)
            if halve.item() > 0.5:
                for g in optimizer.param_groups:
                    g["lr"] = max(g["lr"] / 2, args.plateau_min_lr)
                if rank == 0:
                    print(f"Plateau: halved LR -> {optimizer.param_groups[0]['lr']:.2e}", flush=True)

        # Refresh inference policy and opponent for the next iteration:
        if args.double_buffer:
            copy_weights(inference_policy, raw_policy.state_dict())

        if args.exploit_target:
            opponent_name = f"exploit-target:{Path(args.exploit_target).stem}"  # pinned; never refreshed
        elif update % 5 == 0:
            exploiters = sorted(Path(args.exploiter_dir).rglob("*.pt")) if args.exploiter_dir else []
            r = random.random()
            offset = args.exploiter_prob if exploiters else 0.0  # only shift past-self when exploiters exist
            if exploiters and r < args.exploiter_prob:
                selected = random.choice(exploiters)
                copy_weights(opponent, torch.load(selected, map_location=device, weights_only=False)["model_state_dict"])
                opponent_name = f"exploiter:{selected.stem}"
            elif league and r < offset + 0.3:
                selected = random.choice(league)
                copy_weights(opponent, torch.load(selected, map_location=device, weights_only=False)["model_state_dict"])
                opponent_name = selected.name
            else:
                copy_weights(opponent, raw_policy.state_dict())
                opponent_name = "mirror"

        if args.double_buffer:
            buf_curr, buf_next = buf_next, buf_curr
    if rank == 0:
        print(f"Finished fresh model18 training: {steps} steps, {games} games, {capped} capped.", flush=True)
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
