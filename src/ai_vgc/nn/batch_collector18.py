"""Batched model18/19 rollouts with contingent pivot decisions.

``BattleBatch`` handles observation encoding and stepping. A selected pivot
move carries an additional policy-selected bench choice. This is a declared
turn-start contingency: the replacement executes only if the move actually
causes a self-switch. A fresh mid-turn observation is unavailable when the
choice is made, so policy-position Showdown replay remains the accuracy check.
"""
from __future__ import annotations

import random

import numpy as np
import torch
import vgc_engine

from ai_vgc.nn.preview18 import PreviewDecision, PreviewResult, PREVIEW_PLANS
from ai_vgc.nn.history18 import known_history, apply_known_active_sets
from ai_vgc.nn.ppo18 import mask_closed_observation
from ai_vgc.nn.rl18_extensions import hidden_belief_targets, policy_temperature, temperature_logits

# Mega actions start here in the model18 action space (mirrors actions18.rs).
MEGA_START = 19


class BatchCollector:
    def __init__(self, pool, count, device, max_turns, belief_bank=None, *,
                 fmt="gen9championsvgc2026regmc", tera_allowed=False, seed=0,
                 specialist_team: int | None = None, track_belief_targets=False):
        self.pool = pool
        self.belief_bank = belief_bank
        self.specialist_team = specialist_team
        self.track_belief_targets = track_belief_targets
        self.belief_targets = None
        self.device, self.max_turns, self.count = device, max_turns, count
        self.fmt, self.tera_allowed = fmt, tera_allowed
        self._rng = random.Random(seed)
        self.preview_results = []
        self.preview_decisions = [None] * count
        self.full_teams = np.zeros((count, 2), dtype=np.int64)
        self.native_picks = np.zeros((count, 2, 4), dtype=np.int64)
        self.learner_sides = np.zeros(count, dtype=np.int64)
        self.elapsed_turns = np.zeros(count, dtype=np.float32)
        self.episode_turns = np.zeros(count, dtype=np.float32)
        self.done = np.ones(count, dtype=bool)
        self.replacement_decisions = self.mega_declarations = 0
        # Placeholder valid teams; the first reset_done replaces every slot.
        placeholder = self.pool.selected_json(0, 0)
        self.battles = vgc_engine.BattleBatch.from_teams(
            [placeholder] * count, [placeholder] * count, format=fmt,
            seeds=[self._rng.randrange(2**63) for _ in range(count)],
            tera_allowed=tera_allowed, decision_phases=True, stat_points=True,
        )
        if belief_bank is not None:
            self.battles.enable_beliefs19(belief_bank.native)

        pin = device.type == "cuda"
        self.cat_buf = torch.empty((2, count, 17, 4), dtype=torch.long, pin_memory=pin)
        self.num_buf = torch.empty((2, count, 17, 16), dtype=torch.float32, pin_memory=pin)
        self.mask_buf = torch.empty((2, count, 2, 35), dtype=torch.bool, pin_memory=pin)
        self.history_bufs = {
            "history_cat": torch.empty((2, count, 12, 3), dtype=torch.long, pin_memory=pin),
            "history_num": torch.empty((2, count, 12, 6), dtype=torch.float32, pin_memory=pin),
            "history_moves": torch.empty((2, count, 12, 4), dtype=torch.long, pin_memory=pin),
            "history_flags": torch.empty((2, count, 12, 6), dtype=torch.float32, pin_memory=pin),
        }
        self.belief_buf = (
            torch.empty((2, count, 17, 16), dtype=torch.float32, pin_memory=pin)
            if belief_bank is not None else None
        )
        self.facts_buf = np.empty((2, count, 2, 6, 20), dtype=np.uint32)
        self.codes_buf = np.empty(count, dtype=np.uint8)
        self.step_elapsed_buf = np.empty(count, dtype=np.float32)
        self.after_buf = np.empty(count, dtype=np.int64)
        self._cat_ptr = self.cat_buf.data_ptr()
        self._num_ptr = self.num_buf.data_ptr()
        self._mask_ptr = self.mask_buf.data_ptr()
        self._facts_ptr = self.facts_buf.ctypes.data
        self._ls_ptr = self.learner_sides.ctypes.data
        self._bel_ptr = self.belief_buf.data_ptr() if self.belief_buf is not None else 0
        self._codes_ptr = self.codes_buf.ctypes.data
        self._step_elapsed_ptr = self.step_elapsed_buf.ctypes.data
        self._after_ptr = self.after_buf.ctypes.data
        self._p1a_buf = np.empty((count, 2), dtype=np.uint8)
        self._p2a_buf = np.empty((count, 2), dtype=np.uint8)
        self._p1_pivot_buf = np.empty((count, 2), dtype=np.uint8)
        self._p2_pivot_buf = np.empty((count, 2), dtype=np.uint8)
        self._p1a_ptr = self._p1a_buf.ctypes.data
        self._p2a_ptr = self._p2a_buf.ctypes.data
        self._p1_pivot_ptr = self._p1_pivot_buf.ctypes.data
        self._p2_pivot_ptr = self._p2_pivot_buf.ctypes.data
        self._has_step18_into = hasattr(self.battles, "step18_into")
        if not hasattr(self.battles, "pivot_masks18") or not self._has_step18_into:
            raise RuntimeError("Installed vgc_engine lacks pivot-aware model18 batch methods; rebuild it before training")

    @torch.no_grad()
    def reset_done(self, model, opponent, amp, *, balanced=False):
        indices = [i for i in range(self.count) if self.done[i]]
        if not indices:
            return
        if self.specialist_team is not None:
            ours = [self.specialist_team for _ in indices]
        else:
            ours = [self._rng.randrange(len(self.pool)) for _ in indices]
        theirs = [self._rng.randrange(len(self.pool)) for _ in indices]
        sampled = []
        for policy, a, b in ((model, ours, theirs), (opponent, theirs, ours)):
            cat, num, moves = [torch.from_numpy(x).to(self.device) for x in self.pool.observations(a, b)]
            with amp():
                output = policy(preview_cat=cat, preview_num=num, preview_moves=moves, preview_only=True)
                dist = torch.distributions.Categorical(logits=temperature_logits(output["preview_plan"], policy_temperature(policy)))
                plans = dist.sample()
            sampled.append((plans.cpu().tolist(), dist.log_prob(plans).cpu().tolist(),
                            output["preview_value"].float().cpu().tolist()))
        p1_json, p2_json, seeds = [], [], []
        for j, i in enumerate(indices):
            side = i % 2 if balanced else self._rng.randrange(2)
            plan_us, plan_them = sampled[0][0][j], sampled[1][0][j]
            us_json = self.pool.selected_json(ours[j], plan_us)
            them_json = self.pool.selected_json(theirs[j], plan_them)
            # Learner occupies physical side `side`; p1 is always physical side 0.
            p1_json.append(us_json if side == 0 else them_json)
            p2_json.append(them_json if side == 0 else us_json)
            seeds.append(self._rng.randrange(2**63))
            self.full_teams[i] = (ours[j], theirs[j])
            self.learner_sides[i] = side
            picks = (PREVIEW_PLANS[plan_us], PREVIEW_PLANS[plan_them])
            self.native_picks[i] = picks if side == 0 else picks[::-1]
            self.preview_decisions[i] = PreviewDecision(
                ours[j], theirs[j], plan_us, sampled[0][1][j], sampled[0][2][j],
            )
            self.elapsed_turns[i] = 0
            self.episode_turns[i] = 0
            self.done[i] = False
        # reset_slots re-enables belief recording and starts fresh posteriors.
        self.battles.reset_slots(indices, p1_json, p2_json, seeds)

    def encode(self):
        self.battles.encode18_into(
            self._cat_ptr,
            self._num_ptr,
            self._mask_ptr,
            self._facts_ptr,
            self._ls_ptr,
            self._bel_ptr,
        )
        cat_np = self.cat_buf.numpy()
        num_np = self.num_buf.numpy()
        if self.track_belief_targets:
            self.belief_targets = hidden_belief_targets(cat_np, num_np)

        if self.belief_buf is not None:
            belief_np = self.belief_buf.numpy()
            for actor in range(2):
                sp = belief_np[:, :, actor, 15] != 0
                num_np[:, :, actor, 7][sp] = 0

        mask_closed_observation(cat_np, num_np)
        facts = self.facts_buf
        apply_known_active_sets(cat_np, facts, self.learner_sides)
        encoded_history = known_history(self.pool, self.full_teams, self.native_picks, self.learner_sides, facts)

        for k, v in zip(("history_cat", "history_num", "history_moves", "history_flags"), encoded_history):
            self.history_bufs[k].numpy()[:] = v

        history = {k: v.to(self.device, non_blocking=True) for k, v in self.history_bufs.items()}
        if self.belief_buf is not None:
            history["belief_features"] = self.belief_buf.to(self.device, non_blocking=True)

        cat_t = self.cat_buf.to(self.device, non_blocking=True)
        num_t = self.num_buf.to(self.device, non_blocking=True)
        mask_t = self.mask_buf.to(self.device, non_blocking=True)
        return cat_t, num_t, mask_t, history

    def pivot_masks(self, actions):
        """Defer pivot choices to the actual mid-turn replacement prompt."""
        masks = np.zeros((2, self.count, 2, 35), dtype=bool)
        masks[..., 18] = True
        return torch.from_numpy(masks).to(self.device, non_blocking=True)

    def step(self, actions, pivots=None):
        """actions: (count, 4) = [learner_a, learner_b, opp_a, opp_b] per slot."""
        actions = np.asarray(actions)
        ls = self.learner_sides[:, None]
        learner = actions[:, 0:2]
        opp = actions[:, 2:4]
        p1a = np.where(ls == 0, learner, opp).astype(np.uint8)
        p2a = np.where(ls == 0, opp, learner).astype(np.uint8)
        if pivots is None:
            pivots = np.full((self.count, 4), 18, dtype=np.uint8)
        pivots = np.asarray(pivots, dtype=np.uint8)
        learner_pivot, opp_pivot = pivots[:, :2], pivots[:, 2:4]
        p1_pivot = np.where(ls == 0, learner_pivot, opp_pivot).astype(np.uint8)
        p2_pivot = np.where(ls == 0, opp_pivot, learner_pivot).astype(np.uint8)

        if self._has_step18_into:
            np.copyto(self._p1a_buf, p1a)
            np.copyto(self._p2a_buf, p2a)
            np.copyto(self._p1_pivot_buf, p1_pivot)
            np.copyto(self._p2_pivot_buf, p2_pivot)
            repl, mega = self.battles.step18_into(
                self._p1a_ptr,
                self._p2a_ptr,
                self._codes_ptr,
                self._step_elapsed_ptr,
                self._after_ptr,
                self._ls_ptr, self._p1_pivot_ptr, self._p2_pivot_ptr,
            )
            self.replacement_decisions += repl
            self.mega_declarations += mega
            codes = self.codes_buf
            self.elapsed_turns[:] = self.step_elapsed_buf
            after = self.after_buf
        else:
            pre_repl = set(self.battles.needs_replacements_indices(None))
            self.mega_declarations += int((learner >= MEGA_START).any(axis=1).sum())
            self.replacement_decisions += sum(
                1 for i in range(self.count)
                if i in pre_repl and ((16 <= learner[i, 0] <= 18) or (16 <= learner[i, 1] <= 18))
            )
            before = np.asarray(self.battles.turns(), dtype=np.int64)
            idx = np.arange(self.count, dtype=np.int64)
            codes = np.frombuffer(
                self.battles.step18_action_array(np.ascontiguousarray(p1a), np.ascontiguousarray(p2a), idx),
                np.uint8,
            )
            after = np.asarray(self.battles.turns(), dtype=np.int64)
            self.elapsed_turns = (after - before).astype(np.float32)

        self.episode_turns += self.elapsed_turns
        rewards = np.zeros(self.count, np.float32)
        dones = np.zeros(self.count, np.float32)
        games = capped = 0

        # Fast vectorized scan for finished battles
        terminal = (codes != 1)
        truncated = (~terminal) & (after >= self.max_turns)
        ended = terminal | truncated

        if np.any(ended):
            for i in np.where(ended)[0]:
                code = int(codes[i])
                is_trunc = bool(truncated[i])
                if code in (2, 3):
                    winner = 0 if code == 2 else 1
                    rewards[i] = 1.0 if winner == int(self.learner_sides[i]) else -1.0
                dones[i] = 1.0
                self.done[i] = True
                games += 1
                capped += int(is_trunc)
                if not is_trunc:
                    self.preview_results.append(PreviewResult(self.preview_decisions[i], float(rewards[i])))

        return (torch.from_numpy(rewards).to(self.device, non_blocking=True),
                torch.from_numpy(dones).to(self.device, non_blocking=True), games, capped)

    def belief_evidence(self):
        return self.battles.belief_evidence() if self.belief_bank is not None else (0, 0)
