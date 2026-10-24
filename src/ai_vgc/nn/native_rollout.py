"""Persistent Bo1 slots with explicit terminal and bootstrapped segment ends.

Every sampled learner decision is exported exactly once. Battle state survives
collection boundaries, while trajectories contain only the current behavior
policy's decisions. Frozen opponents belong to a runner, not a horizon.
"""
from __future__ import annotations

import time
import numpy as np
import torch

from ai_vgc.nn.encode import N_ACT
from ai_vgc.nn.player import sample_packed_batch
from ai_vgc.nn.vgc_env import _PACKED_KEYS, packed_policy_arrays


class Bo1Rollout:
    def __init__(self, owner, slots, league, learner_first=True, ots=None):
        self.owner, self.slots, self.league = owner, slots, league
        # Every game in this rollout shares one sheet mode. A 50/50 open+closed
        # curriculum is delivered as two homogeneous rollouts (see
        # NativeBatchSelfPlay.collect_persistent_league_data), not by mixing
        # open and closed games in one batch — the native engine mishandles a
        # heterogeneous persistent batch at team selection.
        self.ots = bool(owner.config.ots if ots is None else ots)
        self.sides = (np.arange(slots, dtype=np.intp) + int(not learner_first)) % 2
        self.battles = owner._battles(slots, self.sides if league else None)
        self.turns = np.zeros(slots, dtype=np.int32)
        self.terminal = ['continue'] * slots
        self.pending_preview = np.ones(slots, dtype=bool) if owner.config.team_preview else np.zeros(slots, dtype=bool)
        self.version = 0

    def packed(self, indices, preview=False):
        return packed_policy_arrays(self.battles.policy_tensors(
            self.owner.config.rating, ots=self.ots,
            indices=np.asarray(indices).tolist(), preview=preview,
            cts_version=self.owner.config.cts_version))

    @torch.no_grad()
    def values(self, model, indices):
        """Value only: bootstrap evaluation must not consume policy-sampling RNG."""
        packed = self.packed(indices)
        rows = (2*np.arange(len(indices)) + self.sides[indices]) if self.league else np.arange(2*len(indices))
        device = next(model.parameters()).device
        obs = {k: torch.from_numpy(v[rows].copy()).to(device) for k, v in packed.items()}
        if device.type == 'cuda':
            with torch.autocast('cuda', dtype=torch.float16):
                logits = model.value(model.encode(obs)[0])[:, 0]
        else:
            logits = model.value(model.encode(obs)[0])[:, 0]
        values = logits.float().sigmoid().cpu().numpy()
        if not np.isfinite(values).all():
            raise RuntimeError('Nonfinite rollout bootstrap value')
        return values

    def collect(self, learner, opponent, horizon):
        if horizon < 1:
            raise ValueError('Collection horizon must be positive')
        started = time.perf_counter()
        stores = {k: [] for k in (*_PACKED_KEYS, 'action', 'logp', 'value', 'preview')}
        rows_by_side = [[] for _ in range(2*self.slots)]
        segments = []
        total_rows = 0
        outcomes, turns, p1_rewards, p2_rewards = [], [], [], []
        cutoffs = 0
        replacement_states = 0

        def append(obs, actions, logp, values, physical, preview):
            nonlocal total_rows
            for k in _PACKED_KEYS:
                stores[k].append(obs[k])
            stores['action'].append(actions.astype(np.int16, copy=False))
            stores['logp'].append(logp.astype(np.float32, copy=False))
            stores['value'].append(values.astype(np.float32, copy=False))
            stores['preview'].append(np.full(len(actions), preview, dtype=bool))
            for offset, side in enumerate(physical):
                rows_by_side[int(side)].append(total_rows+offset)
            total_rows += len(actions)

        def sample(packed, indices, preview=False):
            local = np.arange(len(indices), dtype=np.intp)
            if not self.league:
                obs, acts, lp, values = sample_packed_batch(
                    learner, packed, np.arange(2*len(indices)), greedy=self.owner.config.greedy,
                    use_compile=self.owner.config.compile_inference and not preview)
                physical = (2*indices[:, None] + np.arange(2)).reshape(-1)
                append(obs, acts, lp, values, physical, preview)
                return acts
            mine = 2*local+self.sides[indices]
            theirs = 2*local+1-self.sides[indices]
            obs, acts, lp, values = sample_packed_batch(learner, packed, mine, greedy=self.owner.config.greedy)
            _, other, _, _ = sample_packed_batch(opponent, packed, theirs, greedy=self.owner.config.greedy)
            append(obs, acts, lp, values, 2*indices+self.sides[indices], preview)
            physical = np.empty((2*len(indices), 2), dtype=np.uint8)
            physical[mine], physical[theirs] = acts, other
            return physical

        def preview(indices):
            source = self.packed(indices, preview=True)
            picked = np.empty((2*len(indices), 0), dtype=np.uint8)
            rows = np.arange(len(picked))
            for _ in range(2):
                obs = dict(source)
                obs['tok_num'] = source['tok_num'].copy()
                obs['tok_num'][:, :6, 76] = 0
                mask = np.zeros((len(picked), 2, N_ACT), dtype=bool)
                mask[:, :, 1:7] = True
                for prior in picked.T:
                    obs['tok_num'][rows, prior-1, 76] = 1
                    mask[rows, :, prior] = False
                obs['mask'] = mask
                acts = sample(obs, indices, True)
                if np.any((acts < 1) | (acts > 6)) or np.any(acts[:, 0] == acts[:, 1]):
                    raise RuntimeError('Invalid sampled preview pair')
                picked = np.concatenate((picked, acts), axis=1)
            self.battles.select_teams_at(indices.tolist(), (picked[::2]-1).tolist(), (picked[1::2]-1).tolist())
            self.pending_preview[indices] = False

        def close(game, reward=None, boot=None):
            sides = [int(self.sides[game])] if self.league else [0, 1]
            for j, side in enumerate(sides):
                row = 2*game+side
                if not rows_by_side[row]:
                    continue
                r = 0.0 if reward is None else (reward if side == 0 else 1.0-reward)
                segments.append((rows_by_side[row], r, reward is not None, 0.0 if boot is None else float(boot[j])))
                rows_by_side[row] = []

        all_indices = np.arange(self.slots, dtype=np.intp)
        for _ in range(horizon):
            pending = np.flatnonzero(self.pending_preview)
            if len(pending):
                preview(pending)
            acts = sample(self.packed(all_indices), all_indices)
            active = self.owner._step_packed(self.battles, acts[::2], acts[1::2], all_indices, self.terminal)
            while True:
                indices = np.asarray(self.battles.needs_replacements_indices(np.flatnonzero(active).tolist()), dtype=np.intp)
                if not len(indices):
                    break
                replacement_states += len(indices) * (1 if self.league else 2)
                acts = sample(self.packed(indices), indices)
                active = self.owner._step_packed(self.battles, acts[::2], acts[1::2], indices, self.terminal)
            self.turns += 1
            cutoff = active & (self.turns >= self.owner.config.max_turns)
            cut_indices = np.flatnonzero(cutoff)
            boots = self.values(learner, cut_indices) if len(cut_indices) else []
            for j, game in enumerate(cut_indices):
                boot = boots[j:j+1] if self.league else boots[2*j:2*j+2]
                close(int(game), boot=boot)
            cutoffs += len(cut_indices)
            for game in np.flatnonzero(~active):
                result = self.terminal[int(game)]
                r = 1.0 if result == 'p1_win' else 0.0 if result == 'p2_win' else .5
                close(int(game), reward=r)
                learner_reward = r if not self.league or self.sides[game] == 0 else 1-r
                outcomes.append(learner_reward)
                turns.append(int(self.turns[game]))
                (p1_rewards if self.sides[game] == 0 else p2_rewards).append(learner_reward)
            finished = np.flatnonzero(~active | cutoff)
            if len(finished):
                self.sides[finished] = 1-self.sides[finished]
                self.owner._reset_slots(self.battles, finished, self.sides[finished] if self.league else None)
                # Bo1: each game is its own one-game series, so the per-slot
                # series tracker must reset alongside the battle. Otherwise its
                # game_number keeps climbing across the many games a persistent
                # slot plays in a horizon and overflows the length-3 per-game
                # arrays (t1_double_target/lead_pairs), panicking the preview
                # packing's write_ser_glob. Bo3Rollout resets on series end; the
                # Bo1 path previously never reset it at all.
                self.battles.reset_series_slots(finished.tolist())
                self.turns[finished] = 0
                self.pending_preview[finished] = self.owner.config.team_preview
                for game in finished:
                    self.terminal[int(game)] = 'continue'

        # Only live, already-started games have tails. Fresh resets are not
        # sampled until the next horizon, avoiding unused preview decisions.
        unfinished = np.asarray([g for g in all_indices if rows_by_side[2*g] or rows_by_side[2*g+1]], dtype=np.intp)
        if len(unfinished):
            boots = self.values(learner, unfinished)
            for j, game in enumerate(unfinished):
                close(int(game), boot=boots[j:j+1] if self.league else boots[2*j:2*j+2])
        joined = {k: np.concatenate(v) for k, v in stores.items()}
        order = np.asarray([row for rows, _, _, _ in segments for row in rows], dtype=np.intp)
        if len(order) != total_rows or len(np.unique(order)) != total_rows:
            raise RuntimeError('Rollout must export each sampled learner decision exactly once')
        data = {k: v[order] for k, v in joined.items()}
        for k, dtype in [('reward', np.float32), ('last', bool), ('terminated', bool), ('bootstrap', np.float32)]:
            data[k] = np.zeros(total_rows, dtype=dtype)
        cursor = 0
        for rows, r, terminal, boot in segments:
            cursor += len(rows)
            data['reward'][cursor-1] = r
            data['last'][cursor-1] = True
            data['terminated'][cursor-1] = terminal
            data['bootstrap'][cursor-1] = boot
        data['policy_version'] = np.full(total_rows, self.version, dtype=np.int64)
        self.version += 1
        elapsed = time.perf_counter()-started
        stats = dict(slots=self.slots, waves=horizon, completed_games=len(outcomes), completed_series=0,
                     completed_player_episodes=int(data['terminated'].sum()), mean_turns=float(np.mean(turns)) if turns else 0.,
                     normal_states=self.slots*horizon*(1 if self.league else 2), replacement_states=replacement_states,
                     preview_states=int(data['preview'].sum()), elapsed_s=elapsed, games_s=len(outcomes)/elapsed,
                     generated_rows=total_rows, retained_rows=total_rows, discarded_rows=0,
                     truncated_segments=int((data['last'] & ~data['terminated']).sum()), cutoff_games=cutoffs,
                     continued_games=len(unfinished), policy_version=self.version-1,
                     learner_p1_games=len(p1_rewards), learner_p2_games=len(p2_rewards),
                     learner_p1_reward=float(np.mean(p1_rewards)) if p1_rewards else .5,
                     learner_p2_reward=float(np.mean(p2_rewards)) if p2_rewards else .5)
        return data, stats


class Bo3Rollout:
    def __init__(self, owner, slots, league, learner_first=True):
        self.owner, self.slots, self.league = owner, slots, league
        self.sides = (np.arange(slots, dtype=np.intp) + int(not learner_first)) % 2
        self.battles = owner._battles(slots, self.sides if league else None)
        self.turns = np.zeros(slots, dtype=np.int32)
        self.terminal = ['continue'] * slots
        self.pending_preview = np.ones(slots, dtype=bool) if owner.config.team_preview else np.zeros(slots, dtype=bool)
        self.version = 0

    def packed(self, indices, preview=False):
        return packed_policy_arrays(self.battles.policy_tensors(
            self.owner.config.rating, ots=self.owner.config.ots,
            indices=np.asarray(indices).tolist(), preview=preview,
            cts_version=self.owner.config.cts_version))

    @torch.no_grad()
    def values(self, model, indices):
        """Value only: bootstrap evaluation must not consume policy-sampling RNG."""
        packed = self.packed(indices)
        rows = (2*np.arange(len(indices)) + self.sides[indices]) if self.league else np.arange(2*len(indices))
        device = next(model.parameters()).device
        obs = {k: torch.from_numpy(v[rows].copy()).to(device) for k, v in packed.items()}
        if device.type == 'cuda':
            with torch.autocast('cuda', dtype=torch.float16):
                logits = model.value(model.encode(obs)[0])[:, 0]
        else:
            logits = model.value(model.encode(obs)[0])[:, 0]
        values = logits.float().sigmoid().cpu().numpy()
        if not np.isfinite(values).all():
            raise RuntimeError('Nonfinite rollout bootstrap value')
        return values

    def collect(self, learner, opponent, horizon):
        if horizon < 1:
            raise ValueError('Collection horizon must be positive')
        started = time.perf_counter()
        stores = {k: [] for k in (*_PACKED_KEYS, 'action', 'logp', 'value', 'preview')}
        rows_by_side = [[] for _ in range(2*self.slots)]
        segments = []
        total_rows = 0
        outcomes, turns, p1_rewards, p2_rewards = [], [], [], []
        series_wins = []
        cutoffs = 0
        replacement_states = 0

        def append(obs, actions, logp, values, physical, preview):
            nonlocal total_rows
            for k in _PACKED_KEYS:
                stores[k].append(obs[k])
            stores['action'].append(actions.astype(np.int16, copy=False))
            stores['logp'].append(logp.astype(np.float32, copy=False))
            stores['value'].append(values.astype(np.float32, copy=False))
            stores['preview'].append(np.full(len(actions), preview, dtype=bool))
            for offset, side in enumerate(physical):
                rows_by_side[int(side)].append(total_rows+offset)
            total_rows += len(actions)

        def sample(packed, indices, preview=False):
            local = np.arange(len(indices), dtype=np.intp)
            if not self.league:
                obs, acts, lp, values = sample_packed_batch(
                    learner, packed, np.arange(2*len(indices)), greedy=self.owner.config.greedy,
                    use_compile=self.owner.config.compile_inference and not preview)
                physical = (2*indices[:, None] + np.arange(2)).reshape(-1)
                append(obs, acts, lp, values, physical, preview)
                return acts
            mine = 2*local+self.sides[indices]
            theirs = 2*local+1-self.sides[indices]
            obs, acts, lp, values = sample_packed_batch(learner, packed, mine, greedy=self.owner.config.greedy)
            _, other, _, _ = sample_packed_batch(opponent, packed, theirs, greedy=self.owner.config.greedy)
            append(obs, acts, lp, values, 2*indices+self.sides[indices], preview)
            physical = np.empty((2*len(indices), 2), dtype=np.uint8)
            physical[mine], physical[theirs] = acts, other
            return physical

        def preview(indices):
            source = self.packed(indices, preview=True)
            picked = np.empty((2*len(indices), 0), dtype=np.uint8)
            rows = np.arange(len(picked))
            for _ in range(2):
                obs = dict(source)
                obs['tok_num'] = source['tok_num'].copy()
                obs['tok_num'][:, :6, 76] = 0
                mask = np.zeros((len(picked), 2, N_ACT), dtype=bool)
                mask[:, :, 1:7] = True
                for prior in picked.T:
                    obs['tok_num'][rows, prior-1, 76] = 1
                    mask[rows, :, prior] = False
                obs['mask'] = mask
                acts = sample(obs, indices, True)
                if np.any((acts < 1) | (acts > 6)) or np.any(acts[:, 0] == acts[:, 1]):
                    raise RuntimeError('Invalid sampled preview pair')
                picked = np.concatenate((picked, acts), axis=1)
            self.battles.select_teams_at(indices.tolist(), (picked[::2]-1).tolist(), (picked[1::2]-1).tolist())
            self.pending_preview[indices] = False

        def close(game, reward=None, boot=None):
            sides = [int(self.sides[game])] if self.league else [0, 1]
            for j, side in enumerate(sides):
                row = 2*game+side
                if not rows_by_side[row]:
                    continue
                r = 0.0 if reward is None else (reward if side == 0 else 1.0-reward)
                segments.append((rows_by_side[row], r, reward is not None, 0.0 if boot is None else float(boot[j])))
                rows_by_side[row] = []

        all_indices = np.arange(self.slots, dtype=np.intp)
        for _ in range(horizon):
            pending = np.flatnonzero(self.pending_preview)
            if len(pending):
                preview(pending)
            acts = sample(self.packed(all_indices), all_indices)
            active = self.owner._step_packed(self.battles, acts[::2], acts[1::2], all_indices, self.terminal)
            while True:
                indices = np.asarray(self.battles.needs_replacements_indices(np.flatnonzero(active).tolist()), dtype=np.intp)
                if not len(indices):
                    break
                replacement_states += len(indices) * (1 if self.league else 2)
                acts = sample(self.packed(indices), indices)
                active = self.owner._step_packed(self.battles, acts[::2], acts[1::2], indices, self.terminal)
            self.turns += 1
            cutoff = active & (self.turns >= self.owner.config.max_turns)
            cut_indices = np.flatnonzero(cutoff)
            if len(cut_indices):
                boots = self.values(learner, cut_indices)
                for j, game in enumerate(cut_indices):
                    boot = boots[j:j+1] if self.league else boots[2*j:2*j+2]
                    close(int(game), boot=boot)
                cutoffs += len(cut_indices)
                self.battles.record_cutoff_slots(cut_indices.tolist())
            for game in np.flatnonzero(~active):
                result = self.terminal[int(game)]
                r = 1.0 if result == 'p1_win' else 0.0 if result == 'p2_win' else .5
                close(int(game), reward=r)
                learner_reward = r if not self.league or self.sides[game] == 0 else 1-r
                outcomes.append(learner_reward)
                turns.append(int(self.turns[game]))
                (p1_rewards if self.sides[game] == 0 else p2_rewards).append(learner_reward)
            finished = np.flatnonzero(~active | cutoff)
            if len(finished):
                reuse_series = np.zeros(len(finished), dtype=bool)
                series_finished = []
                for row, game in enumerate(finished):
                    g = int(game)
                    if self.battles.is_series_over(g):
                        series_finished.append(g)
                        score = self.battles.series_score(g)
                        if self.league:
                            us_wins = score[self.sides[g]]
                            them_wins = score[1 - self.sides[g]]
                        else:
                            us_wins, them_wins = score[0], score[1]
                        series_reward = 1.0 if us_wins > them_wins else (0.0 if us_wins < them_wins else 0.5)
                        series_wins.append(series_reward)
                    else:
                        reuse_series[row] = True

                if series_finished:
                    sf_arr = np.asarray(series_finished, dtype=np.intp)
                    if self.league:
                        self.sides[sf_arr] = 1 - self.sides[sf_arr]
                    self.battles.reset_series_slots(series_finished)

                self.owner._reset_slots(
                    self.battles, finished,
                    self.sides[finished] if self.league else None,
                    reuse_series=reuse_series,
                )
                self.turns[finished] = 0
                self.pending_preview[finished] = self.owner.config.team_preview
                for game in finished:
                    self.terminal[int(game)] = 'continue'

        unfinished = np.asarray([g for g in all_indices if rows_by_side[2*g] or rows_by_side[2*g+1]], dtype=np.intp)
        if len(unfinished):
            boots = self.values(learner, unfinished)
            for j, game in enumerate(unfinished):
                close(int(game), boot=boots[j:j+1] if self.league else boots[2*j:2*j+2])
        joined = {k: np.concatenate(v) for k, v in stores.items()}
        order = np.asarray([row for rows, _, _, _ in segments for row in rows], dtype=np.intp)
        if len(order) != total_rows or len(np.unique(order)) != total_rows:
            raise RuntimeError('Rollout must export each sampled learner decision exactly once')
        data = {k: v[order] for k, v in joined.items()}
        for k, dtype in [('reward', np.float32), ('last', bool), ('terminated', bool), ('bootstrap', np.float32)]:
            data[k] = np.zeros(total_rows, dtype=dtype)
        cursor = 0
        for rows, r, terminal, boot in segments:
            cursor += len(rows)
            data['reward'][cursor-1] = r
            data['last'][cursor-1] = True
            data['terminated'][cursor-1] = terminal
            data['bootstrap'][cursor-1] = boot
        data['policy_version'] = np.full(total_rows, self.version, dtype=np.int64)
        self.version += 1
        elapsed = time.perf_counter()-started
        stats = dict(slots=self.slots, waves=horizon, completed_games=len(outcomes),
                     completed_series=len(series_wins),
                     series_win_rate=float(np.mean(series_wins)) if series_wins else 0.5,
                     completed_player_episodes=int(data['terminated'].sum()),
                     mean_turns=float(np.mean(turns)) if turns else 0.,
                     normal_states=self.slots*horizon*(1 if self.league else 2),
                     replacement_states=replacement_states,
                     preview_states=int(data['preview'].sum()), elapsed_s=elapsed,
                     games_s=len(outcomes)/elapsed if elapsed else 0.0,
                     series_s=len(series_wins)/elapsed if elapsed else 0.0,
                     generated_rows=total_rows, retained_rows=total_rows, discarded_rows=0,
                     truncated_segments=int((data['last'] & ~data['terminated']).sum()), cutoff_games=cutoffs,
                     continued_games=len(unfinished), policy_version=self.version-1,
                     learner_p1_games=len(p1_rewards), learner_p2_games=len(p2_rewards),
                     learner_p1_reward=float(np.mean(p1_rewards)) if p1_rewards else .5,
                     learner_p2_reward=float(np.mean(p2_rewards)) if p2_rewards else .5)
        return data, stats
