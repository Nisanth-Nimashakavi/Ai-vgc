"""Improve the imitation-learned policy by self-play reinforcement learning (PPO).

    uv run python -m ai_vgc.nn.rl --init data/models/archive/mb-bo3-bc-v1.pt --out data/models/archive/mb-bo3-ppo-v1.pt
    uv run python -m ai_vgc.nn.rl --iters 2 --games 64 --workers 4   # quick local test

Each iteration:

1. Worker processes play `--games` battles on local Showdown servers with the
   current policy, recording every decision (inputs, action, log-prob, value).
   Opponents are mixed so the policy doesn't just chase its own latest version:
   itself (both sides' decisions are used), the imitation policy it started
   from, a random earlier snapshot, and the heuristic bot.
2. The learner (GPU) turns the win/loss at the end of each game into
   per-decision advantages (GAE, gamma = 1, with the value head as baseline)
   and takes a few PPO epochs over the batch.

Changes from vgc-bench's PPO:

- Start from the imitation policy and keep a KL penalty towards it, so
  self-play refines human play instead of drifting into strange strategies
  that only beat itself.
- The value head predicts win probability (sigmoid, as in imitation training),
  so rewards are 1 / 0.5 / 0 and the value loss stays a BCE.
- The first `--value-warmup` iterations train only the value head: the
  imitation value head is overconfident, and a bad baseline makes early policy
  updates noisy.

Double oracle (`--mix` with `nash`, see `do_rl` in scripts/slurm/archive/mb-bo3-ppo-v1.sh):
the pool of snapshots (plus `--pool-init` checkpoints) forms a meta-game. Each
new snapshot plays every pool member (`--meta-games` each), the win-rate table
is solved for its Nash mixture (the mix of snapshots that is hardest to beat),
and "nash" opponents are drawn with those weights. Snapshots that are
dominated get weight ~0, so games go to the opponents that still matter. The
table lives in <out stem>_meta.json, so --resume keeps it.

Regularized Nash dynamics (`--reg-every N`, from DeepNash): every N iterations
the KL anchor is replaced by the current policy, instead of pulling towards the
imitation policy forever. Each phase then solves a regularized game around the
last policy, and the sequence of anchors converges towards a Nash equilibrium
rather than stalling where the fixed anchor and the pool balance out. The
anchor is saved to <out stem>_reg.pt, so --resume keeps it.

Bo3 formats (e.g. --format gen9championsvgc2026regmcbo3): jobs play series instead of games,
about the same number of games in all. Each game is still an episode with its own result,
and games 2-3 carry the series context (series.py), so a series model (train.py --series)
can learn to adapt, and self-play punishes repeating what lost.

The current policy is saved to --out after every iteration (same format as
train.py, so `player.py` can load it); snapshots go to <out stem>_snapshots/.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import multiprocessing
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ai_vgc.nn.tb import scalars, writer

KEYS = ["tok_cat", "tok_num", "mv_cat", "mv_dyn", "dmg", "glob", "act_tok", "mask", "ser_tok", "ser_mv", "ser_glob"]

# ---------------------------------------------------------------- workers

_W: dict = {}


def _init(ports: list[int], fmt: str, concurrency: int, rating: float, teams: str | None = None,
          my_team: str | None = None, change_after_loss: bool = False, opp_change_after_loss: bool = False,
          closed_sheets: bool = False, speed_inference: bool = False, set_guess: bool = False,
          damage_inference: bool = False, device: str = "cpu", aux_coef: float = 0.0,
          native_max_turns: int = 128, native_seed: int = 0, native_preview: bool = False,
          native_slots: int = 0, native_horizon: int = 0, native_compile: bool = False,
          native_learner_team: str | None = None, native_opponent_teams: str | None = None) -> None:
    torch.set_num_threads(1)
    _W.update(ports=ports, fmt=fmt, concurrency=concurrency, rating=rating, teams=teams, my_team=my_team,
              change_after_loss=change_after_loss, opp_change_after_loss=opp_change_after_loss,
              closed_sheets=closed_sheets, speed_inference=speed_inference,
              set_guess=set_guess, damage_inference=damage_inference, device=device, aux_coef=aux_coef,
              native_max_turns=native_max_turns, native_seed=native_seed, native_preview=native_preview,
              native_slots=native_slots, native_horizon=native_horizon, native_compile=native_compile,
              native_learner_team=native_learner_team, native_opponent_teams=native_opponent_teams)


def _players(policy: str) -> dict:
    from poke_env import ServerConfiguration
    from poke_env.player import SimpleHeuristicsPlayer
    from poke_env.teambuilder import ConstantTeambuilder

    from ai_vgc.nn.player import NNPlayer, load_policy
    from ai_vgc.showdown import account
    from ai_vgc.teams import RandomPoolTeambuilder

    if "me" not in _W:
        port = _W["ports"][os.getpid() % len(_W["ports"])]
        common = dict(
            battle_format=_W["fmt"], max_concurrent_battles=_W["concurrency"],
            accept_open_team_sheet=not _W["closed_sheets"],
            log_level=40,
            server_configuration=ServerConfiguration(f"ws://localhost:{port}/showdown/websocket",
                                                     "https://play.pokemonshowdown.com/action.php?"),
        )
        device = _W.get("device", "cpu")
        model = load_policy(policy, device)
        r = _W["rating"]

        def teams() -> RandomPoolTeambuilder:
            return RandomPoolTeambuilder(Path(_W["teams"])) if _W["teams"] else RandomPoolTeambuilder()

        # "mirror" shares the learner's model: self-play games record both sides.
        mine = ConstantTeambuilder(Path(_W["my_team"]).read_text()) if _W["my_team"] else teams()
        _W["me"] = NNPlayer(model, r, record=True, account_configuration=account("rl"),
                            team=mine, **common)
        # As on the ladder: after losing a Bo3 game, lead with another pair. That preview is a rule,
        # not a policy sample, so it isn't recorded; the rest of the game is.
        _W["me"].change_after_loss = _W["change_after_loss"]
        _W["self"] = NNPlayer(model, r, record=True, account_configuration=account("self"),
                              team=teams(), **common)
        # Mirror a human opponent who leads differently in game N+1 after losing game N, so the
        # trainee learns to beat an opponent that adapts (not just to vary its own play).
        _W["self"].change_after_loss = _W["opp_change_after_loss"]
        _W["snap"] = NNPlayer(load_policy(policy, device), r, account_configuration=account("snap"),
                              team=teams(), **common)
        _W["snap"].change_after_loss = _W["opp_change_after_loss"]
        _W["evalA"] = NNPlayer(load_policy(policy, device), r, account_configuration=account("evA"),
                               team=teams(), **common)
        _W["evalB"] = NNPlayer(load_policy(policy, device), r, account_configuration=account("evB"),
                               team=teams(), **common)
        _W["heuristic"] = SimpleHeuristicsPlayer(account_configuration=account("heur"),
                                                 team=teams(), **common)
        for k in ("me", "self", "snap", "evalA", "evalB"):  # every model in the worker reads the same features
            _W[k].speed_inference = _W.get("speed_inference", False)
            _W[k].set_guess = (_W["teams"], _W["fmt"]) if _W.get("set_guess") and _W["teams"] else None
            _W[k].damage_inference = _W.get("damage_inference", False)
        if _W.get("aux_coef", 0.0) > 0:  # label self-play episodes so the aux heads don't decay (rl.py --aux-coef)
            _W["me"].aux_labels = True
            _W["self"].aux_labels = True
    return _W


def _state(path: str) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)["state"]


def _native_model(path: str, device: str) -> torch.nn.Module:
    """Cache worker-local policy objects and refresh only changed checkpoints."""
    from ai_vgc.nn.player import load_policy

    resolved = str(Path(path).resolve())
    stat = os.stat(resolved)
    signature = (stat.st_mtime_ns, stat.st_size)
    cache = _W.setdefault("native_models", {})
    cached = cache.get(resolved)
    if cached is None:
        model = load_policy(resolved, device)
        cache[resolved] = (signature, model)
        cached = cache[resolved]
    old_signature, model = cached
    if old_signature != signature:
        model.load_state_dict(_state(resolved))
        cache[resolved] = (signature, model)
    model.eval()
    return model


def _load_into(player, path: str) -> None:
    """Give an opponent `player` the policy at `path`. Checkpoints with another architecture
    (e.g. archive/mb-bo3-do-v1, from before the blocked flag and opponent head) get their own model."""
    from ai_vgc.nn.player import load_policy

    device = _W.get("device", "cpu")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    own = player.__dict__.setdefault("own_model", player.model)
    if ckpt["config"] == own.config:
        player.model = own
        own.load_state_dict(ckpt["state"])
    else:
        cache = _W.setdefault("other_models", {})
        if path not in cache:
            cache[path] = load_policy(path, device)
        player.model = cache[path]


def _flatten(episodes: list[tuple[list[dict], float]]) -> dict[str, np.ndarray]:
    steps = [s for ep, _ in episodes for s in ep]
    out = {k: np.stack([s[k] for s in steps]) for k in KEYS + ["action"]}
    if steps and "opp" in steps[0]:
        out["opp"] = np.stack([s["opp"] for s in steps])
    out["logp"] = np.array([s["logp"] for s in steps], np.float32)
    out["value"] = np.array([s["value"] for s in steps], np.float32)
    out["preview"] = np.array([s["preview"] for s in steps], bool)
    out["reward"] = np.zeros(len(steps), np.float32)
    out["last"] = np.zeros(len(steps), bool)
    i = 0
    for ep, r in episodes:
        i += len(ep)
        out["reward"][i - 1] = r
        out["last"][i - 1] = True
    return out


def _play_native(job: tuple[str, str, str | None, int]) -> tuple[str, dict[str, np.ndarray] | None, list[float]]:
    """Collect one fixed-horizon pure-native PPO batch.

    The learner checkpoint is fixed for the complete horizon.  When ``kind``
    names a frozen policy, the engine collector records only learner-side rows;
    otherwise it uses the cheaper symmetric self-play collector unless a
    specialist learner team was requested.
    """
    policy, kind, opp_path, n = job
    try:
        from ai_vgc.nn.vgc_env import NativeBatchSelfPlay, NativeVGCConfig

        torch.set_num_threads(1)
        device = _W.get("device", "cpu")
        learner = _native_model(policy, device)
        opponent = _native_model(opp_path, device) if opp_path else learner
        if opponent is not learner:
            for parameter in opponent.parameters():
                parameter.requires_grad_(False)

        rollout_id = _W.get("native_rollout_id", 0)
        _W["native_rollout_id"] = rollout_id + 1
        runner = NativeBatchSelfPlay(NativeVGCConfig(
            fmt=_W["fmt"], teams_dir=_W.get("teams"), rating=_W["rating"],
            learner_team=_W.get("native_learner_team"),
            opponent_teams_dir=_W.get("native_opponent_teams"),
            greedy=False, seed=((os.getpid() << 16) ^ _W.get("native_seed", 0)) + rollout_id,
            max_turns=_W.get("native_max_turns", 128),
            team_preview=_W.get("native_preview", False),
            compile_inference=_W.get("native_compile", False),
        ))
        slots = _W.get("native_slots", 0)
        horizon = _W.get("native_horizon", 0)
        specialist = _W.get("native_learner_team") is not None
        if not slots:
            if kind != "self" or specialist:
                raise ValueError("native league rollouts require --native-slots and --native-horizon")
            data, rewards = runner.play_data(learner, n)
            return kind, data, rewards
        if kind == "self" and not specialist:
            data, _stats = runner.collect_persistent_data(learner, slots, horizon)
        else:
            data, _stats = runner.collect_persistent_league_data(
                learner, opponent, slots, horizon,
            )
        terminal_rewards = data["reward"][data["last"]].astype(np.float32, copy=False)
        # Symmetric collection compacts p1/p2 episodes consecutively. Retain
        # the physical P1 result for familiar self-play win-rate reporting;
        # league collection has exactly one learner episode per game.
        rewards = terminal_rewards[::2].tolist() if kind == "self" and not specialist else terminal_rewards.tolist()
        return kind, data if len(data["action"]) else None, rewards
    except Exception as e:
        print(f"native worker {os.getpid()}: {kind} job failed: {e!r}", flush=True)
        return kind, None, []


def _native_actor_configs(args, ports: list[int], dev: torch.device) -> list[dict]:
    """Build persistent actor configs without colocating them with the learner.

    CUDA actor assignment reserves the learner's selected GPU when another
    visible device exists. A one-GPU run deliberately falls back to sharing it;
    multi-GPU training otherwise gets one actor process and CUDA context per
    remaining accelerator. The learner still trains on its own GPU, so this
    avoids loading a second actor model there whenever possible.
    """
    if dev.type == "cuda":
        learner_index = dev.index or 0
        candidates = [index for index in range(torch.cuda.device_count()) if index != learner_index]
        if not candidates:
            candidates = [learner_index]
        actor_count = args.native_actors or len(candidates)
        if actor_count > len(candidates):
            raise ValueError(
                f"native actors ({actor_count}) exceed actor GPUs ({len(candidates)}); "
                "the learner device is reserved"
            )
        devices = [f"cuda:{index}" for index in candidates[:actor_count]]
    else:
        actor_count = args.native_actors or args.workers
        if actor_count < 1:
            raise ValueError("native collection needs at least one actor")
        devices = ["cpu"] * actor_count

    return [dict(
        ports=ports,
        fmt=args.format,
        concurrency=args.concurrency,
        rating=args.rating,
        teams=str(args.teams) if args.teams else None,
        my_team=str(args.my_team) if args.my_team else None,
        change_after_loss=args.change_after_loss,
        opp_change_after_loss=args.opp_change_after_loss,
        closed_sheets=args.closed_sheets,
        speed_inference=args.speed_inference,
        set_guess=args.set_guess,
        damage_inference=args.damage_inference,
        device=device,
        aux_coef=args.aux_coef,
        native_max_turns=args.native_max_turns,
        native_seed=args.native_seed + actor * 1_000_003,
        native_preview=args.native_preview,
        native_slots=args.native_slots,
        native_horizon=args.native_horizon,
        native_compile=args.native_compile,
        native_learner_team=str(args.my_team) if args.my_team else None,
        native_opponent_teams=str(args.native_opponent_teams) if args.native_opponent_teams else None,
    ) for actor, device in enumerate(devices)]


def _native_actor_main(worker_config: dict, requests, results) -> None:
    """Own one native device for the actor lifetime and service PPO horizons."""
    _init(**worker_config)
    while (request := requests.get()) is not None:
        request_id, jobs = request
        try:
            results.put((request_id, [_play_native(job) for job in jobs], None))
        except BaseException as exc:  # propagate worker trace text to the learner
            results.put((request_id, [], repr(exc)))


class NativeActorPool:
    """Long-lived native rollout actors, one CUDA context per selected GPU."""

    def __init__(self, worker_configs: list[dict]):
        self.ctx = multiprocessing.get_context("spawn")
        self.requests = []
        self.results = self.ctx.Queue()
        self.processes = []
        for config in worker_configs:
            request = self.ctx.Queue(maxsize=1)
            process = self.ctx.Process(
                target=_native_actor_main, args=(config, request, self.results), daemon=True,
            )
            process.start()
            self.requests.append(request)
            self.processes.append(process)

    def collect(self, jobs: list[tuple[str, str, str | None, int]]) -> list:
        if len(jobs) != len(self.requests):
            raise ValueError("native actor collection requires one horizon per actor")
        for actor, job in enumerate(jobs):
            self.requests[actor].put((actor, [job]))
        received: list = []
        for _ in self.requests:
            _actor, result, error = self.results.get()
            if error:
                raise RuntimeError(f"native actor failed: {error}")
            received.extend(result)
        return received

    def close(self) -> None:
        for request in self.requests:
            request.put(None)
        for process in self.processes:
            process.join(timeout=30)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        for request in self.requests:
            request.close()
        self.results.close()


def _native_snapshot_choice(pool: list[str], record: dict[str, dict[str, float]]) -> str:
    """Sample a frozen opponent using PFSP's hard-opponent priority.

    ``reward`` is the learner's observed terminal reward against a snapshot.
    Squaring its loss rate emphasizes policies that the learner still cannot
    beat, while the 0.5 prior keeps a fresh checkpoint eligible immediately.
    """
    if not pool:
        raise ValueError("native snapshot pool is empty")
    weights = []
    for path in pool:
        stats = record.get(path, {})
        games = float(stats.get("games", 0.0))
        reward = float(stats.get("reward", 0.5 * games))
        win_rate = reward / games if games else 0.5
        weights.append(max((1.0 - win_rate) ** 2, 1e-3))
    return random.choices(pool, weights)[0]


def _native_record_snapshot_results(
    results: dict[str, list[float]],
    pool: list[str],
    record: dict[str, dict[str, float]],
) -> None:
    """Fold learner rewards from labelled pool horizons into PFSP statistics."""
    pool_set = set(pool)
    for kind, rewards in results.items():
        if not kind.startswith("pool:") or not rewards:
            continue
        path = kind.removeprefix("pool:")
        if path not in pool_set:
            continue
        stats = record.setdefault(path, {"games": 0.0, "reward": 0.0})
        stats["games"] += len(rewards)
        stats["reward"] += float(np.sum(rewards))


def _play(job: tuple[str, str, str | None, int]) -> tuple[str, dict[str, np.ndarray] | None, list[float]]:
    """Play n games of the current policy against `kind`; returns (kind, steps, our rewards)."""
    policy, kind, opp_path, n = job
    try:
        w = _players(policy)
        # Bo3: battle_against counts series (about 2.4 games each). Every game is still its own
        # episode, rewarded by its own result; games 2-3 see the series context (series.py).
        if w["fmt"].endswith("bo3"):
            n = max(1, round(n / 2.4))
        limit = 60 + (25 if w["fmt"].endswith("bo3") else 10) * n
        if kind == "eval":  # meta-game entry: `policy` vs `opp_path`, nothing recorded
            a, b = w["evalA"], w["evalB"]
            _load_into(a, policy)
            _load_into(b, opp_path)
            asyncio.run(asyncio.wait_for(a.battle_against(b, n_battles=n), timeout=limit))
            wins = [1.0 if x.won else 0.0 if x.lost else 0.5 for x in a.battles.values() if x.finished]
            a.reset_battles()
            b.reset_battles()
            return kind, None, wins
        me = w["me"]
        if torch.load(policy, map_location="cpu", weights_only=False)["config"] != me.model.config:
            # Players were built by an eval job for a pool checkpoint of another architecture
            # (e.g. mc-bo1-rnad-v3 without series inputs): rebuild them around the learner's.
            for k in ("me", "self", "snap", "evalA", "evalB", "heuristic"):
                _W.pop(k, None)
            w = _players(policy)
            me = w["me"]
        me.model.load_state_dict(_state(policy))
        opp = w["snap" if kind in ("bc", "pool", "nash", "spec", "human") else kind]
        pool_team = None
        if kind == "spec":  # "model::team file": a specialist on its own team
            from poke_env.teambuilder import ConstantTeambuilder

            opp_path, team_file = opp_path.split("::")
            pool_team, opp._team = opp._team, ConstantTeambuilder(Path(team_file).read_text())
        if opp_path:
            _load_into(opp, opp_path)
        me.episodes.clear()
        if kind == "self":
            opp.episodes.clear()
        asyncio.run(asyncio.wait_for(me.battle_against(opp, n_battles=n), timeout=limit))
        if me.aux_labels:
            me.drain_aux_labels()
        if kind == "self" and opp.aux_labels:
            opp.drain_aux_labels()
        mine = [r for _, r in me.episodes]
        # With --my-team only the learner's side plays that team, so only it is trained on.
        episodes = me.episodes + (opp.episodes if kind == "self" and not w["my_team"] else [])
        data = _flatten(episodes) if episodes else None
        me.episodes.clear()
        if kind == "self":
            opp.episodes.clear()
        me.reset_battles()
        opp.reset_battles()
        me.games.clear()
        opp.__dict__.get("games", []).clear()
        if pool_team is not None:
            opp._team = pool_team
        return kind, data, mine
    except Exception as e:  # a stuck or broken battle: rebuild the players next time
        print(f"worker {os.getpid()}: {kind} job failed: {e!r}", flush=True)
        for k in ("me", "self", "snap", "evalA", "evalB", "heuristic"):
            _W.pop(k, None)
        return kind, None, []


# ---------------------------------------------------------------- double oracle

def nash(win: np.ndarray, iters: int = 20000) -> np.ndarray:
    """Nash mixture of the symmetric zero-sum game with win-rate table `win`
    (win[i, j] = how often i beats j), by regret matching; the average strategy converges."""
    pay = (win - win.T) / 2  # antisymmetric: i's payoff against j
    n = len(pay)
    rx, ry, avg = np.zeros(n), np.zeros(n), np.zeros(n)
    for _ in range(iters):
        x = np.maximum(rx, 0)
        x = x / x.sum() if x.sum() > 0 else np.full(n, 1 / n)
        y = np.maximum(ry, 0)
        y = y / y.sum() if y.sum() > 0 else np.full(n, 1 / n)
        ux, uy = pay @ y, pay @ x  # symmetric game: -pay.T = pay
        rx += ux - x @ ux
        ry += uy - y @ uy
        avg += x
    return avg / avg.sum()


def meta_add(ex, meta: dict, path: str, games: int, job_games: int) -> None:
    """Add `path` to the meta-game: play it against every pool member, extend the table."""
    pool = meta["pool"]
    jobs, owner = [], []
    for j, other in enumerate(pool):
        for _ in range(math.ceil(games / job_games)):
            jobs.append((path, "eval", other, job_games))
            owner.append(j)
    res: dict[int, list[float]] = {}
    for j, (_, _, wins) in zip(owner, ex.map(_play, jobs)):
        res.setdefault(j, []).extend(wins)
    n = len(pool)
    win = np.full((n + 1, n + 1), 0.5)
    win[:n, :n] = np.array(meta["win"]) if n else win[:0, :0]
    for j in range(n):
        w = float(np.mean(res[j])) if res.get(j) else 0.5
        win[n, j], win[j, n] = w, 1 - w
    pool.append(path)
    meta["win"] = win.tolist()
    meta["nash"] = nash(win).tolist()


# ---------------------------------------------------------------- learner

def gae(value_logit: np.ndarray, reward: np.ndarray, last: np.ndarray, lam: float):
    """Advantages and returns with gamma = 1; values are win probabilities."""
    v = 1 / (1 + np.exp(-value_logit.astype(np.float64)))
    adv = np.zeros_like(v)
    g = 0.0
    for i in range(len(v) - 1, -1, -1):
        if last[i]:
            nxt, g = reward[i], 0.0
        else:
            nxt = v[i + 1]
        g = nxt - v[i] + lam * g
        adv[i] = g
    return adv.astype(np.float32), (adv + v).astype(np.float32)


def ppo_update(model, ref, data: dict[str, torch.Tensor], opt, args, value_only: bool,
               vopt=None) -> dict[str, float]:
    from ai_vgc.nn.model import masked, slot_b_mask

    n = len(data["action"])
    stats: dict[str, float] = {"pg": 0, "vf": 0, "ent": 0, "kl": 0, "clip": 0, "n": 0}
    stop = False
    for _ in range(args.ppo_epochs):
        if stop:
            break
        perm = torch.randperm(n, device=data["action"].device)
        for i in range(0, n, args.minibatch):
            b = {k: v[perm[i:i + args.minibatch]] for k, v in data.items()}
            if value_only:
                # Train the value head alone: the encoder is shared with the policy,
                # so letting value gradients into it would move the policy unchecked.
                with torch.no_grad():
                    cls = model.encode(b)[0]
                vf = F.binary_cross_entropy_with_logits(model.value(cls)[:, 0].float(), b["ret"].clamp(0, 1))
                vopt.zero_grad(set_to_none=True)
                vf.backward()
                vopt.step()
                stats["vf"] += vf.item()
                stats["n"] += 1
                continue
            act = b["action"].long()
            ma, mb = b["mask"][:, 0], slot_b_mask(b["mask"][:, 1], act[:, 0])
            # Match the fp16 autocast used at rollout time (player.sample_batch), so the
            # recomputed logp is comparable to b["logp"] and the KL/stop check isn't
            # tripped by fp16-vs-fp32 numerical noise instead of real policy drift.
            if b["action"].device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    la, lb, v = model(b, act[:, 0])
            else:
                la, lb, v = model(b, act[:, 0])
            lpa, lpb = F.log_softmax(masked(la.float(), ma), -1), F.log_softmax(masked(lb.float(), mb), -1)
            logp = lpa.gather(1, act[:, :1])[:, 0] + lpb.gather(1, act[:, 1:])[:, 0]
            ratio = torch.exp(logp - b["logp"])
            adv = b["adv"]
            pg = -torch.min(ratio * adv, ratio.clamp(1 - args.clip, 1 + args.clip) * adv).mean()
            vf = F.binary_cross_entropy_with_logits(v.float(), b["ret"].clamp(0, 1))
            ent = -((lpa.exp() * lpa).sum(-1) + (lpb.exp() * lpb).sum(-1)).mean()
            with torch.no_grad():
                ra, rb, _ = ref(b, act[:, 0])
                rpa, rpb = F.log_softmax(masked(ra, ma), -1), F.log_softmax(masked(rb, mb), -1)
            kl = ((lpa.exp() * (lpa - rpa)).sum(-1) + (lpb.exp() * (lpb - rpb)).sum(-1)).mean()
            loss = pg + args.vf_coef * vf - args.ent_coef * ent + args.kl_coef * kl
            if args.aux_coef > 0 and "opp" in b and model.config.get("aux"):
                # Keep the opponent-action head from decaying under RL fine-tuning (it isn't
                # otherwise touched by the PPO loss): same cross-entropy as train.py's `step`,
                # with live labels from this rollout's own battle logs (player.drain_aux_labels).
                enc = model.encode(b)
                om, ot = model.opp_logits(enc, b)
                opp = b["opp"].long()
                # Targets of the move actually used (switches have target -1 and are ignored).
                ot = ot.gather(2, opp[..., 0].clamp(0, 3)[..., None, None].expand(-1, -1, 1, 3))[:, :, 0]
                aux_n = (opp[..., 0] >= 0).sum().clamp(min=1)
                aux_ce = (F.cross_entropy(om.flatten(0, 1), opp[..., 0].flatten(), ignore_index=-1, reduction="sum")
                          + F.cross_entropy(ot.flatten(0, 1), opp[..., 1].flatten(), ignore_index=-1,
                                            reduction="sum")) / aux_n
                loss = loss + args.aux_coef * aux_ce
                stats.setdefault("aux", 0.0)
                stats["aux"] += aux_ce.item()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            for k, x in (("pg", pg), ("vf", vf), ("ent", ent), ("kl", kl),
                         ("clip", ((ratio - 1).abs() > args.clip).float().mean())):
                stats[k] += x.item()
            stats["n"] += 1
            # Stop future minibatches once the policy has moved too far from the one
            # that played the games. This minibatch's update above still counts.
            if (b["logp"] - logp).mean().item() > args.target_kl:
                stop = True
                break
    return {k: v / max(stats["n"], 1) for k, v in stats.items() if k != "n"}


def _print_nash(meta: dict) -> None:
    w = sorted(zip(meta["nash"], meta["pool"]), reverse=True)
    print("nash mix: " + ", ".join(f"{Path(p).stem} {x:.2f}" for x, p in w if x >= 0.01), flush=True)


def _meta_fold_in(meta: dict, meta_path: Path, p: str, w: float, t0: float) -> None:
    """Add `p` to the meta-game (win rate `w` vs the current policy) and refold the nash mix.
    Shared tail of the three meta-add variants below (sync ex.map, sync ProcessPoolExecutor of
    vectorized workers, async single vectorized worker)."""
    n = len(meta["pool"])
    win = np.full((n + 1, n + 1), 0.5)
    win[:n, :n] = np.array(meta["win"]) if n else win[:0, :0]
    win[n, n - 1] = w
    win[n - 1, n] = 1 - w
    meta["pool"].append(p)
    meta["win"] = win.tolist()
    meta["nash"] = nash(win).tolist()
    print(f"meta-game: added {Path(p).name} ({time.time() - t0:.0f}s, win={w:.3f})", flush=True)
    meta_path.write_text(json.dumps(meta))
    _print_nash(meta)


def _meta_add_mp(ex, worker_configs: list[dict], current: str, meta: dict, meta_path: Path, p: str, args) -> None:
    """Fold `p` into the meta-game by playing it against the current policy, with eval jobs spread
    across a pool of multi-process vectorized workers (the "Multi-process vectorized workers"
    branch of main(); mirrors the plain-`meta_add` dispatch but via `_run_vectorized_worker`)."""
    t = time.time()
    eval_jobs = [(current, "eval", p, args.job_games) for _ in range(math.ceil(args.meta_games / args.job_games))]
    chunk_size = math.ceil(len(eval_jobs) / len(worker_configs))
    job_chunks = [eval_jobs[i:i + chunk_size] for i in range(0, len(eval_jobs), chunk_size)]
    futures = [ex.submit(_run_vectorized_worker, worker_configs[i], current, chunk)
               for i, chunk in enumerate(job_chunks)]
    wins = [x for fut in futures for _, _, mine in fut.result() if mine for x in mine]
    _meta_fold_in(meta, meta_path, p, float(np.mean(wins)) if wins else 0.5, t)


async def _meta_add_vectorized(vworker, current: str, meta: dict, meta_path: Path, p: str, args) -> None:
    """Fold `p` into the meta-game via the single in-process vectorized worker (async)."""
    t = time.time()
    eval_jobs = [(current, "eval", p, args.job_games) for _ in range(math.ceil(args.meta_games / args.job_games))]
    results = await vworker.play_jobs(eval_jobs)
    wins = [x for _, _, mine in results if mine for x in mine]
    _meta_fold_in(meta, meta_path, p, float(np.mean(wins)) if wins else 0.5, t)


def save(model, path: Path, args, extra: dict) -> None:
    from ai_vgc.nn.encode import vocab_sizes

    tmp = path.with_suffix(".tmp")
    torch.save({"state": {k: v.cpu() for k, v in model.state_dict().items()}, "config": model.config,
                "sizes": vocab_sizes(), "args": {k: str(v) for k, v in vars(args).items()}} | extra, tmp)
    tmp.replace(path)  # atomic, so workers never read a half-written file


# Vectorized worker function (runs in separate process for multi-GPU)
async def _vectorized_worker_main(config_dict: dict, policy_path: str, jobs: list, result_queue: asyncio.Queue) -> None:
    """Run vectorized worker and put results in queue."""
    from ai_vgc.nn.vectorized import VectorizedWorkerConfig, run_vectorized_worker

    config = VectorizedWorkerConfig(**config_dict)
    try:
        results = await run_vectorized_worker(config, policy_path, jobs)
        await result_queue.put(results)
    except Exception as e:
        await result_queue.put(e)
    finally:
        await result_queue.put(None)  # sentinel


def _run_vectorized_worker(config_dict: dict, policy_path: str, jobs: list) -> list:
    """Sync wrapper for running vectorized worker in a process."""
    import asyncio
    torch.set_num_threads(1)
    from ai_vgc.nn.vectorized import VectorizedWorkerConfig, run_vectorized_worker

    config = VectorizedWorkerConfig(**config_dict)
    return asyncio.run(run_vectorized_worker(config, policy_path, jobs))


async def _run_iteration_vectorized(vworker, it: int, args, current, pool, meta, meta_path,
                                     mix, log_path, tb, dev, opt, vopt, ref, model) -> str | None:
    """Run one iteration with the vectorized worker."""
    t0 = time.time()
    jobs = []
    for _ in range(math.ceil(args.games / args.job_games)):
        kinds = [k for k, weight in mix.items()
                 if weight > 0 and (k not in ("pool", "nash") or pool)
                 and (k != "spec" or args.specialists)
                 and (k != "human" or args.human)]
        if not kinds:
            raise ValueError("no eligible rollout opponents for --mix")
        kind = random.choices(kinds, [mix[k] for k in kinds])[0]
        opp = (args.init if kind == "bc" else random.choice(pool) if kind == "pool"
               else random.choices(meta["pool"], meta["nash"])[0] if kind == "nash"
               else random.choice(args.specialists) if kind == "spec"
               else args.human if kind == "human" else None)
        jobs.append((str(current), kind, opp, args.job_games))

    results = await vworker.play_jobs(jobs)

    parts, results_dict = [], {}
    for kind, data, mine in results:
        if data is not None:
            parts.append(data)
        results_dict.setdefault(kind, []).extend(mine)

    t_play = time.time() - t0
    if not parts:
        print(f"iter {it}: no games finished", flush=True)
        return None

    return _process_iteration_results(parts, results_dict, it, t_play, t0, args, current, pool, meta, meta_path,
                               Path(args.out).with_name(args.out.stem + "_snapshots"), log_path, tb, dev, opt, vopt, ref, model, mix)


def _process_iteration_results(parts, results, it: int, t_play: float, t0: float, args, current, pool, meta, meta_path,
                                snap_dir, log_path, tb, dev, opt, vopt, ref, model, mix: dict) -> str | None:
    """Common post-processing for both vectorized and original modes. Returns the new snapshot's
    path when one was just saved and `nash` is in the mix, so the caller can re-run the meta-game
    eval for it (how it's dispatched differs per execution mode, so that part isn't done here)."""
    data = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    adv, ret = gae(data.pop("value"), data.pop("reward"), data.pop("last"), args.lam)
    data["adv"] = (adv - adv.mean()) / (adv.std() + 1e-8)
    data["ret"] = ret
    data.pop("preview")
    tens = {k: torch.from_numpy(v).to(dev) for k, v in data.items()}
    st = ppo_update(model, ref, tens, opt, args, value_only=it <= args.value_warmup, vopt=vopt)

    save(model, current, args, {})
    save(model, args.out, args, {"iter": it, "opt": opt.state_dict()})
    reg_path = args.out.with_name(args.out.stem + "_reg.pt")
    if args.reg_every and it > args.value_warmup and it % args.reg_every == 0:
        ref.load_state_dict(model.state_dict())
        save(model, reg_path, args, {"iter": it})
        print(f"iter {it}: KL anchor replaced by the current policy (R-NaD)", flush=True)
    new_snap = None
    if it % args.snapshot_every == 0:
        snap = snap_dir / f"iter_{it:04d}.pt"
        save(model, snap, args, {"iter": it})
        pool.append(str(snap))
        if "nash" in {k for k, v in mix.items() if v > 0}:
            new_snap = str(snap)
    wr = {k: (float(np.mean(v)), len(v)) for k, v in sorted(results.items()) if v}
    rec = {"iter": it, "steps": len(adv), "play_s": round(t_play), "total_s": round(time.time() - t0),
           "win": wr, **{k: round(v, 4) for k, v in st.items()}}
    with log_path.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    scalars(tb, "loss", st, it)
    scalars(tb, "win", {k: m for k, (m, _) in wr.items()}, it)
    scalars(tb, "time", {"play_s": t_play, "iter_s": time.time() - t0}, it)
    tb.add_scalar("steps", len(adv), it)
    tb.flush()
    wtxt = " ".join(f"{k} {m:.1%}({n})" for k, (m, n) in wr.items())
    print(f"iter {it}: {wtxt} | steps {len(adv)} | pg {st['pg']:.3f} vf {st['vf']:.3f} "
          f"ent {st['ent']:.2f} kl {st['kl']:.3f} clip {st['clip']:.2f} | "
          f"play {t_play:.0f}s total {time.time() - t0:.0f}s", flush=True)

    if dev.type == "cuda" and args.mem_limit < 1.0:
        for i in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(i)
            used_frac = 1 - free / total
            if used_frac >= args.mem_limit:
                print(f"iter {it}: GPU {i} at {used_frac:.1%} (limit {args.mem_limit:.0%}) - "
                      f"checkpoint already saved this iteration, exiting for a safe restart "
                      f"with fewer --vector-workers", flush=True)
                sys.exit(42)
    return new_snap


def main() -> None:
    from ai_vgc.nn.player import load_policy
    from ai_vgc.showdown import ensure_server

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--init", default="data/models/archive/mb-bo3-bc-v1.pt", help="imitation checkpoint to start from")
    ap.add_argument("--backend", choices=["showdown", "vgc-engine"], default="showdown",
                    help="rollout backend; vgc-engine is pure native self-play with no Showdown servers")
    ap.add_argument("--native-max-turns", type=int, default=128,
                    help="draw a native vgc-engine game after this many policy turns")
    ap.add_argument("--native-preview", action="store_true",
                    help="use policy-selected VGC bring-four/lead preview; disabled by default for legacy checkpoint throughput")
    ap.add_argument("--native-seed", type=int, default=0, help="base RNG seed for native vgc-engine workers")
    ap.add_argument("--native-slots", type=int, default=0,
                    help="live native BattleBatch slots per worker horizon (0 keeps finite --job-games rollout)")
    ap.add_argument("--native-horizon", type=int, default=0,
                    help="normal native policy waves per horizon; requires --native-slots")
    ap.add_argument("--native-compile", action="store_true",
                    help="benchmark opt-in torch.compile for static native inference batches")
    ap.add_argument("--native-opponent-teams", type=Path, default=None,
                    help="optional opponent team pool for native league games (default: --teams)")
    ap.add_argument("--out", type=Path, default=Path("data/models/archive/mb-bo3-ppo-v1.pt"))
    ap.add_argument("--resume", action="store_true", help="continue from --out if it exists")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--games", type=int, default=2048,
                    help="games per iteration (finite native mode); native persistent mode runs one horizon per worker")
    ap.add_argument("--job-games", type=int, default=32, help="games per worker job")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--native-actors", type=int, default=0,
                    help="persistent native actor processes (0: --workers on CPU, non-learner GPUs on CUDA)")
    ap.add_argument("--servers", type=int, default=8, help="local Showdown servers")
    ap.add_argument("--port", type=int, default=None, help="first server port (default: from job id)")
    ap.add_argument("--concurrency", type=int, default=8, help="battles at once per worker")
    ap.add_argument("--format", default="gen9championsvgc2026regmb")
    ap.add_argument("--teams", type=Path, default=None, help="folder of team .txt files (default: Reg M-B pool)")
    ap.add_argument("--my-team", type=Path, default=None,
                    help="specialise: the learner always plays this team (opponents still draw from --teams); "
                         "self-play then trains on the learner's side only")
    ap.add_argument("--showdown", type=Path, default=None,
                    help="Showdown checkout for the servers (e.g. pokemon-showdown-mc for Reg M-C)")
    ap.add_argument("--rating", type=float, default=1700)
    ap.add_argument("--mix", default="self:0.5,bc:0.2,pool:0.15,heuristic:0.15",
                    help="opponent mix: self, bc (the --init policy), pool (snapshots; native uses PFSP), "
                         "nash (snapshots, double-oracle weights), heuristic, spec (--specialists), "
                         "human (--human)")
    ap.add_argument("--human", default=None, metavar="MODEL",
                    help="opponent for `human` in --mix: an imitation model of human play (e.g. "
                         "mc-cts-opp-v6, trained on human closed-sheet games), never trained against itself")
    ap.add_argument("--specialists", nargs="*", default=[], metavar="MODEL::TEAM",
                    help="opponents for `spec` in --mix: each a model that plays its own team file")
    ap.add_argument("--snapshot-every", type=int, default=10)
    ap.add_argument("--ref", default=None, help="policy the KL penalty pulls towards (default: --init)")
    ap.add_argument("--pool-init", nargs="*", default=[], help="checkpoints that start the opponent pool")
    ap.add_argument("--meta-games", type=int, default=128, help="games per pair for the double-oracle table")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--ppo-epochs", type=int, default=3)
    ap.add_argument("--minibatch", type=int, default=4096)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--ent-coef", type=float, default=0.003)
    ap.add_argument("--target-kl", type=float, default=0.02,
                    help="end an iteration's update early past this KL from the playing policy")
    ap.add_argument("--kl-coef", type=float, default=0.05, help="penalty for drifting from the --init policy")
    ap.add_argument("--aux-coef", type=float, default=0.0,
                     help="cross-entropy on the opponent-action head (--init must have aux=True), labeled live "
                          "from self-play battle logs; keeps the head from decaying under PPO fine-tuning")
    ap.add_argument("--value-warmup", type=int, default=3, help="first iterations train only the value head")
    ap.add_argument("--damage-inference", action="store_true",
                    help="closed sheets: every model's damage features use estimated opposing stats (ai_vgc.bulk)")
    ap.add_argument("--set-guess", action="store_true",
                    help="closed sheets: every model reads the opponent's unrevealed sets filled in with search's guess")
    ap.add_argument("--speed-inference", action="store_true",
                    help="every model in the games reads opposing Speed narrowed from turn order (ai_vgc.speed); "
                         "for models trained with it (rl.py --speed-inference)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                    help="device for worker inference (cuda/cpu)")
    ap.add_argument("--closed-sheets", action="store_true",
                    help="every player declines open team sheets (Bo1 M-C, where they're optional): closed (CTS) games")
    ap.add_argument("--change-after-loss", action="store_true",
                    help="Bo3: the learner changes its leads after losing a game, as the ladder bot does")
    ap.add_argument("--opp-change-after-loss", action="store_true",
                    help="Bo3: training opponents (self/snap) change their leads after losing a game, mirroring "
                         "how human ladder opponents adapt, so the learner has to beat an opponent that reads it")
    ap.add_argument("--reg-every", type=int, default=0,
                    help="R-NaD: replace the KL anchor with the current policy every N iterations (0 = keep --ref)")
    ap.add_argument("--vectorized", action="store_true",
                    help="Use single-process vectorized worker with shared GPU inference (replaces --workers processes)")
    ap.add_argument("--vector-workers", type=int, default=1,
                    help="Number of vectorized worker processes (for multi-GPU); each runs all battles in one process")
    ap.add_argument("--mem-limit", type=float, default=1.0,
                    help="exit with code 42 after checkpointing if any GPU's memory usage fraction reaches this "
                         "(e.g. 0.85); the checkpoint is already fresh since this is checked after each iteration's "
                         "save, so nn_rl.sh can resubmit --resume with fewer --vector-workers. 1.0 disables the check")
    args = ap.parse_args()
    if args.backend == "vgc-engine" and args.vectorized:
        raise ValueError("--backend vgc-engine uses native worker processes; --vectorized is not supported yet")

    if args.device.startswith("cuda") and torch.cuda.is_available():
        dev = torch.device(args.device)
    else:
        dev = torch.device("cpu")
    model = load_policy(args.init, str(dev))
    if dev.type == "cpu":
        model.share_memory()  # CPU Showdown workers can share these storage pages.
    ref = load_policy(args.ref or args.init, str(dev))
    if dev.type == "cpu":
        ref.share_memory()
    for p in ref.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    vopt = torch.optim.AdamW(model.value.parameters(), lr=args.lr * 10, weight_decay=0.0)  # value warmup
    start = 0
    snap_dir = args.out.with_name(args.out.stem + "_snapshots")
    snap_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.out.with_suffix(".log.jsonl")
    tb = writer("rl", args.out, args)
    if args.resume and args.out.exists():
        ckpt = torch.load(args.out, map_location=dev, weights_only=False)
        model.load_state_dict(ckpt["state"])
        if "opt" in ckpt:
            opt.load_state_dict(ckpt["opt"])
        start = ckpt.get("iter", 0)
        print(f"resumed from {args.out} at iteration {start}", flush=True)
    reg_path = args.out.with_name(args.out.stem + "_reg.pt")
    if args.resume and args.reg_every and reg_path.exists():
        ref.load_state_dict(torch.load(reg_path, map_location=dev, weights_only=False)["state"])
        print(f"KL anchor from {reg_path}", flush=True)
    model.eval()  # no dropout: the sampled log-probs came from the eval-mode network

    current = args.out.with_name(args.out.stem + "_current.pt")
    save(model, current, args, {})
    mix = {k: float(v) for k, v in (x.split(":") for x in args.mix.split(","))}
    native_kinds = {"self", "bc", "pool"}
    if args.backend == "vgc-engine":
        if any(weight < 0 for weight in mix.values()) or not any(mix.values()):
            raise ValueError("--mix needs at least one positive native opponent weight")
        unsupported = {kind for kind, weight in mix.items() if weight > 0 and kind not in native_kinds}
        if unsupported:
            raise ValueError(
                "--backend vgc-engine supports only self, bc, and pool opponents; "
                f"unsupported: {', '.join(sorted(unsupported))}"
            )
        if bool(args.native_slots) != bool(args.native_horizon):
            raise ValueError("--native-slots and --native-horizon must be supplied together")
        if args.my_team and not args.native_slots:
            raise ValueError("native --my-team league collection requires --native-slots and --native-horizon")
        if args.native_slots < 0 or args.native_horizon < 0:
            raise ValueError("--native-slots and --native-horizon must be non-negative")
        league_weight = sum(mix.get(kind, 0.0) for kind in ("bc", "pool"))
        if league_weight and not args.my_team:
            raise ValueError("native bc/pool league opponents require --my-team")
        if args.aux_coef:
            raise ValueError("native vgc-engine rollout does not provide --aux-coef labels")
        if args.native_actors < 0:
            raise ValueError("--native-actors must be non-negative")
    pool = [*args.pool_init, *sorted(str(p) for p in snap_dir.glob("iter_*.pt"))]
    if (args.backend == "vgc-engine" and mix.get("pool", 0.0) > 0 and not pool
            and not (mix.get("self", 0.0) > 0 or mix.get("bc", 0.0) > 0)):
        raise ValueError("native --mix pool needs --pool-init or a positive self/bc bootstrap weight")
    meta_path = args.out.with_name(args.out.stem + "_meta.json")
    meta = json.loads(meta_path.read_text()) if args.resume and meta_path.exists() else {"pool": [], "win": []}
    # Native snapshots use empirical PFSP rather than the Showdown-dependent
    # double-oracle table. Keep it in a separate durable file so native and
    # Showdown resume paths can share a checkpoint directory safely.
    native_pfsp_path = args.out.with_name(args.out.stem + "_native_pfsp.json")
    if args.backend == "vgc-engine" and args.resume and native_pfsp_path.exists():
        native_pfsp = json.loads(native_pfsp_path.read_text())
    else:
        native_pfsp = {}
    use_nash = mix.get("nash", 0.0) > 0

    base = args.port or 20000 + (int(os.environ.get("SLURM_JOB_ID", os.getpid())) % 400) * 25
    ports = [base + i for i in range(args.servers)]
    procs = [ensure_server(p, args.showdown) for p in ports] if args.backend == "showdown" else []
    if args.backend == "showdown":
        print(f"{args.servers} Showdown servers on ports {ports[0]}-{ports[-1]}; learner on {dev}", flush=True)
    else:
        native_mode = "league" if set(mix) - {"self"} or args.my_team else "self-play"
        if dev.type == "cuda":
            actor_count = args.native_actors or max(torch.cuda.device_count() - 1, 1)
        else:
            actor_count = args.native_actors or args.workers
        horizon_note = (f"; {actor_count} actor horizon(s) of {args.native_slots} slots × "
                        f"{args.native_horizon} waves" if args.native_slots else "")
        print(f"native vgc-engine {native_mode}; no Showdown servers; learner on {dev}{horizon_note}", flush=True)

    # Prepare common worker config
    worker_config = dict(
        ports=ports,
        fmt=args.format,
        concurrency=args.concurrency,
        rating=args.rating,
        teams_dir=str(args.teams) if args.teams else None,
        my_team=str(args.my_team) if args.my_team else None,
        change_after_loss=args.change_after_loss,
        opp_change_after_loss=args.opp_change_after_loss,
        closed_sheets=args.closed_sheets,
        speed_inference=args.speed_inference,
        set_guess=args.set_guess,
        damage_inference=args.damage_inference,
        device=str(dev),
        aux_coef=args.aux_coef,
        native_max_turns=args.native_max_turns,
        native_seed=args.native_seed,
    )

    # Vectorized mode: single-process asyncio with shared GPU inference
    if args.vectorized:
        print(f"Vectorized mode: {args.vector_workers} worker process(es), each running {args.servers * args.concurrency} concurrent battles", flush=True)

        if args.vector_workers == 1:
            # Single vectorized worker in main process (asyncio)
            import asyncio
            from ai_vgc.nn.vectorized import VectorizedWorkerConfig, VectorizedWorker

            vconfig = VectorizedWorkerConfig(**worker_config)
            vworker = VectorizedWorker(vconfig)

            async def run_vectorized():
                await vworker.initialize(str(current), {})
                try:
                    if use_nash:
                        for p in pool:
                            if p not in meta["pool"]:
                                await _meta_add_vectorized(vworker, str(current), meta, meta_path, p, args)

                    for it in range(start + 1, args.iters + 1):
                        new_snap = await _run_iteration_vectorized(vworker, it, args, current, pool, meta, meta_path,
                                                                    mix, log_path, tb, dev, opt, vopt, ref, model)
                        if new_snap is not None:
                            await _meta_add_vectorized(vworker, str(current), meta, meta_path, new_snap, args)
                finally:
                    await vworker.close()

            asyncio.run(run_vectorized())

        else:
            # Multi-process vectorized workers (one per GPU)
            ctx = multiprocessing.get_context("spawn")
            # Partition ports among vector workers to avoid conflicts
            ports_per_worker = math.ceil(args.servers / args.vector_workers)
            worker_port_chunks = [ports[i:i + ports_per_worker] for i in range(0, len(ports), ports_per_worker)]
            while len(worker_port_chunks) < args.vector_workers:
                worker_port_chunks.append([ports[-1]])
            num_gpus = torch.cuda.device_count() if args.device == "cuda" else 1
            worker_configs = []
            for i, w_ports in enumerate(worker_port_chunks):
                wc = dict(worker_config)
                wc["ports"] = w_ports
                if num_gpus > 1:
                    wc["device"] = f"cuda:{i % num_gpus}"
                worker_configs.append(wc)

            with ProcessPoolExecutor(args.vector_workers, mp_context=ctx) as ex:
                if use_nash:
                    for p in pool:
                        if p not in meta["pool"]:
                            _meta_add_mp(ex, worker_configs, str(current), meta, meta_path, p, args)

                for it in range(start + 1, args.iters + 1):
                    t0 = time.time()
                    jobs = []
                    for _ in range(math.ceil(args.games / args.job_games)):
                        kinds = [k for k, weight in mix.items()
                                 if weight > 0 and (k not in ("pool", "nash") or pool)
                                 and (k != "spec" or args.specialists)
                                 and (k != "human" or args.human)]
                        if not kinds:
                            raise ValueError("no eligible rollout opponents for --mix")
                        kind = random.choices(kinds, [mix[k] for k in kinds])[0]
                        opp = (args.init if kind == "bc" else random.choice(pool) if kind == "pool"
                               else random.choices(meta["pool"], meta["nash"])[0] if kind == "nash"
                               else random.choice(args.specialists) if kind == "spec"
                               else args.human if kind == "human" else None)
                        jobs.append((str(current), kind, opp, args.job_games))

                    # Split jobs across vector workers
                    chunk_size = math.ceil(len(jobs) / args.vector_workers)
                    job_chunks = [jobs[i:i + chunk_size] for i in range(0, len(jobs), chunk_size)]
                    futures = [ex.submit(_run_vectorized_worker, worker_configs[i], str(current), chunk)
                               for i, chunk in enumerate(job_chunks)]

                    parts, results = [], {}
                    for fut in futures:
                        for kind, data, mine in fut.result():
                            if data is not None:
                                parts.append(data)
                            results.setdefault(kind, []).extend(mine)

                    t_play = time.time() - t0
                    if not parts:
                        print(f"iter {it}: no games finished", flush=True)
                        continue

                    new_snap = _process_iteration_results(parts, results, it, t_play, t0, args, current, pool, meta, meta_path,
                                               snap_dir, log_path, tb, dev, opt, vopt, ref, model, mix)
                    if new_snap is not None:
                        _meta_add_mp(ex, worker_configs, str(current), meta, meta_path, new_snap, args)

    # Original ProcessPoolExecutor mode (backward compatible)
    else:
        # Native persistent actors own their BattleBatch and model cache for the
        # complete training run. A horizon returns compact PPO rows; its slots
        # are immediately refilled internally instead of rebuilding a tail
        # batch or reloading a model for every job.
        if args.backend == "vgc-engine" and args.native_slots:
            actor_pool = NativeActorPool(_native_actor_configs(args, ports, dev))
            try:
                for it in range(start + 1, args.iters + 1):
                    t0 = time.time()
                    jobs = []
                    for _ in actor_pool.processes:
                        kinds = [k for k, weight in mix.items()
                                 if weight > 0 and (k != "pool" or pool)]
                        if not kinds:
                            raise ValueError("no eligible native rollout opponents for --mix")
                        kind = random.choices(kinds, [mix[k] for k in kinds])[0]
                        opponent = args.init if kind == "bc" else None
                        result_kind = kind
                        if kind == "pool":
                            opponent = _native_snapshot_choice(pool, native_pfsp)
                            # Preserve the snapshot identity through the actor
                            # result so each terminal learner reward updates
                            # the correct PFSP arm rather than a pool aggregate.
                            result_kind = f"pool:{opponent}"
                        jobs.append((str(current), result_kind, opponent, args.job_games))
                    parts, results = [], {}
                    for kind, data, mine in actor_pool.collect(jobs):
                        if data is not None:
                            parts.append(data)
                        results.setdefault(kind, []).extend(mine)
                    _native_record_snapshot_results(results, pool, native_pfsp)
                    t_play = time.time() - t0
                    if not parts:
                        print(f"iter {it}: no games finished", flush=True)
                        continue
                    new_snap = _process_iteration_results(
                        parts, results, it, t_play, t0, args, current, pool, meta, meta_path,
                        snap_dir, log_path, tb, dev, opt, vopt, ref, model, mix,
                    )
                    if new_snap is not None:
                        raise RuntimeError("native nash snapshot promotion is not supported")
                    # Snapshots saved by `_process_iteration_results` are
                    # immediately eligible as frozen PFSP opponents. Retire
                    # records for snapshots absent after a resumed run.
                    pool[:] = [*args.pool_init, *sorted(str(p) for p in snap_dir.glob("iter_*.pt"))]
                    native_pfsp = {path: stats for path, stats in native_pfsp.items() if path in pool}
                    native_pfsp_path.write_text(json.dumps(native_pfsp, sort_keys=True))
            finally:
                actor_pool.close()
            return

        ctx = multiprocessing.get_context("spawn")
        try:
            with ProcessPoolExecutor(args.workers, mp_context=ctx, initializer=_init,
                                     initargs=(ports, args.format, args.concurrency, args.rating,
                                               str(args.teams) if args.teams else None,
                                               str(args.my_team) if args.my_team else None,
                                               args.change_after_loss, args.opp_change_after_loss,
                                               args.closed_sheets, args.speed_inference,
                                               args.set_guess, args.damage_inference, str(dev),
                                               args.aux_coef, args.native_max_turns, args.native_seed,
                                               args.native_preview, args.native_slots, args.native_horizon,
                                               args.native_compile,
                                               str(args.my_team) if args.my_team else None,
                                               str(args.native_opponent_teams) if args.native_opponent_teams else None)) as ex:
                if use_nash:
                    for p in pool:
                        if p not in meta["pool"]:
                            t = time.time()
                            meta_add(ex, meta, p, args.meta_games, args.job_games)
                            print(f"meta-game: added {Path(p).name} ({time.time() - t:.0f}s)", flush=True)
                    meta_path.write_text(json.dumps(meta))
                    _print_nash(meta)
                for it in range(start + 1, args.iters + 1):
                    t0 = time.time()
                    jobs = []
                    job_count = math.ceil(args.games / args.job_games)
                    if args.backend == "vgc-engine" and args.native_slots:
                        # A native job is an actor horizon, not a game quota.
                        # Submit one horizon per worker so --workers maps to a
                        # bounded synchronous actor batch every PPO iteration.
                        job_count = args.workers
                    for _ in range(job_count):
                        kinds = [k for k, weight in mix.items()
                                 if weight > 0 and (k not in ("pool", "nash") or pool)
                                 and (k != "spec" or args.specialists)
                                 and (k != "human" or args.human)]
                        if not kinds:
                            raise ValueError("no eligible rollout opponents for --mix")
                        kind = random.choices(kinds, [mix[k] for k in kinds])[0]
                        opp = (args.init if kind == "bc" else random.choice(pool) if kind == "pool"
                               else random.choices(meta["pool"], meta["nash"])[0] if kind == "nash"
                               else random.choice(args.specialists) if kind == "spec"
                               else args.human if kind == "human" else None)
                        jobs.append((str(current), kind, opp, args.job_games))
                    parts, results = [], {}
                    rollout = _play_native if args.backend == "vgc-engine" else _play
                    for kind, data, mine in ex.map(rollout, jobs):
                        if data is not None:
                            parts.append(data)
                        results.setdefault(kind, []).extend(mine)
                    t_play = time.time() - t0
                    if not parts:
                        print(f"iter {it}: no games finished", flush=True)
                        continue

                    new_snap = _process_iteration_results(parts, results, it, t_play, t0, args, current, pool, meta, meta_path,
                                               snap_dir, log_path, tb, dev, opt, vopt, ref, model, mix)
                    if new_snap is not None:
                        t = time.time()
                        meta_add(ex, meta, new_snap, args.meta_games, args.job_games)
                        print(f"meta-game: added {Path(new_snap).name} ({time.time() - t:.0f}s)", flush=True)
                        meta_path.write_text(json.dumps(meta))
                        _print_nash(meta)
        finally:
            for p in procs:
                if p:
                    p.terminate()


if __name__ == "__main__":
    main()
