"""Vectorized RL worker: runs many battles in one process with shared GPU inference.

Replaces ProcessPoolExecutor (32 processes) with a single asyncio process that:
- Maintains N concurrent battles across M Showdown servers
- Shares one _PipelineBatcher for all inference (batched GPU)
- Uses asyncio for WebSocket I/O (no process overhead)
"""

from __future__ import annotations

import asyncio
import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from poke_env import ServerConfiguration
from poke_env.player import SimpleHeuristicsPlayer
from poke_env.teambuilder import ConstantTeambuilder

from ai_vgc.nn.player import NNPlayer, load_policy, _PipelineBatcher, batcher
from ai_vgc.showdown import account
from ai_vgc.teams import RandomPoolTeambuilder

if TYPE_CHECKING:
    from ai_vgc.nn.model import Policy

KEYS = ["tok_cat", "tok_num", "mv_cat", "mv_dyn", "dmg", "glob", "act_tok", "mask", "ser_tok", "ser_mv", "ser_glob"]


@dataclass
class BattleRecord:
    """Collected data from one battle (mirrors NNPlayer.episodes)."""
    steps: list[dict] = field(default_factory=list)
    reward: float = 0.0
    finished: bool = False


@dataclass
class VectorizedWorkerConfig:
    """Configuration for the vectorized worker."""
    ports: list[int]
    fmt: str
    concurrency: int  # battles per server
    rating: float
    teams_dir: str | None = None
    my_team: str | None = None
    opp_team: str | None = None
    change_after_loss: bool = False
    opp_change_after_loss: bool = False
    closed_sheets: bool = False
    speed_inference: bool = False
    set_guess: bool = False
    damage_inference: bool = False
    device: str = "cuda"
    aux_coef: float = 0.0


class VectorizedWorker:
    """Single-process worker running all battles with shared batched inference."""

    def __init__(self, config: VectorizedWorkerConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.players: dict[str, NNPlayer] = {}
        self.battles: dict[int, BattleRecord] = {}
        self.battle_to_player: dict[int, str] = {}
        self.battle_to_opponent: dict[int, str] = {}
        self.episode_results: list[tuple[list[dict], float]] = []
        self._running = False
        self._tasks: set[asyncio.Task] = set()
        
        # Shared batched inference pipeline
        self._shared_model: Policy | None = None

    async def initialize(self, policy_path: str, opponent_policies: dict[str, str]) -> None:
        """Create all players (connections established lazily on first battle)."""
        import time
        t0 = time.time()
        print(f"[Vectorized] Loading model from {policy_path}...", flush=True)
        # Load shared model
        self._shared_model = load_policy(policy_path, str(self.device))
        self._shared_model.share_memory()
        print(f"[Vectorized] Model loaded in {time.time() - t0:.1f}s", flush=True)
        
        # Create players for each role (they'll auto-share batcher via model instance)
        print(f"[Vectorized] Creating {len(self.config.ports)} servers × {self.config.concurrency} concurrency players...", flush=True)
        t1 = time.time()
        self.players = await self._create_players(policy_path, opponent_policies)
        print(f"[Vectorized] Players created in {time.time() - t1:.1f}s (total {time.time() - t0:.1f}s)", flush=True)
        self._running = True

    async def _create_players(self, policy_path: str, opponent_policies: dict[str, str]) -> dict[str, NNPlayer]:
        """Share only learner/self weights; frozen and evaluation models are isolated.

        Players that battle each other MUST be on the same Showdown server.
        Pairings:
        - "me" vs "self" (self-play)
        - "me" vs "snap" (bc, pool, nash, spec, human)
        - "me" vs "evalB" (eval)
        - "me" vs "heuristic" (heuristic)
        """
        import copy
        players = {}
        
        def make_teambuilder():
            if self.config.teams_dir:
                return RandomPoolTeambuilder(Path(self.config.teams_dir))
            return RandomPoolTeambuilder()
        
        my_teambuilder = ConstantTeambuilder(Path(self.config.my_team).read_text()) if self.config.my_team else make_teambuilder()
        
        common = dict(
            battle_format=self.config.fmt,
            max_concurrent_battles=self.config.concurrency,
            accept_open_team_sheet=not self.config.closed_sheets,
            log_level=40,
        )
        
        # All players that battle each other must be on the SAME Showdown server.
        # Pairings: me vs self, me vs snap, me vs evalB, me vs heuristic, evalA vs evalB
        # Minimum 1 server needed, but 2+ allows more concurrent connections
        n_servers = len(self.config.ports)
        if n_servers < 1:
            raise ValueError(f"Need at least 1 server for vectorized mode, got {n_servers}")
        
        # Put ALL players on server 0 (first port) so they can all battle each other
        port_main = self.config.ports[0]
        server_main = ServerConfiguration(
            f"ws://localhost:{port_main}/showdown/websocket",
            "https://play.pokemonshowdown.com/action.php?"
        )
        
        # "me" - the learner, records episodes
        players["me"] = NNPlayer(
            self._shared_model,
            self.config.rating,
            record=True,
            account_configuration=account("rl"),
            team=my_teambuilder,
            server_configuration=server_main,
            **common
        )
        players["me"].speed_inference = self.config.speed_inference
        players["me"].set_guess = (self.config.teams_dir, self.config.fmt) if self.config.set_guess and self.config.teams_dir else None
        players["me"].damage_inference = self.config.damage_inference
        players["me"].change_after_loss = self.config.change_after_loss
        
        # "self" - self-play opponent (also records), on SAME server as "me"
        players["self"] = NNPlayer(
            self._shared_model,
            self.config.rating,
            record=True,
            account_configuration=account("self"),
            team=make_teambuilder(),
            server_configuration=server_main,
            **common
        )
        players["self"].speed_inference = self.config.speed_inference
        players["self"].set_guess = (self.config.teams_dir, self.config.fmt) if self.config.set_guess and self.config.teams_dir else None
        players["self"].damage_inference = self.config.damage_inference
        # Mirror a human opponent who reads game N and leads differently in N+1 after losing it,
        # instead of only our own bot varying (that taught the trainee nothing about facing one).
        players["self"].change_after_loss = self.config.opp_change_after_loss
        if self.config.aux_coef > 0:  # label self-play episodes so the aux heads don't decay (rl.py --aux-coef)
            players["me"].aux_labels = True
            players["self"].aux_labels = True

        # "snap" - snapshot/pool opponent, on SAME server as "me"
        players["snap"] = NNPlayer(
            copy.deepcopy(self._shared_model),
            self.config.rating,
            record=False,
            account_configuration=account("snap"),
            team=make_teambuilder(),
            server_configuration=server_main,
            **common
        )
        players["snap"].speed_inference = self.config.speed_inference
        players["snap"].set_guess = (self.config.teams_dir, self.config.fmt) if self.config.set_guess and self.config.teams_dir else None
        players["snap"].damage_inference = self.config.damage_inference
        players["snap"].change_after_loss = self.config.opp_change_after_loss

        # "evalA" - eval opponent A, on SAME server (for evalA vs evalB)
        players["evalA"] = NNPlayer(
            copy.deepcopy(self._shared_model),
            self.config.rating,
            record=False,
            account_configuration=account("evA"),
            team=make_teambuilder(),
            server_configuration=server_main,
            **common
        )
        players["evalA"].speed_inference = self.config.speed_inference
        players["evalA"].set_guess = (self.config.teams_dir, self.config.fmt) if self.config.set_guess and self.config.teams_dir else None
        players["evalA"].damage_inference = self.config.damage_inference
        
        # "evalB" - eval opponent B, on SAME server (for evalA vs evalB and me vs evalB)
        opp_teambuilder = ConstantTeambuilder(Path(self.config.opp_team).read_text()) if self.config.opp_team else make_teambuilder()
        players["evalB"] = NNPlayer(
            copy.deepcopy(self._shared_model),
            self.config.rating,
            record=False,
            account_configuration=account("evB"),
            team=opp_teambuilder,
            server_configuration=server_main,
            **common
        )
        players["evalB"].speed_inference = self.config.speed_inference
        players["evalB"].set_guess = (self.config.teams_dir, self.config.fmt) if self.config.set_guess and self.config.teams_dir else None
        players["evalB"].damage_inference = self.config.damage_inference
        
        # Heuristic opponent on same server
        players["heuristic"] = SimpleHeuristicsPlayer(
            account_configuration=account("heur"),
            team=make_teambuilder(),
            server_configuration=server_main,
            **common
        )
        
        from ai_vgc.nn.player import configure_inference
        for key in ("me", "self", "snap", "evalA", "evalB"):
            configure_inference(players[key], self.config.teams_dir, self.config.fmt)
        return players

    async def play_jobs(self, jobs: list[tuple[str, str, str | None, int]]) -> list[tuple[str, dict | None, list[float]]]:
        """Execute a batch of jobs sequentially within this worker.
        
        Each job: (policy_path, kind, opp_path, n_games)
        Returns: list of (kind, flattened_data, rewards)
        """
        results = []
        
        for policy_path, kind, opp_path, n in jobs:
            if self.config.fmt.endswith("bo3"):
                n = max(1, round(n / 2.4))
            
            me = self.players["evalA" if kind == "eval" else "me"]
            opp = self._get_opponent_for_kind(kind, opp_path, policy_path)
            
            # Load policy if the file has changed since we last loaded it (path alone isn't
            # enough: the trainer overwrites the same path every iteration).
            mtime = os.path.getmtime(policy_path)
            if mtime != getattr(me, "_policy_mtime", None):
                from ai_vgc.nn.player import load_player_checkpoint, configure_inference
                load_player_checkpoint(me, policy_path, self.config.teams_dir, self.config.fmt)
                configure_inference(self.players["self"], self.config.teams_dir, self.config.fmt)
                me._policy_mtime = mtime
            
            me.episodes.clear()
            if kind == "self":
                opp.episodes.clear()
            
            # Longer timeout for first jobs (connection + compilation warmup)
            base_timeout = 180 + (25 if self.config.fmt.endswith("bo3") else 30) * n
            # Add 120s warmup for first few jobs per player
            warmup_extra = 120 if not getattr(me, "_warmed_up", False) else 0
            # Extra timeout for eval jobs (they play meta-games)
            eval_extra = 300 if kind == "eval" else 0
            limit = base_timeout + warmup_extra + eval_extra
            me._warmed_up = True
            
            try:
                await asyncio.wait_for(me.battle_against(opp, n_battles=n), timeout=limit)
            except asyncio.TimeoutError:
                print(f"Vectorized worker: {kind} job timed out (limit={limit}s)", flush=True)
                results.append((kind, None, []))
                continue

            if me.aux_labels:
                me.drain_aux_labels()
            if kind == "self" and opp.aux_labels:
                opp.drain_aux_labels()

            mine = ([1.0 if b.won else 0.0 if b.lost else .5 for b in me.battles.values() if b.finished]
                    if kind == "eval" else [r for _, r in me.episodes])
            episodes = me.episodes + (opp.episodes if kind == "self" and not self.config.my_team else [])
            
            if episodes and kind != "eval":
                data = self._flatten(episodes)
            else:
                data = None
            
            me.episodes.clear()
            if kind == "self":
                opp.episodes.clear()
            me.reset_battles()
            opp.reset_battles()
            
            results.append((kind, data, mine))
        
        return results

    def _get_opponent_for_kind(self, kind: str, opp_path: str | None, current_policy: str):
        """Get opponent player for job kind."""
        from ai_vgc.nn.player import load_player_checkpoint
        if kind == "eval":
            a, b = self.players["evalA"], self.players["evalB"]
            if opp_path:
                load_player_checkpoint(a, current_policy, self.config.teams_dir, self.config.fmt)
                load_player_checkpoint(b, opp_path, self.config.teams_dir, self.config.fmt)
            return b
        elif kind == "self":
            return self.players["self"]
        elif kind in ("bc", "snap", "pool", "nash", "spec", "human"):
            if opp_path:
                load_player_checkpoint(self.players["snap"], opp_path, self.config.teams_dir, self.config.fmt)
            return self.players["snap"]
        else:
            return self.players["heuristic"]

    def _flatten(self, episodes: list[tuple[list[dict], float]]) -> dict[str, np.ndarray]:
        """Flatten episodes to batched arrays (same as _flatten in rl.py)."""
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

    async def close(self) -> None:
        """Clean shutdown."""
        self._running = False
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        for player in self.players.values():
            if hasattr(player, 'close'):
                await player.close()


# Async entry point for running as a standalone worker
async def run_vectorized_worker(config: VectorizedWorkerConfig, policy_path: str, jobs: list) -> list:
    """Run jobs in a vectorized worker and return results."""
    worker = VectorizedWorker(config)
    try:
        await worker.initialize(policy_path, {})
        return await worker.play_jobs(jobs)
    finally:
        await worker.close()


# Sync wrapper for use with ProcessPoolExecutor (transitional)
def vectorized_play_job(args: tuple) -> tuple:
    """Sync wrapper: (config_dict, policy_path, job) -> (kind, data, rewards)"""
    config_dict, policy_path, job = args
    config = VectorizedWorkerConfig(**config_dict)
    return asyncio.run(run_vectorized_worker(config, policy_path, [job]))[0]


if __name__ == "__main__":
    # Quick self-test
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    async def test():
        config = VectorizedWorkerConfig(
            ports=[8001],
            fmt="gen9championsvgc2026regmcbo3",
            concurrency=2,
            rating=1700,
            device="cpu"
        )
        # This would need a running Showdown server
        print("VectorizedWorker module loaded OK")
    
    asyncio.run(test())
