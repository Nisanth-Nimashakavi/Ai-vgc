"""Batched Neural Self-Play & Prioritized League RL for 17-Token Transformer.

Breakthrough Self-Play Architecture:
1. Neural vs Neural: Opponent actions are driven by a neural policy (not random sampling).
2. Dynamic League Pool (PFSP): Opponent alternates between mirror self-play (current weights),
   historical milestone checkpoints, and the imitation anchor to prevent policy cycling.
3. 100% Closed Team Sheet (CTS): Both Learner and Opponent observe hidden enemy item/ability features.
4. Annealed KL Anchor: Decays from 0.03 to 0.005 so the policy breaks past the imitation plateau.
5. High-Throughput Batched DMA: Uses pinned host memory and PyTorch AMP on GPU Tensor Cores.
"""

from __future__ import annotations

import argparse
import copy
import glob
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

import vgc_engine
from ai_vgc.nn.model17 import Transformer17


def load_pool_teams(pool_dir: str | Path) -> list[list[dict[str, Any]]]:
    """Load JSON or Showdown TXT formatted teams from pool directory."""
    path = Path(pool_dir)
    teams = []
    json_files = sorted(path.glob("*.json"))
    for f in json_files:
        try:
            data = json.loads(f.read_text("utf-8", errors="ignore"))
            mons = data.get("pokemon", data.get("team", data)) if isinstance(data, dict) else data
            if isinstance(mons, list) and len(mons) >= 4:
                teams.append(mons)
        except Exception:
            continue
    if not teams:
        from poke_env.teambuilder import Teambuilder
        txt_files = sorted(path.glob("*.txt"))
        for f in txt_files:
            try:
                parsed = Teambuilder.parse_showdown_team(f.read_text("utf-8", errors="ignore"))
                mons = [{
                    "species": m.species or m.nickname,
                    "ability": m.ability or "",
                    "item": m.item or "",
                    "nature": m.nature or "Serious",
                    "moves": m.moves,
                    "evs": m.evs or {},
                    "ivs": m.ivs or {},
                } for m in parsed]
                if len(mons) >= 4:
                    teams.append(mons)
            except Exception:
                continue
    return teams


class FastBattleSlot:
    """Manages a single active battle slot with zero-copy prebuilt team JSONs."""

    def __init__(self, team_jsons: list[str]):
        self.team_jsons = team_jsons
        self.battle: vgc_engine.Battle | None = None
        self.learner_side: int = 0
        self.opp_side: int = 1
        self.turn: int = 0
        self.done: bool = True
        self.reset()

    def reset(self):
        t1 = random.choice(self.team_jsons)
        t2 = random.choice(self.team_jsons)
        self.learner_side = random.choice([0, 1])
        self.opp_side = 1 - self.learner_side

        self.battle = vgc_engine.Battle.from_teams(
            t1, t2, format="gen9championsvgc2026regmc", stat_points=True
        )
        self.turn = 1
        self.done = False


def _replacement_choices(battle: Any, side: int) -> list[tuple]:
    used: set[int] = set()
    choices: list[tuple] = []
    for slot in range(2):
        legal = battle.legal_choices(side, slot)
        switches = [c for c in legal if c[0] == "switch" and c[2] not in used]
        choice = random.choice(switches) if switches else next((c for c in legal if c[0] == "pass"), legal[0])
        if choice[0] == "switch":
            used.add(choice[2])
        choices.append(choice)
    return choices


@torch.no_grad()
def evaluate_holdout(
    model: Transformer17,
    holdout_teams: list[list[dict]],
    num_games: int = 20,
    device: torch.device = torch.device("cpu"),
    cts: bool = True,
) -> float:
    """Evaluate win rate against holdout tournament teams."""
    if not holdout_teams:
        return 0.5
    model.eval()
    wins = 0
    total = 0

    eval_model = model.module if isinstance(model, DDP) else model

    for _ in range(num_games):
        t1 = random.choice(holdout_teams)
        t2 = random.choice(holdout_teams)
        t1_json = json.dumps(t1)
        t2_json = json.dumps(t2)

        preview_b = vgc_engine.Battle.from_teams(t1_json, t2_json, format="gen9championsvgc2026regmc", stat_points=True)
        
        enc0 = preview_b.encode_17(0)
        c0 = torch.from_numpy(np.frombuffer(enc0["tok_cat"], dtype=np.uint32).reshape(1, 17, 4).copy()).long().to(device)
        n0 = torch.from_numpy(np.frombuffer(enc0["tok_num"], dtype=np.float32).reshape(1, 17, 16).copy()).float().to(device)
        if cts:
            c0[:, 4:8, 1:3] = 0
        p0 = int(eval_model(c0, n0)["preview_plan"].argmax(dim=-1).item())
        picks0 = list(vgc_engine.plan_to_picks(p0))
        picks1 = random.sample(range(len(t2)), min(4, len(t2)))

        chosen_1 = [t1[i] for i in picks0]
        chosen_2 = [t2[i] for i in picks1]

        b = vgc_engine.Battle.from_teams(json.dumps(chosen_1), json.dumps(chosen_2), format="gen9championsvgc2026regmc", stat_points=True)
        
        for _turn in range(30):
            enc = b.encode_17(0)
            c = torch.from_numpy(np.frombuffer(enc["tok_cat"], dtype=np.uint32).reshape(1, 17, 4).copy()).long().to(device)
            n = torch.from_numpy(np.frombuffer(enc["tok_num"], dtype=np.float32).reshape(1, 17, 16).copy()).float().to(device)
            if cts:
                c[:, 4:8, 1:3] = 0
            out = eval_model(c, n)

            m0 = torch.tensor(b.legal_action_mask_17(0, 0), dtype=torch.bool, device=device)
            m1 = torch.tensor(b.legal_action_mask_17(0, 1), dtype=torch.bool, device=device)

            la = out["policy_a"][0]
            la[~m0] = -1e9
            act_a = int(la.argmax().item())

            out_b = eval_model(c, n, action_a=torch.tensor([act_a], device=device))
            lb = out_b["policy_b"][0]
            if act_a in (16, 17) and act_a < len(m1):
                m1[act_a] = False
            if m1.sum() == 0:
                m1[18] = True
            lb[~m1] = -1e9
            act_b = int(lb.argmax().item())

            ch0, ch1 = b.actions_17_to_choices(0, act_a, act_b)

            opp_leg0 = b.legal_choices(1, 0)
            opp_leg1 = b.legal_choices(1, 1)
            opp_ch0 = random.choice(opp_leg0) if opp_leg0 else ("pass", 0)
            opp_ch1 = random.choice(opp_leg1) if opp_leg1 else ("pass", 1)

            res = b.step_move([ch0, ch1], [opp_ch0, opp_ch1])
            while b.needs_replacements():
                r0 = _replacement_choices(b, 0)
                r1 = _replacement_choices(b, 1)
                res = b.step_move(r0, r1)
                if res != "continue":
                    break

            if res == "p1_win":
                wins += 1
                total += 1
                break
            elif res in ("p2_win", "tie"):
                total += 1
                break
        else:
            total += 1

    return wins / max(1, total)


def load_checkpoint_weights(model: Transformer17, ckpt_path: str | Path, device: torch.device):
    """Load weights into Transformer17, stripping module. prefix if needed."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
    clean = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
    model.load_state_dict(clean)
    return ckpt


def main():
    parser = argparse.ArgumentParser(description="Batched Neural Self-Play & League RL for 17-Token Transformer")
    parser.add_argument("--model", default="data/models/rl17_cts_production/model17_rl_135004k.pt", help="Initial checkpoint")
    parser.add_argument("--ref-model", default="data/models/model17_imitation_1m.pt", help="Anchor imitation checkpoint for KL")
    parser.add_argument("--league-dir", default="data/models/rl17_cts_production", help="Directory of historical milestone checkpoints")
    parser.add_argument("--train-pool", default="data/teams/pool_5600/train", help="Training teams pool directory")
    parser.add_argument("--holdout-pool", default="data/teams/pool_5600/holdout", help="Holdout evaluation pool directory")
    parser.add_argument("--out-dir", default="data/models/rl17_selfplay", help="Output directory for checkpoints")
    parser.add_argument("--timesteps", type=int, default=30_000_000, help="Total RL timesteps")
    parser.add_argument("--num-envs", type=int, default=256, help="Parallel battle slots per GPU")
    parser.add_argument("--rollout-len", type=int, default=128, help="Rollout horizon per update")
    parser.add_argument("--batch-size", type=int, default=1024, help="PPO mini-batch size")
    parser.add_argument("--ppo-epochs", type=int, default=2, help="PPO SGD epochs per rollout")
    parser.add_argument("--lr", type=float, default=1.5e-5, help="Learning rate")
    parser.add_argument("--kl-init", type=float, default=0.03, help="Initial KL penalty coefficient against imitation anchor")
    parser.add_argument("--kl-min", type=float, default=0.005, help="Minimum annealed KL penalty coefficient")
    parser.add_argument("--opp-update-interval", type=int, default=5, help="Updates between opponent league/snapshot refreshes")
    parser.add_argument("--eval-games", type=int, default=20, help="Number of games for holdout evaluation")
    parser.add_argument("--eval-interval", type=int, default=250_000, help="Timesteps between holdout evaluations")
    parser.add_argument("--save-interval", type=int, default=1_000_000, help="Timesteps between checkpoint saves")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda parameter")
    parser.add_argument("--clip-eps", type=float, default=0.2, help="PPO clip parameter")
    parser.add_argument("--vf-coef", type=float, default=0.5, help="Value loss coefficient")
    parser.add_argument("--ent-coef", type=float, default=0.01, help="Entropy bonus coefficient")
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Max gradient norm")
    parser.add_argument("--cts", action="store_true", default=True, help="100% Closed Team Sheet (opponent item/ability hidden)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # DDP Distributed Setup
    is_distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if is_distributed:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group("nccl")
        device = torch.device(f"cuda:{local_rank}")
    else:
        rank = 0
        local_rank = 0
        world_size = 1
        device = torch.device(args.device)

    out_dir = Path(args.out_dir)
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        print("=" * 75)
        print("BATCHED NEURAL SELF-PLAY & LEAGUE RL (17-TOKEN TRANSFORMER)")
        print("=" * 75)
        print(f"Device:            {device} (Total GPUs: {world_size})")
        print(f"Opponent Policy:   Neural Network (Mirror + Historical League Snapshots)")
        print(f"Sheet Format:      100% Closed Team Sheet (CTS)")
        print(f"Initial Checkpoint:{args.model}")
        print(f"Anchor Model:      {args.ref_model}")
        print(f"League Checkpoints:{args.league_dir}")
        print(f"Total Timesteps:   {args.timesteps}")
        print(f"Parallel Envs/GPU: {args.num_envs} (Total Envs: {args.num_envs * world_size})")
        print(f"Rollout Length:    {args.rollout_len} ({args.num_envs * args.rollout_len} samples/gpu/update)")
        print(f"Mini-Batch Size:   {args.batch_size} (Epochs: {args.ppo_epochs})")
        print(f"KL Anneal Range:   {args.kl_init} -> {args.kl_min}")
        print(f"Opp Refresh Rate:  Every {args.opp_update_interval} updates")

    train_teams = load_pool_teams(args.train_pool)
    holdout_teams = load_pool_teams(args.holdout_pool)
    if rank == 0:
        print(f"Loaded {len(train_teams)} training teams and {len(holdout_teams)} holdout teams.")

    # Collect historical league milestone checkpoints
    league_ckpts = []
    if Path(args.league_dir).exists():
        league_ckpts = sorted(Path(args.league_dir).glob("model17_rl_*.pt"))
    if Path(args.ref_model).exists() and Path(args.ref_model) not in league_ckpts:
        league_ckpts.append(Path(args.ref_model))
    if rank == 0:
        print(f"Found {len(league_ckpts)} historical league checkpoints.")

    # Prebuild serialized 4-mon JSON strings for instant zero-overhead resets
    if rank == 0:
        print("Pre-serializing 4-mon team combinations into cache...")
    team_jsons = []
    for t in train_teams:
        for _ in range(2):
            p = random.sample(t, min(4, len(t)))
            team_jsons.append(json.dumps(p))

    # 1. Learner Model
    if rank == 0:
        print("Initializing Learner Transformer17...")
    model = Transformer17().to(device)
    init_ckpt = load_checkpoint_weights(model, args.model, device)

    # 2. Opponent League Model (Frozen for inference)
    if rank == 0:
        print("Initializing Opponent League Transformer17...")
    opp_model = Transformer17().to(device)
    opp_model.load_state_dict(model.state_dict())
    opp_model.eval()
    for param in opp_model.parameters():
        param.requires_grad = False

    # 3. Reference Imitation Anchor Model (Frozen for KL penalty)
    ref_model = Transformer17().to(device)
    load_checkpoint_weights(ref_model, args.ref_model, device)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
        eval_model = model.module
    else:
        eval_model = model

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Initialize parallel battle slots
    num_envs = args.num_envs
    slots = [FastBattleSlot(team_jsons) for _ in range(num_envs)]
    if rank == 0:
        print(f"Initialized {len(slots)} parallel battle environments on rank {rank}.")

    # Pinned host memory buffers for ultra-fast DMA (Learner + Opponent)
    pin = device.type == "cuda"
    host_l_cat = torch.empty((num_envs, 17, 4), dtype=torch.long, pin_memory=pin)
    host_l_num = torch.empty((num_envs, 17, 16), dtype=torch.float32, pin_memory=pin)
    host_l_mask_a = torch.empty((num_envs, 19), dtype=torch.bool, pin_memory=pin)
    host_l_mask_b = torch.empty((num_envs, 19), dtype=torch.bool, pin_memory=pin)

    host_o_cat = torch.empty((num_envs, 17, 4), dtype=torch.long, pin_memory=pin)
    host_o_num = torch.empty((num_envs, 17, 16), dtype=torch.float32, pin_memory=pin)
    host_o_mask_a = torch.empty((num_envs, 19), dtype=torch.bool, pin_memory=pin)
    host_o_mask_b = torch.empty((num_envs, 19), dtype=torch.bool, pin_memory=pin)

    np_l_cat = host_l_cat.numpy()
    np_l_num = host_l_num.numpy()
    np_l_mask_a = host_l_mask_a.numpy()
    np_l_mask_b = host_l_mask_b.numpy()

    np_o_cat = host_o_cat.numpy()
    np_o_num = host_o_num.numpy()
    np_o_mask_a = host_o_mask_a.numpy()
    np_o_mask_b = host_o_mask_b.numpy()

    # Preallocated GPU rollout storage
    rollout_cat = torch.empty((args.rollout_len, num_envs, 17, 4), dtype=torch.long, device=device)
    rollout_num = torch.empty((args.rollout_len, num_envs, 17, 16), dtype=torch.float32, device=device)
    rollout_act_a = torch.empty((args.rollout_len, num_envs), dtype=torch.long, device=device)
    rollout_act_b = torch.empty((args.rollout_len, num_envs), dtype=torch.long, device=device)
    rollout_logp = torch.empty((args.rollout_len, num_envs), dtype=torch.float32, device=device)
    rollout_val = torch.empty((args.rollout_len, num_envs), dtype=torch.float32, device=device)
    rollout_rew = torch.empty((args.rollout_len, num_envs), dtype=torch.float32, device=device)
    rollout_done = torch.empty((args.rollout_len, num_envs), dtype=torch.float32, device=device)

    # Initial holdout evaluation
    if rank == 0:
        print("\nRunning baseline holdout evaluation...")
        base_winrate = evaluate_holdout(eval_model, holdout_teams, num_games=args.eval_games, device=device, cts=args.cts)
        print(f"--> Baseline Win Rate vs Holdout: {base_winrate * 100:.1f}%\n")

    total_steps = int(init_ckpt.get("total_steps", 0)) if isinstance(init_ckpt, dict) else 0
    total_games = int(init_ckpt.get("total_games", 0)) if isinstance(init_ckpt, dict) else 0
    update_idx = int(init_ckpt.get("update", 0)) if isinstance(init_ckpt, dict) else 0

    if rank == 0 and total_steps > 0:
        print(f"[*] Resuming Self-Play training from Step {total_steps:,} (Update {update_idx}, {total_games:,} games)")

    t0 = time.time()
    current_opp_label = "Mirror (Self-Play)"

    while total_steps < args.timesteps:
        update_idx += 1
        update_t0 = time.time()
        model.eval()

        # Dynamic Annealed KL Penalty: slowly decreases from kl_init to kl_min
        progress = min(1.0, total_steps / max(1, args.timesteps))
        current_kl_coef = args.kl_init - progress * (args.kl_init - args.kl_min)

        # Periodic Opponent League Refresh
        if update_idx % args.opp_update_interval == 0:
            roll = random.random()
            if roll < 0.6:
                # 60% Mirror Self-Play: snapshot current weights
                opp_model.load_state_dict(eval_model.state_dict())
                current_opp_label = "Mirror (Current Policy)"
            elif roll < 0.9 and league_ckpts:
                # 30% Historical Milestone Checkpoint
                chosen_ckpt = random.choice(league_ckpts)
                load_checkpoint_weights(opp_model, chosen_ckpt, device)
                current_opp_label = f"League ({chosen_ckpt.stem})"
            else:
                # 10% Anchor Imitation
                load_checkpoint_weights(opp_model, args.ref_model, device)
                current_opp_label = "Imitation Anchor"

        step_completed_games = 0

        # -------------------------------------------------------------
        # 1. BATCHED NEURAL SELF-PLAY ROLLOUT COLLECTION
        # -------------------------------------------------------------
        for step in range(args.rollout_len):
            for i in range(num_envs):
                if slots[i].done:
                    slots[i].reset()

            # Encode both Learner and Opponent perspectives
            for i in range(num_envs):
                slot = slots[i]
                l_side = slot.learner_side
                o_side = slot.opp_side

                enc_l = slot.battle.encode_17(l_side)
                np_l_cat[i] = np.frombuffer(enc_l["tok_cat"], dtype=np.uint32).reshape(17, 4)
                np_l_num[i] = np.frombuffer(enc_l["tok_num"], dtype=np.float32).reshape(17, 16)
                np_l_mask_a[i] = slot.battle.legal_action_mask_17(l_side, 0)
                np_l_mask_b[i] = slot.battle.legal_action_mask_17(l_side, 1)

                enc_o = slot.battle.encode_17(o_side)
                np_o_cat[i] = np.frombuffer(enc_o["tok_cat"], dtype=np.uint32).reshape(17, 4)
                np_o_num[i] = np.frombuffer(enc_o["tok_num"], dtype=np.float32).reshape(17, 16)
                np_o_mask_a[i] = slot.battle.legal_action_mask_17(o_side, 0)
                np_o_mask_b[i] = slot.battle.legal_action_mask_17(o_side, 1)

            # Fast non-blocking DMA transfer to GPU
            gpu_l_cat = host_l_cat.to(device, non_blocking=True)
            gpu_l_num = host_l_num.to(device, non_blocking=True)
            gpu_l_mask_a = host_l_mask_a.to(device, non_blocking=True)
            gpu_l_mask_b = host_l_mask_b.to(device, non_blocking=True)

            gpu_o_cat = host_o_cat.to(device, non_blocking=True)
            gpu_o_num = host_o_num.to(device, non_blocking=True)
            gpu_o_mask_a = host_o_mask_a.to(device, non_blocking=True)
            gpu_o_mask_b = host_o_mask_b.to(device, non_blocking=True)

            if args.cts:
                # 100% Closed Team Sheet: hide opposing items and abilities
                gpu_l_cat[:, 4:8, 1:3] = 0
                gpu_o_cat[:, 4:8, 1:3] = 0

            # Batched GPU forward pass under AMP autocast
            with torch.no_grad():
                with torch.autocast(device_type=device.type, dtype=torch.float16):
                    # 1. Learner Model Decisions
                    out_l = eval_model(gpu_l_cat, gpu_l_num)
                    logits_l_a = out_l["policy_a"].float()
                    logits_l_a.masked_fill_(~gpu_l_mask_a, -1e9)
                    dist_l_a = torch.distributions.Categorical(logits=logits_l_a)
                    act_l_a = dist_l_a.sample()
                    logp_l_a = dist_l_a.log_prob(act_l_a)
                    val_l = out_l["value"].squeeze(-1).float()

                    out_l_b = eval_model(gpu_l_cat, gpu_l_num, action_a=act_l_a)
                    logits_l_b = out_l_b["policy_b"].float()
                    dup_switch_l = (act_l_a == 16) | (act_l_a == 17)
                    mask_l_b = gpu_l_mask_b.clone()
                    mask_l_b[torch.arange(num_envs, device=device), act_l_a.clamp(0, 18)] &= ~dup_switch_l
                    mask_l_b[mask_l_b.sum(dim=-1) == 0, 18] = True
                    logits_l_b.masked_fill_(~mask_l_b, -1e9)
                    dist_l_b = torch.distributions.Categorical(logits=logits_l_b)
                    act_l_b = dist_l_b.sample()
                    logp_l_b = dist_l_b.log_prob(act_l_b)

                    # 2. Opponent League Decisions
                    out_o = opp_model(gpu_o_cat, gpu_o_num)
                    logits_o_a = out_o["policy_a"].float()
                    logits_o_a.masked_fill_(~gpu_o_mask_a, -1e9)
                    dist_o_a = torch.distributions.Categorical(logits=logits_o_a)
                    act_o_a = dist_o_a.sample()

                    out_o_b = opp_model(gpu_o_cat, gpu_o_num, action_a=act_o_a)
                    logits_o_b = out_o_b["policy_b"].float()
                    dup_switch_o = (act_o_a == 16) | (act_o_a == 17)
                    mask_o_b = gpu_o_mask_b.clone()
                    mask_o_b[torch.arange(num_envs, device=device), act_o_a.clamp(0, 18)] &= ~dup_switch_o
                    mask_o_b[mask_o_b.sum(dim=-1) == 0, 18] = True
                    logits_o_b.masked_fill_(~mask_o_b, -1e9)
                    dist_o_b = torch.distributions.Categorical(logits=logits_o_b)
                    act_o_b = dist_o_b.sample()

            act_l_a_cpu = act_l_a.cpu().numpy()
            act_l_b_cpu = act_l_b.cpu().numpy()
            act_o_a_cpu = act_o_a.cpu().numpy()
            act_o_b_cpu = act_o_b.cpu().numpy()

            step_rewards = np.zeros(num_envs, dtype=np.float32)
            step_dones = np.zeros(num_envs, dtype=np.float32)

            # Step parallel games with both neural agents in Rust engine
            for i in range(num_envs):
                slot = slots[i]
                ch_l_a, ch_l_b = slot.battle.actions_17_to_choices(slot.learner_side, int(act_l_a_cpu[i]), int(act_l_b_cpu[i]))
                ch_o_a, ch_o_b = slot.battle.actions_17_to_choices(slot.opp_side, int(act_o_a_cpu[i]), int(act_o_b_cpu[i]))

                if slot.learner_side == 0:
                    c1 = [ch_l_a, ch_l_b]
                    c2 = [ch_o_a, ch_o_b]
                else:
                    c1 = [ch_o_a, ch_o_b]
                    c2 = [ch_l_a, ch_l_b]

                res = slot.battle.step_move(c1, c2)
                while slot.battle.needs_replacements():
                    r0 = _replacement_choices(slot.battle, 0)
                    r1 = _replacement_choices(slot.battle, 1)
                    res = slot.battle.step_move(r0, r1)
                    if res != "continue":
                        break
                slot.turn += 1

                if res == "p1_win":
                    step_rewards[i] = 1.0 if slot.learner_side == 0 else -1.0
                    step_dones[i] = 1.0
                    slot.done = True
                    step_completed_games += 1
                elif res == "p2_win":
                    step_rewards[i] = -1.0 if slot.learner_side == 0 else 1.0
                    step_dones[i] = 1.0
                    slot.done = True
                    step_completed_games += 1
                elif res == "tie" or slot.turn >= 30:
                    step_rewards[i] = 0.0
                    step_dones[i] = 1.0
                    slot.done = True
                    step_completed_games += 1

            # Store Learner transitions in GPU rollout buffer
            rollout_cat[step].copy_(gpu_l_cat)
            rollout_num[step].copy_(gpu_l_num)
            rollout_act_a[step].copy_(act_l_a)
            rollout_act_b[step].copy_(act_l_b)
            rollout_logp[step].copy_(logp_l_a + logp_l_b)
            rollout_val[step].copy_(val_l)
            rollout_rew[step].copy_(torch.from_numpy(step_rewards).to(device, non_blocking=True))
            rollout_done[step].copy_(torch.from_numpy(step_dones).to(device, non_blocking=True))

        n_samples = args.rollout_len * num_envs
        total_steps += n_samples * world_size
        total_games += step_completed_games * world_size

        # -------------------------------------------------------------
        # 2. GAE ADVANTAGE CALCULATION DIRECTLY ON GPU
        # -------------------------------------------------------------
        with torch.no_grad():
            advantages = torch.zeros((args.rollout_len, num_envs), dtype=torch.float32, device=device)
            last_gae = torch.zeros(num_envs, dtype=torch.float32, device=device)
            for t in reversed(range(args.rollout_len)):
                if t == args.rollout_len - 1:
                    next_non_terminal = 1.0 - rollout_done[t]
                    next_values = torch.zeros(num_envs, dtype=torch.float32, device=device)
                else:
                    next_non_terminal = 1.0 - rollout_done[t]
                    next_values = rollout_val[t + 1]
                delta = rollout_rew[t] + args.gamma * next_values * next_non_terminal - rollout_val[t]
                last_gae = delta + args.gamma * args.gae_lambda * next_non_terminal * last_gae
                advantages[t] = last_gae
            returns = advantages + rollout_val

        flat_cat = rollout_cat.reshape(n_samples, 17, 4)
        flat_num = rollout_num.reshape(n_samples, 17, 16)
        flat_act_a = rollout_act_a.reshape(n_samples)
        flat_act_b = rollout_act_b.reshape(n_samples)
        flat_old_logp = rollout_logp.reshape(n_samples)
        flat_adv = advantages.reshape(n_samples)
        flat_ret = returns.reshape(n_samples)

        flat_adv = (flat_adv - flat_adv.mean()) / (flat_adv.std() + 1e-8)

        # -------------------------------------------------------------
        # 3. PPO MINI-BATCH SGD OPTIMIZATION (HIGH THROUGHPUT)
        # -------------------------------------------------------------
        model.train()
        clip_loss_total = 0.0
        vf_loss_total = 0.0
        kl_loss_total = 0.0
        n_updates = 0

        for epoch in range(args.ppo_epochs):
            perm = torch.randperm(n_samples, device=device)
            for start in range(0, n_samples, args.batch_size):
                b_idx = perm[start : start + args.batch_size]

                b_cat = flat_cat[b_idx]
                b_num = flat_num[b_idx]
                b_act_a = flat_act_a[b_idx]
                b_act_b = flat_act_b[b_idx]
                b_old_logp = flat_old_logp[b_idx]
                b_adv = flat_adv[b_idx]
                b_ret = flat_ret[b_idx]

                with torch.autocast(device_type=device.type, dtype=torch.float16):
                    out = model(b_cat, b_num, action_a=b_act_a)
                    val_pred = out["value"].squeeze(-1)

                    logits_a = out["policy_a"]
                    dist_a = torch.distributions.Categorical(logits=logits_a)
                    new_logp_a = dist_a.log_prob(b_act_a)
                    ent_a = dist_a.entropy().mean()

                    logits_b = out["policy_b"]
                    dist_b = torch.distributions.Categorical(logits=logits_b)
                    new_logp_b = dist_b.log_prob(b_act_b)
                    ent_b = dist_b.entropy().mean()

                    new_logp = new_logp_a + new_logp_b
                    ratio = torch.exp(new_logp - b_old_logp)
                    surr1 = ratio * b_adv
                    surr2 = torch.clamp(ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps) * b_adv
                    policy_loss = -torch.min(surr1, surr2).mean()

                    vf_loss = F.mse_loss(val_pred, b_ret)

                    with torch.no_grad():
                        ref_out = ref_model(b_cat, b_num, action_a=b_act_a)
                        ref_logits_a = ref_out["policy_a"]
                        ref_logits_b = ref_out["policy_b"]

                    kl_a = F.kl_div(F.log_softmax(logits_a, dim=-1), F.softmax(ref_logits_a, dim=-1), reduction="batchmean")
                    kl_b = F.kl_div(F.log_softmax(logits_b, dim=-1), F.softmax(ref_logits_b, dim=-1), reduction="batchmean")
                    kl_div = kl_a + kl_b

                    total_loss = (
                        policy_loss
                        + args.vf_coef * vf_loss
                        - args.ent_coef * (ent_a + ent_b)
                        + current_kl_coef * kl_div
                    )

                optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()

                clip_loss_total += policy_loss.item()
                vf_loss_total += vf_loss.item()
                kl_loss_total += kl_div.item()
                n_updates += 1

        # -------------------------------------------------------------
        # 4. TELEMETRY & THROUGHPUT LOGGING
        # -------------------------------------------------------------
        update_time = time.time() - update_t0
        avg_rew = float(rollout_rew.mean().item())
        pol_loss_avg = clip_loss_total / max(1, n_updates)
        vf_loss_avg = vf_loss_total / max(1, n_updates)
        kl_loss_avg = kl_loss_total / max(1, n_updates)

        u_steps_per_sec = (n_samples * world_size) / update_time
        u_games_per_sec = (step_completed_games * world_size) / update_time
        u_games_per_gpu = step_completed_games / update_time

        if rank == 0:
            print(
                f"Update {update_idx:4d} | "
                f"Steps: {total_steps:8d}/{args.timesteps} ({u_steps_per_sec:5.0f} st/s) | "
                f"Games: {total_games:7d} ({u_games_per_sec:4.0f} g/s) | "
                f"Opp: {current_opp_label[:14]:<14} | "
                f"Rew: {avg_rew:+.3f} | "
                f"Pol: {pol_loss_avg:.3f} | VF: {vf_loss_avg:.3f} | KL: {kl_loss_avg:.3f} (coef: {current_kl_coef:.4f})"
            )

        # Periodic checkpointing
        if rank == 0 and total_steps % args.save_interval < (n_samples * world_size):
            save_path = out_dir / f"model17_sp_{total_steps // 1000}k.pt"
            torch.save(
                {
                    "model_state_dict": eval_model.state_dict(),
                    "total_steps": total_steps,
                    "total_games": total_games,
                    "update": update_idx,
                },
                save_path,
            )
            print(f"[*] Saved Self-Play RL checkpoint to {save_path}")

        # Periodic holdout evaluation
        if rank == 0 and total_steps % args.eval_interval < (n_samples * world_size):
            winrate = evaluate_holdout(eval_model, holdout_teams, num_games=args.eval_games, device=device, cts=args.cts)
            print(f"--> [Eval @ {total_steps} steps] Win Rate vs Holdout ({args.eval_games} games): {winrate * 100:.1f}%\n")

    # Final Checkpoint
    if rank == 0:
        final_path = out_dir / "model17_sp_final.pt"
        torch.save(
            {
                "model_state_dict": eval_model.state_dict(),
                "total_steps": total_steps,
                "total_games": total_games,
                "update": update_idx,
            },
            final_path,
        )
        print(f"\n[✓] Self-Play PPO RL Complete! Final model saved to {final_path}")

    if is_distributed:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
