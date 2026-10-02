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
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

KEYS = ["tok_cat", "tok_num", "mv_cat", "mv_dyn", "dmg", "glob", "act_tok", "mask", "ser_tok", "ser_mv", "ser_glob"]

# ---------------------------------------------------------------- workers

_W: dict = {}


def _init(ports: list[int], fmt: str, concurrency: int, rating: float, teams: str | None = None,
          my_team: str | None = None, change_after_loss: bool = False, closed_sheets: bool = False,
          speed_inference: bool = False, set_guess: bool = False, damage_inference: bool = False) -> None:
    torch.set_num_threads(1)
    _W.update(ports=ports, fmt=fmt, concurrency=concurrency, rating=rating, teams=teams, my_team=my_team,
              change_after_loss=change_after_loss, closed_sheets=closed_sheets, speed_inference=speed_inference,
              set_guess=set_guess, damage_inference=damage_inference)


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
        model = load_policy(policy)
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
        _W["snap"] = NNPlayer(load_policy(policy), r, account_configuration=account("snap"),
                              team=teams(), **common)
        _W["evalA"] = NNPlayer(load_policy(policy), r, account_configuration=account("evA"),
                               team=teams(), **common)
        _W["evalB"] = NNPlayer(load_policy(policy), r, account_configuration=account("evB"),
                               team=teams(), **common)
        _W["heuristic"] = SimpleHeuristicsPlayer(account_configuration=account("heur"),
                                                 team=teams(), **common)
        for k in ("me", "self", "snap", "evalA", "evalB"):  # every model in the worker reads the same features
            _W[k].speed_inference = _W.get("speed_inference", False)
            _W[k].set_guess = (_W["teams"], _W["fmt"]) if _W.get("set_guess") and _W["teams"] else None
            _W[k].damage_inference = _W.get("damage_inference", False)
    return _W


def _state(path: str) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)["state"]


def _load_into(player, path: str) -> None:
    """Give an opponent `player` the policy at `path`. Checkpoints with another architecture
    (e.g. archive/mb-bo3-do-v1, from before the blocked flag and opponent head) get their own model."""
    from ai_vgc.nn.player import load_policy

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    own = player.__dict__.setdefault("own_model", player.model)
    if ckpt["config"] == own.config:
        player.model = own
        own.load_state_dict(ckpt["state"])
    else:
        cache = _W.setdefault("other_models", {})
        if path not in cache:
            cache[path] = load_policy(path)
        player.model = cache[path]


def _flatten(episodes: list[tuple[list[dict], float]]) -> dict[str, np.ndarray]:
    steps = [s for ep, _ in episodes for s in ep]
    out = {k: np.stack([s[k] for s in steps]) for k in KEYS + ["action"]}
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
            la, lb, v = model(b, act[:, 0])
            lpa, lpb = F.log_softmax(masked(la, ma), -1), F.log_softmax(masked(lb, mb), -1)
            logp = lpa.gather(1, act[:, :1])[:, 0] + lpb.gather(1, act[:, 1:])[:, 0]
            ratio = torch.exp(logp - b["logp"])
            # Stop once the policy has moved too far from the one that played the games.
            if (b["logp"] - logp).mean().item() > args.target_kl:
                stop = True
                break
            adv = b["adv"]
            pg = -torch.min(ratio * adv, ratio.clamp(1 - args.clip, 1 + args.clip) * adv).mean()
            vf = F.binary_cross_entropy_with_logits(v, b["ret"].clamp(0, 1))
            ent = -((lpa.exp() * lpa).sum(-1) + (lpb.exp() * lpb).sum(-1)).mean()
            with torch.no_grad():
                ra, rb, _ = ref(b, act[:, 0])
                rpa, rpb = F.log_softmax(masked(ra, ma), -1), F.log_softmax(masked(rb, mb), -1)
            kl = ((lpa.exp() * (lpa - rpa)).sum(-1) + (lpb.exp() * (lpb - rpb)).sum(-1)).mean()
            loss = pg + args.vf_coef * vf - args.ent_coef * ent + args.kl_coef * kl
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            for k, x in (("pg", pg), ("vf", vf), ("ent", ent), ("kl", kl),
                         ("clip", ((ratio - 1).abs() > args.clip).float().mean())):
                stats[k] += x.item()
            stats["n"] += 1
    return {k: v / max(stats["n"], 1) for k, v in stats.items() if k != "n"}


def _print_nash(meta: dict) -> None:
    w = sorted(zip(meta["nash"], meta["pool"]), reverse=True)
    print("nash mix: " + ", ".join(f"{Path(p).stem} {x:.2f}" for x, p in w if x >= 0.01), flush=True)


def save(model, path: Path, args, extra: dict) -> None:
    from ai_vgc.nn.encode import vocab_sizes

    tmp = path.with_suffix(".tmp")
    torch.save({"state": {k: v.cpu() for k, v in model.state_dict().items()}, "config": model.config,
                "sizes": vocab_sizes(), "args": {k: str(v) for k, v in vars(args).items()}} | extra, tmp)
    tmp.replace(path)  # atomic, so workers never read a half-written file


def main() -> None:
    from ai_vgc.nn.player import load_policy
    from ai_vgc.nn.tb import scalars, writer
    from ai_vgc.showdown import ensure_server

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--init", default="data/models/archive/mb-bo3-bc-v1.pt", help="imitation checkpoint to start from")
    ap.add_argument("--out", type=Path, default=Path("data/models/archive/mb-bo3-ppo-v1.pt"))
    ap.add_argument("--resume", action="store_true", help="continue from --out if it exists")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--games", type=int, default=2048, help="games per iteration")
    ap.add_argument("--job-games", type=int, default=32, help="games per worker job")
    ap.add_argument("--workers", type=int, default=32)
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
                    help="opponent mix: self, bc (the --init policy), pool (snapshots, uniform), "
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
    ap.add_argument("--value-warmup", type=int, default=3, help="first iterations train only the value head")
    ap.add_argument("--damage-inference", action="store_true",
                    help="closed sheets: every model's damage features use estimated opposing stats (ai_vgc.bulk)")
    ap.add_argument("--set-guess", action="store_true",
                    help="closed sheets: every model reads the opponent's unrevealed sets filled in with search's guess")
    ap.add_argument("--speed-inference", action="store_true",
                    help="every model in the games reads opposing Speed narrowed from turn order (ai_vgc.speed)")
    ap.add_argument("--closed-sheets", action="store_true",
                    help="every player declines open team sheets (Bo1 M-C, where they're optional): closed (CTS) games")
    ap.add_argument("--change-after-loss", action="store_true",
                    help="Bo3: the learner changes its leads after losing a game, as the ladder bot does")
    ap.add_argument("--reg-every", type=int, default=0,
                    help="R-NaD: replace the KL anchor with the current policy every N iterations (0 = keep --ref)")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_policy(args.init, str(dev))
    ref = load_policy(args.ref or args.init, str(dev))
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
    pool = [*args.pool_init, *sorted(str(p) for p in snap_dir.glob("iter_*.pt"))]
    meta_path = args.out.with_name(args.out.stem + "_meta.json")
    meta = json.loads(meta_path.read_text()) if args.resume and meta_path.exists() else {"pool": [], "win": []}
    use_nash = "nash" in mix

    base = args.port or 20000 + (int(os.environ.get("SLURM_JOB_ID", os.getpid())) % 400) * 25
    ports = [base + i for i in range(args.servers)]
    procs = [ensure_server(p, args.showdown) for p in ports]
    print(f"{args.servers} Showdown servers on ports {ports[0]}-{ports[-1]}; learner on {dev}", flush=True)

    ctx = multiprocessing.get_context("spawn")
    try:
        with ProcessPoolExecutor(args.workers, mp_context=ctx, initializer=_init,
                                 initargs=(ports, args.format, args.concurrency, args.rating,
                                           str(args.teams) if args.teams else None,
                                           str(args.my_team) if args.my_team else None,
                                           args.change_after_loss, args.closed_sheets, args.speed_inference,
                                           args.set_guess, args.damage_inference)) as ex:
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
                for _ in range(math.ceil(args.games / args.job_games)):
                    kinds = [k for k in mix if (k not in ("pool", "nash") or pool) and (k != "spec" or args.specialists)
                             and (k != "human" or args.human)]
                    kind = random.choices(kinds, [mix[k] for k in kinds])[0]
                    opp = (args.init if kind == "bc" else random.choice(pool) if kind == "pool"
                           else random.choices(meta["pool"], meta["nash"])[0] if kind == "nash"
                           else random.choice(args.specialists) if kind == "spec"
                           else args.human if kind == "human" else None)
                    jobs.append((str(current), kind, opp, args.job_games))
                parts, results = [], {}
                for kind, data, mine in ex.map(_play, jobs):
                    if data is not None:
                        parts.append(data)
                    results.setdefault(kind, []).extend(mine)
                t_play = time.time() - t0
                if not parts:
                    print(f"iter {it}: no games finished", flush=True)
                    continue
                data = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
                adv, ret = gae(data.pop("value"), data.pop("reward"), data.pop("last"), args.lam)
                data["adv"] = (adv - adv.mean()) / (adv.std() + 1e-8)
                data["ret"] = ret
                data.pop("preview")
                tens = {k: torch.from_numpy(v).to(dev) for k, v in data.items()}
                st = ppo_update(model, ref, tens, opt, args, value_only=it <= args.value_warmup, vopt=vopt)

                save(model, current, args, {})
                save(model, args.out, args, {"iter": it, "opt": opt.state_dict()})
                if args.reg_every and it > args.value_warmup and it % args.reg_every == 0:
                    ref.load_state_dict(model.state_dict())
                    save(model, reg_path, args, {"iter": it})
                    print(f"iter {it}: KL anchor replaced by the current policy (R-NaD)", flush=True)
                if it % args.snapshot_every == 0:
                    snap = snap_dir / f"iter_{it:04d}.pt"
                    save(model, snap, args, {"iter": it})
                    pool.append(str(snap))
                    if use_nash:
                        t = time.time()
                        meta_add(ex, meta, str(snap), args.meta_games, args.job_games)
                        meta_path.write_text(json.dumps(meta))
                        print(f"meta-game: added {snap.name} ({time.time() - t:.0f}s)", flush=True)
                        _print_nash(meta)
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
    finally:
        for p in procs:
            if p:
                p.terminate()


if __name__ == "__main__":
    main()
