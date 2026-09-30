"""Expert iteration (idea 7, step 4): train the policy towards what search picks.

Search (search.py) wins about 6 points more than the policy alone, but it costs ~1.6 s a
decision. Distilling it into the policy keeps the gain at no play-time cost, and a better policy
makes the next round of search better too.

1. gen: play games with search and record every searched decision: the inputs, our k candidate
   joint actions, search's expected value for each (against the opponent head's replies) and the
   policy's log-prob, plus the game result.

       uv run --extra nn python -m ai_vgc.nn.exit gen --model data/models/archive/mb-bo3-rnad-v3.pt \\
           --opp-model data/models/all-bo3-opp-v2.pt --opponent nn:data/models/all-bo3-bc-v2.pt --n 50 \\
           --out data/exit/r1/a.npz

2. train: fine-tune from --init. The target over the k candidates is softmax(score / tau), with
   score = value + prior x log-prob as search itself ranks them (tau 0 = search's pick only).
   A KL penalty towards --init keeps the policy close to the one search was built on (and to
   human play), and the value head keeps learning from the results.

       uv run --extra nn python -m ai_vgc.nn.exit train --init data/models/archive/mb-bo3-rnad-v3.pt \\
           --data data/exit/r1 --out data/models/archive/mb-bo3-exit-v1.pt

Then evaluate the new policy without search (player.py), and repeat gen with it.
"""

from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ai_vgc.nn.model import masked, slot_b_mask
from ai_vgc.nn.tb import scalars, writer

INPUTS = ["tok_cat", "tok_num", "mv_cat", "mv_dyn", "dmg", "glob", "act_tok", "mask", "ser_tok", "ser_mv", "ser_glob"]


# ---------------------------------------------------------------- gen

def gen(args: argparse.Namespace) -> None:
    from poke_env import ServerConfiguration
    from poke_env.player import SimpleHeuristicsPlayer

    from ai_vgc.nn.player import NNPlayer
    from ai_vgc.nn.search import SearchPlayer
    from ai_vgc.showdown import account, ensure_server
    from ai_vgc.teams import TEAMS_DIR, RandomPoolTeambuilder

    torch.set_num_threads(1)
    k = args.search_k
    rows: dict[str, list[dict]] = {}  # battle tag -> searched decisions
    done: list[dict] = []

    def record(battle, d) -> None:
        n = len(d["actions"])
        acts = np.full((k, 2), -1, np.int16)
        acts[:n] = d["actions"]
        val, lp = np.full(k, np.nan, np.float32), np.full(k, np.nan, np.float32)
        val[:n], lp[:n] = d["value"], d["logp"]
        rows.setdefault(battle.battle_tag, []).append(
            {**d["obs"], "cand": acts, "cand_v": val, "cand_lp": lp})

    class Recorder(SearchPlayer):
        def _battle_finished_callback(self, battle) -> None:
            super()._battle_finished_callback(battle)
            won = 1.0 if battle.won else 0.0 if battle.lost else 0.5
            for r in rows.pop(battle.battle_tag, []):
                done.append(r | {"won": np.float32(won)})

    teams = lambda: RandomPoolTeambuilder(args.teams) if args.teams else RandomPoolTeambuilder()  # noqa: E731
    proc = ensure_server(args.port, args.showdown)
    common = dict(max_concurrent_battles=args.concurrency, accept_open_team_sheet=True, log_level=40,
                  server_configuration=ServerConfiguration(f"ws://localhost:{args.port}/showdown/websocket",
                                                           "https://play.pokemonshowdown.com/action.php?"))
    me = Recorder(args.model, args.rating, True, account_configuration=account("nnx"), team=teams(),
                  opp_model=args.opp_model, fmt=args.format, teams_dir=args.teams or TEAMS_DIR,
                  showdown=args.showdown or "pokemon-showdown", k=k, opp_k=args.search_opp_k,
                  seeds=args.search_seeds, prior=args.search_prior, **common)
    me.on_search = record
    opps = args.opponent.split(",")
    t = time.time()
    try:
        for i, name in enumerate(opps):
            n = args.n // len(opps) + (i < args.n % len(opps))
            if name.startswith("nn:"):
                opp = NNPlayer(name[3:], args.rating, account_configuration=account("nnxo"), team=teams(),
                               battle_format=args.format, **common)
            else:
                opp = SimpleHeuristicsPlayer(account_configuration=account("heur"), team=teams(),
                                             battle_format=args.format, **common)
            before = me.n_won_battles, me.n_finished_battles
            asyncio.run(me.battle_against(opp, n_battles=n))
            w, g = me.n_won_battles - before[0], me.n_finished_battles - before[1]
            print(f"vs {name}: {w}/{g} = {w / max(g, 1):.1%}", flush=True)
    finally:
        if proc:
            proc.terminate()
    if not done:
        raise SystemExit("no searched decisions recorded")
    out = {key: np.stack([r[key] for r in done]) for key in done[0]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **out)
    print(f"{len(done)} searched decisions from {me.n_finished_battles} games -> {args.out} "
          f"({time.time() - t:.0f}s; {me.fallbacks} decisions by policy)", flush=True)


# ---------------------------------------------------------------- train

def targets(v: torch.Tensor, lp: torch.Tensor, prior: float, tau: float) -> torch.Tensor:
    """[B, k] target weights over the candidates (padding gets 0)."""
    score = torch.nan_to_num(v + prior * lp, nan=-1e9)
    if tau <= 0:
        return F.one_hot(score.argmax(1), score.shape[1]).float()
    return F.softmax(score / tau, 1)


def losses(model, ref, b: dict[str, torch.Tensor], args) -> dict[str, torch.Tensor]:
    cand = b["cand"].long()  # [B, k, 2]
    B, k = cand.shape[:2]
    tgt = targets(b["cand_v"].float(), b["cand_lp"].float(), args.prior, args.tau)
    ok = cand[..., 0] >= 0
    ma = b["mask"][:, 0]
    enc = model.encode(b)
    lpa = F.log_softmax(masked(model.slot_logits(enc, b, 0), ma), -1)
    # Slot b's log-probs given each candidate's slot-a action: the batch repeated k times.
    rep = lambda x: x.repeat_interleave(k, 0)  # noqa: E731
    enc_k = tuple(rep(e) for e in enc)
    b_k = {key: rep(b[key]) for key in ("act_tok", "dmg")}
    prev = cand[..., 0].clamp(min=0).flatten()
    mb = slot_b_mask(rep(b["mask"][:, 1]), prev)
    lpb = F.log_softmax(masked(model.slot_logits(enc_k, b_k, 1, prev), mb), -1)
    joint = lpa.gather(1, cand[..., 0].clamp(min=0)) + \
        lpb.gather(1, cand[..., 1].clamp(min=0).flatten()[:, None])[:, 0].view(B, k)
    # Candidates come in the policy's order, so candidate 0 is its own pick: where search agreed,
    # --agree-weight scales the example down (0 = learn only from search's overrides).
    over = tgt.argmax(1) != 0
    w = torch.where(over, 1.0, args.agree_weight)
    ce = -(w * (tgt * torch.where(ok, joint, 0.0)).sum(1)).sum() / w.sum().clamp(min=1e-6)

    # KL to the reference on slot a, and on slot b given the target's best slot-a action.
    best = tgt.argmax(1)
    a0 = cand[torch.arange(B), best, 0]
    mb0 = slot_b_mask(b["mask"][:, 1], a0)
    lpb0 = F.log_softmax(masked(model.slot_logits(enc, b, 1, a0), mb0), -1)
    with torch.no_grad():
        renc = ref.encode(b)
        rpa = F.log_softmax(masked(ref.slot_logits(renc, b, 0), ma), -1)
        rpb = F.log_softmax(masked(ref.slot_logits(renc, b, 1, a0), mb0), -1)
    kl = ((lpa.exp() * (lpa - rpa)).sum(-1) + (lpb0.exp() * (lpb0 - rpb)).sum(-1)).mean()
    vf = F.binary_cross_entropy_with_logits(model.value(enc[0])[:, 0].float(), b["won"].float())
    hit = joint.masked_fill(~ok, -1e9).argmax(1) == tgt.argmax(1)
    return {"loss": ce + args.kl_coef * kl + args.vf_coef * vf, "ce": ce, "kl": kl, "vf": vf,
            "agree": hit.float().mean(),
            # How often the policy now makes search's override itself (0 for the policy search ran on).
            "overrides": (hit & over).float().sum() / over.float().sum().clamp(min=1)}


def train(args: argparse.Namespace) -> None:
    from ai_vgc.nn.player import load_policy
    from ai_vgc.nn.rl import save

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parts = [dict(np.load(p)) for p in sorted(args.data.glob("**/*.npz"))]
    if not parts:
        raise SystemExit(f"no .npz files under {args.data}")
    data = {key: torch.from_numpy(np.concatenate([p[key] for p in parts])) for key in parts[0]}
    n = len(data["won"])
    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(n, generator=g)
    val, tr = perm[:max(1, int(n * args.val_frac))], perm[max(1, int(n * args.val_frac)):]
    print(f"{n} searched decisions from {len(parts)} files; {len(val)} held out", flush=True)

    model = load_policy(args.init, str(dev))
    ref = load_policy(args.init, str(dev))
    for p in ref.parameters():
        p.requires_grad_(False)
    model.eval()  # no dropout, as in rl.py
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)

    def batch(idx):
        return {key: v[idx].to(dev) for key, v in data.items()}

    @torch.no_grad()
    def evaluate() -> dict[str, float]:
        tot: dict[str, float] = {}
        for i in range(0, len(val), args.batch):
            idx = val[i:i + args.batch]
            for key, x in losses(model, ref, batch(idx), args).items():
                tot[key] = tot.get(key, 0.0) + x.item() * len(idx)
        return {key: x / len(val) for key, x in tot.items()}

    fmt = lambda d: " ".join(f"{key} {x:.4f}" for key, x in d.items())  # noqa: E731
    tb = writer("exit", args.out, args)
    m = evaluate()
    scalars(tb, "val", m, 0)
    print(f"before: {fmt(m)}", flush=True)
    for ep in range(1, args.epochs + 1):
        t = time.time()
        order = tr[torch.randperm(len(tr), generator=g)]
        for i in range(0, len(order), args.batch):
            out = losses(model, ref, batch(order[i:i + args.batch]), args)
            opt.zero_grad(set_to_none=True)
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        m = evaluate()
        scalars(tb, "val", m, ep)
        tb.add_scalar("time/epoch_s", time.time() - t, ep)
        tb.flush()
        print(f"epoch {ep}: {fmt(m)} ({time.time() - t:.0f}s)", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save(model, args.out, args, {})
    print(f"saved {args.out}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen", help="play search games and record the searched decisions")
    g.add_argument("--model", default="data/models/archive/mb-bo3-rnad-v3.pt")
    g.add_argument("--opp-model", default="data/models/all-bo3-opp-v2.pt", help="opponent-action head for search")
    g.add_argument("--opponent", default="nn:data/models/all-bo3-bc-v2.pt",
                   help="comma-separated: nn:<checkpoint> or heuristic; --n is split between them")
    g.add_argument("--n", type=int, default=50)
    g.add_argument("--out", type=Path, required=True)
    g.add_argument("--format", default="gen9championsvgc2026regmb")
    g.add_argument("--port", type=int, default=8000)
    g.add_argument("--concurrency", type=int, default=8)
    g.add_argument("--rating", type=float, default=1700)
    g.add_argument("--teams", type=Path, default=None)
    g.add_argument("--showdown", type=Path, default=None)
    g.add_argument("--search-k", type=int, default=6)
    g.add_argument("--search-opp-k", type=int, default=6)
    g.add_argument("--search-seeds", type=int, default=2)
    g.add_argument("--search-prior", type=float, default=0.1)
    t = sub.add_parser("train", help="fine-tune the policy towards the recorded search choices")
    t.add_argument("--init", default="data/models/archive/mb-bo3-rnad-v3.pt")
    t.add_argument("--data", type=Path, required=True, help="folder of gen .npz files (searched recursively)")
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--epochs", type=int, default=3)
    t.add_argument("--batch", type=int, default=512)
    t.add_argument("--lr", type=float, default=3e-5)
    t.add_argument("--tau", type=float, default=0.02, help="target temperature over search scores (0 = its pick)")
    t.add_argument("--prior", type=float, default=0.1, help="log-prob weight in the score, as in the search runs")
    t.add_argument("--agree-weight", type=float, default=1.0,
                   help="weight of decisions where search kept the policy's pick (0 = learn only from overrides)")
    t.add_argument("--kl-coef", type=float, default=0.1, help="pull towards --init")
    t.add_argument("--vf-coef", type=float, default=0.5)
    t.add_argument("--val-frac", type=float, default=0.05)
    t.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    gen(args) if args.cmd == "gen" else train(args)


if __name__ == "__main__":
    main()
