"""Train the policy network on human decisions (behavior cloning + win prediction).

    uv run python -m ai_vgc.nn.train                          # all data/nn/*.npz
    uv run python -m ai_vgc.nn.train --epochs 2 --d 64 --layers 1   # quick CPU test

Loss = weighted cross-entropy on each slot's action + value_coef x BCE on the
game result. Per-decision weights favour strong and winning players:

    w = clip(exp((rating - rating_center) / rating_scale), 0.25, 4) x (1 if won else loss_weight)

Unrated games count as rating 1100. The rating is also an input, so at play time
the network can be asked to play "like a 1800 player".

Options for the retrain (docs/training-ideas.md, ideas 2, 4 and 5):

    --aug              mirror our two slots and/or the opponent's two slots at random
                       (actions, targets, masks and features swapped to match): the same
                       decision, seen from the other position
    --awr-beta B       advantage-weighted regression: also multiply w by
                       min(exp(A / (B x std A)), --awr-max), where A = V(next decision) - V(this one)
                       (the game result after the last) and V comes from --awr-value. Human moves
                       that the value net thinks helped count more, blunders less
    --aux-coef C       add C x cross-entropy of the opponent-action head (what each opposing
                       active did that turn: move slot or switch, and target)
    --aux-mt           with --aux-coef: predict the target per move (Protect and an attack get
                       their own targets) rather than one target per Pokemon

Games are split by id: 5% held out for validation. The checkpoint with the
lowest validation policy loss is saved to --out.
"""

from __future__ import annotations

import argparse
import contextlib
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ai_vgc.nn.encode import N_ACT, N_DMG, move_table, vocab_sizes
from ai_vgc.nn.model import Policy, masked, slot_b_mask
from ai_vgc.nn.tb import scalars, writer

DATA = Path("data/nn")
OUT = Path("data/models/all-bo3-bc-new.pt")
INPUTS = ["tok_cat", "tok_num", "mv_cat", "mv_dyn", "dmg", "glob", "act_tok", "mask",
          "ser_tok", "ser_mv", "ser_glob"]
ACT_A, ACT_B = 5, 6  # tok_num columns: "active in slot a / b"


def _target_perm(swap: tuple[int, int]) -> torch.Tensor:
    """Action id permutation that swaps two target indices (t + 2) of every move action."""
    perm = torch.arange(N_ACT)
    for a in range(7, N_ACT):
        base, ti = (a - 7) - (a - 7) % 5, (a - 7) % 5
        ti = swap[1] if ti == swap[0] else swap[0] if ti == swap[1] else ti
        perm[a] = 7 + base + ti
    return perm


OUR_PERM = _target_perm((0, 1))  # t = -2 <-> -1: our slot b <-> our slot a
FOE_PERM = _target_perm((3, 4))  # t = 1 <-> 2: their slot a <-> their slot b


def augment(b: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Swap our two active slots in a random half of the batch, and the opponent's two in an
    independent half. Doubles has no positional asymmetry, so each is an equally valid decision."""
    b = dict(b)
    B, dev = len(b["action"]), b["action"].device
    for ours in (True, False):
        flip = torch.rand(B, device=dev) < 0.5
        f = lambda x: flip.view(-1, *[1] * (x.dim() - 1))  # noqa: E731
        slots, other, toks = ([1, 0, 2, 3], slice(2, 4), slice(0, 6)) if ours else \
            ([0, 1, 3, 2], slice(0, 2), slice(6, 12))
        perm = (OUR_PERM if ours else FOE_PERM).to(dev)
        at = b["act_tok"][:, slots]
        b["act_tok"] = torch.where(f(at), at, b["act_tok"])
        d = b["dmg"][:, slots].clone()
        d[:, other] = d[:, other].flip(3)  # the other side's moves: their targets swapped places
        b["dmg"] = torch.where(f(d), d, b["dmg"])
        tn = b["tok_num"].clone()
        tn[:, toks, ACT_A], tn[:, toks, ACT_B] = b["tok_num"][:, toks, ACT_B], b["tok_num"][:, toks, ACT_A]
        b["tok_num"] = torch.where(f(tn), tn, b["tok_num"])
        m = b["mask"][..., perm]
        act = b["action"].long()
        act = torch.where(act >= 0, perm[act.clamp(min=0)], act)
        if ours:
            m, act = m[:, [1, 0]], act[:, [1, 0]]
        b["mask"] = torch.where(f(m), m, b["mask"])
        b["action"] = torch.where(f(act), act.to(b["action"].dtype), b["action"])
        if "opp" in b:
            o = b["opp"].clone()
            if ours:  # their targets: our a <-> our b
                t = o[..., 1]
                o[..., 1] = torch.where(t == 0, 1, torch.where(t == 1, 0, t))
            else:
                o = o[:, [1, 0]]
            b["opp"] = torch.where(f(o), o, b["opp"])
    return b


@torch.no_grad()
def awr_weights(data: dict[str, torch.Tensor], value_path: str, dev, batch: int, beta: float,
                w_max: float) -> torch.Tensor:
    """Per-decision exp(A / (beta std A)), A = V(next) - V(now) along each player's game."""
    from ai_vgc.nn.player import load_policy

    vm = load_policy(value_path, str(dev))
    n = len(data["action"])
    v = torch.empty(n, device=dev)
    for i in range(0, n, batch):
        b = {k: data[k][i:i + batch].to(dev) for k in INPUTS}
        v[i:i + batch] = torch.sigmoid(vm.value(vm.encode(b)[0])[:, 0].float())
    game, tl, pv = (data[k].to(dev) for k in ("game", "turns_left", "preview"))
    # Rows are stored per game, one player after the other, in decision order.
    same = (game[1:] == game[:-1]) & (tl[1:] <= tl[:-1]) & ~(pv[1:] & ~pv[:-1])
    nxt = torch.where(same, v[1:], data["won"][:-1].to(dev).float())
    target = torch.cat([nxt, data["won"][-1:].to(dev).float()])
    adv = target - v
    print(f"AWR: value from {value_path}; advantage std {adv.std():.3f}", flush=True)
    return torch.exp(adv / (beta * adv.std())).clamp(max=w_max)


def load(data_dir: Path, formats: list[str] | None = None) -> dict[str, torch.Tensor]:
    paths = [p for p in sorted(data_dir.glob("*.npz")) if not formats or p.stem in formats]
    parts = [dict(np.load(p)) for p in paths]
    if not parts:
        raise SystemExit(f"no .npz files in {data_dir}; run `python -m ai_vgc.nn.dataset` first")
    from ai_vgc.nn.encode import T
    from ai_vgc.nn.series import N_SER_GLOB, N_SER_MV, N_SER_TOK

    # Datasets built before the series context: zeros, as in game 1.
    for p in parts:
        n = len(p["action"])
        for k, shape in (("ser_tok", (T, N_SER_TOK)), ("ser_mv", (T, 4, N_SER_MV)), ("ser_glob", (N_SER_GLOB,))):
            p.setdefault(k, np.zeros((n, *shape), np.float16))
    data = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    # CUDA can't index unsigned tensors (the crc32 game id is uint32).
    data = {k: v.astype(np.int64) if v.dtype.kind == "u" else v for k, v in data.items()}
    return {k: torch.from_numpy(v) for k, v in data.items()}


def weights(rating: torch.Tensor, won: torch.Tensor, center: float, scale: float,
            loss_weight: float) -> torch.Tensor:
    r = torch.where(rating > 0, rating.float(), torch.full_like(rating, 1100, dtype=torch.float32))
    w = torch.exp((r - center) / scale).clamp(0.25, 4.0)
    return w * torch.where(won, 1.0, loss_weight)


def step(model: Policy, b: dict[str, torch.Tensor], aux: bool = False):
    """Per-sample losses and predictions for one batch (plus the aux loss when `aux`)."""
    act = b["action"].long()
    mask_a, mask_b = b["mask"][:, 0], slot_b_mask(b["mask"][:, 1], act[:, 0])
    # A label outside the legal mask would be unlearnable; ignore it.
    for s, m in ((0, mask_a), (1, mask_b)):
        ok = act[:, s] >= 0
        ok[ok.clone()] &= m[ok, act[ok, s]]
        act[~ok, s] = -1
    enc = model.encode(b)
    la, lb, v = model.slot_logits(enc, b, 0), model.slot_logits(enc, b, 1, act[:, 0]), \
        model.value(enc[0])[:, 0].float()
    la, lb = masked(la, mask_a), masked(lb, mask_b)
    ce = torch.stack([F.cross_entropy(la, act[:, 0], ignore_index=-1, reduction="none"),
                      F.cross_entropy(lb, act[:, 1], ignore_index=-1, reduction="none")], 1)
    valid = act >= 0
    bce = F.binary_cross_entropy_with_logits(v, b["won"].float(), reduction="none")
    hits = torch.stack([la.argmax(1) == act[:, 0], lb.argmax(1) == act[:, 1]], 1) & valid
    if not aux:
        return ce, valid, bce, hits, v
    om, ot = model.opp_logits(enc, b)
    opp = b["opp"].long()
    # Targets of the move actually used (switches have target -1 and are ignored).
    ot = ot.gather(2, opp[..., 0].clamp(0, 3)[..., None, None].expand(-1, -1, 1, 3))[:, :, 0]
    aux_ce = F.cross_entropy(om.flatten(0, 1), opp[..., 0].flatten(), ignore_index=-1, reduction="sum") + \
        F.cross_entropy(ot.flatten(0, 1), opp[..., 1].flatten(), ignore_index=-1, reduction="sum")
    aux_hit = ((om.argmax(-1) == opp[..., 0]) & (opp[..., 0] >= 0)).sum()
    return ce, valid, bce, hits, v, aux_ce, aux_hit, (opp[..., 0] >= 0).sum()


def evaluate(model: Policy, data, idx: torch.Tensor, dev, batch: int, amp, aux: bool = False) -> dict[str, float]:
    model.eval()
    tot = {"ce": 0.0, "n": 0, "hit": 0, "pv_hit": 0, "pv_n": 0, "bce": 0.0, "v_hit": 0, "g": 0,
           "aux_hit": 0, "aux_n": 0}
    with torch.no_grad(), amp():
        for i in range(0, len(idx), batch):
            b = {k: t[idx[i:i + batch]].to(dev, non_blocking=True) for k, t in data.items()}
            ce, valid, bce, hits, v, *extra = step(model, b, aux)
            if extra:
                tot["aux_hit"] += extra[1].item()
                tot["aux_n"] += extra[2].item()
            pv = b["preview"][:, None].expand_as(valid)
            turn = valid & ~pv
            tot["ce"] += ce[turn].sum().item()
            tot["n"] += turn.sum().item()
            tot["hit"] += hits[turn].sum().item()
            tot["pv_hit"] += hits[valid & pv].sum().item()
            tot["pv_n"] += (valid & pv).sum().item()
            tot["bce"] += bce.sum().item()
            tot["v_hit"] += ((v > 0) == b["won"]).sum().item()
            tot["g"] += len(v)
    model.train()
    return {
        "loss": tot["ce"] / max(tot["n"], 1),
        "top1": tot["hit"] / max(tot["n"], 1),
        "preview_top1": tot["pv_hit"] / max(tot["pv_n"], 1),
        "value_bce": tot["bce"] / max(tot["g"], 1),
        "value_acc": tot["v_hit"] / max(tot["g"], 1),
        "opp_top1": tot["aux_hit"] / max(tot["aux_n"], 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--formats", nargs="*", default=None,
                    help="only these .npz files in --data (stems, e.g. gen9championsvgc2026regmc)")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--value-coef", type=float, default=0.5)
    ap.add_argument("--rating-center", type=float, default=1400)
    ap.add_argument("--rating-scale", type=float, default=300)
    ap.add_argument("--loss-weight", type=float, default=0.5, help="weight of the losing side's decisions")
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--aug", action="store_true", help="random slot-swap symmetry augmentation")
    ap.add_argument("--awr-beta", type=float, default=0.0, help="advantage-weighted regression temperature (0 = off)")
    ap.add_argument("--awr-value", default="data/models/archive/mb-bo3-bc-v1.pt", help="checkpoint whose value head gives A")
    ap.add_argument("--awr-max", type=float, default=5.0, help="cap on the AWR weight")
    ap.add_argument("--aux-mt", action="store_true",
                    help="opponent head predicts a target per move rather than one per Pokemon")
    ap.add_argument("--series", action="store_true", help="read the Bo3 series context (series.py)")
    ap.add_argument("--init", default=None,
                    help="start from this checkpoint (its architecture; --series layers start at zero)")
    ap.add_argument("--aux-coef", type=float, default=0.0, help="weight of the opponent-action head's loss (0 = no head)")
    args = ap.parse_args()
    aux = args.aux_coef > 0

    torch.manual_seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":
        amp = lambda: torch.autocast("cuda", dtype=torch.bfloat16)  # noqa: E731
    else:
        amp = contextlib.nullcontext
    t0 = time.time()
    data = load(args.data, args.formats)
    n = len(data["action"])
    val = (data["game"].long() % 1000) < args.val_frac * 1000
    tr_idx, va_idx = torch.nonzero(~val)[:, 0], torch.nonzero(val)[:, 0]
    if "opp" not in data:  # datasets built before the opponent-action labels
        if aux:
            raise SystemExit("--aux-coef needs a dataset with opponent labels; rebuild it with ai_vgc.nn.dataset")
        data["opp"] = torch.full((n, 2, 2), -1, dtype=torch.int8)
    w_all = weights(data["rating"], data["won"], args.rating_center, args.rating_scale, args.loss_weight)
    if args.awr_beta > 0:
        w_all = w_all * awr_weights(data, args.awr_value, dev, args.batch * 4, args.awr_beta, args.awr_max).cpu()
    w_all = w_all / w_all[tr_idx].mean()
    data["w"] = w_all
    if dev.type == "cuda":
        # The whole dataset (a few GB) fits on one GPU, so batches never cross PCIe.
        data = {k: v.to(dev) for k, v in data.items()}
        tr_idx, va_idx = tr_idx.to(dev), va_idx.to(dev)
    print(f"{n} decisions ({len(tr_idx)} train / {len(va_idx)} val) loaded in {time.time() - t0:.0f}s; "
          f"device {dev}", flush=True)

    if args.init:
        ckpt = torch.load(args.init, map_location="cpu", weights_only=False)
        cfg = ckpt["config"] | {"series": args.series or ckpt["config"].get("series", False)}
        if aux and not cfg.get("aux"):
            raise SystemExit(f"--aux-coef needs an opponent head, and {args.init} has none")
        model = Policy(ckpt["sizes"], move_table(), **cfg)
        missing, unexpected = model.load_state_dict(ckpt["state"], strict=False)
        assert not unexpected and all(k.startswith("ser_") for k in missing), (missing, unexpected)
        model = model.to(dev)
        print(f"initialised from {args.init}; new: {sorted({k.split('.')[0] for k in missing}) or 'none'}",
              flush=True)
    else:
        model = Policy(vocab_sizes(), move_table(), args.d, args.layers, args.heads, args.dropout,
                       n_dmg=N_DMG, aux=aux, aux_mt=aux and args.aux_mt, series=args.series).to(dev)
    print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps = args.epochs * math.ceil(len(tr_idx) / args.batch)
    warm = min(1000, steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(warm, 1)) * 0.5 * (1 + math.cos(math.pi * min(s / steps, 1))))

    best = float("inf")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tb = writer("train", args.out, args)
    for epoch in range(1, args.epochs + 1):
        t = time.time()
        perm = tr_idx[torch.randperm(len(tr_idx), device=tr_idx.device)]
        run_loss = run_n = 0.0
        for i in range(0, len(perm), args.batch):
            b = {k: v[perm[i:i + args.batch]].to(dev, non_blocking=True) for k, v in data.items()}
            if args.aug:
                b = augment(b)
            with amp():
                ce, valid, bce, _, _, *extra = step(model, b, aux)
            w = b["w"]
            pol = (ce * w[:, None] * valid).sum() / (w[:, None] * valid).sum().clamp(min=1e-6)
            loss = pol + args.value_coef * (bce * w).sum() / w.sum()
            if extra:
                loss = loss + args.aux_coef * extra[0] / extra[2].clamp(min=1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            run_loss += pol.item()
            run_n += 1
        m = evaluate(model, {k: v for k, v in data.items() if k != "w"}, va_idx, dev, args.batch * 2, amp, aux)
        saved = ""
        if m["loss"] < best:
            best = m["loss"]
            torch.save({"state": model.state_dict(), "config": model.config, "sizes": vocab_sizes(),
                        "args": vars(args) | {"data": str(args.data), "out": str(args.out)},
                        "val": m, "epoch": epoch}, args.out)
            saved = " (saved)"
        scalars(tb, "val", m, epoch)
        tb.add_scalar("train/loss", run_loss / max(run_n, 1), epoch)
        tb.add_scalar("time/epoch_s", time.time() - t, epoch)
        tb.flush()
        print(f"epoch {epoch}: train {run_loss / max(run_n, 1):.3f} | val loss {m['loss']:.3f} "
              f"top1 {m['top1']:.1%} preview {m['preview_top1']:.1%} | value acc {m['value_acc']:.1%} "
              f"bce {m['value_bce']:.3f}" + (f" | opp top1 {m['opp_top1']:.1%}" if aux else "") +
              f" | {time.time() - t:.0f}s{saved}", flush=True)


if __name__ == "__main__":
    main()
