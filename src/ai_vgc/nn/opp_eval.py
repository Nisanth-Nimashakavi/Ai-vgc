"""How well the opponent-action head predicts what human opponents actually did, on the
held-out games (the same 5% split as training). Search (idea 7) plays our candidates against
the head's top joint replies, so if the real reply is often missing from them, search is
optimising against the wrong opponent.

    uv run --extra nn python -m ai_vgc.nn.opp_eval data/models/all-bo3-bc-v2.pt data/models/archive/all-bo3-bc-v2-aux.pt

Per opposing slot (turns where the log shows what it did, about 65%):
  move top-1 / top-2     over its four moves + switch
  action top-1 / top-3   move and target (our slot a / our slot b / other) or switch

Joint, both slots labelled (what search's candidate list sees):
  move top-6             the pair of moves (or switches) among the 6 likeliest of 25
  action top-6 / top-12  move and target pairs, among the likeliest of 13 x 13

Search keeps the target only for targeted moves, so its real list sits between the two joint
numbers. The uniform line is the same measure for a head that knows nothing. Rows are split by
the rating recorded for the game (unrated = 0).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from ai_vgc.nn.player import load_policy
from ai_vgc.nn.train import DATA, INPUTS, load

BUCKETS = [("unrated", 0, 1), ("<1300", 1, 1300), ("1300-1500", 1300, 1500), ("1500+", 1500, 10**6)]


def slot_actions(mv: torch.Tensor, tg: torch.Tensor) -> torch.Tensor:
    """[B, 2, 13] log-probs: move i with target j at 3i + j, switch at 12 (tg is [B, 2, 4, 3])."""
    lm, lt = F.log_softmax(mv, -1), F.log_softmax(tg, -1)
    moves = (lm[..., :4, None] + lt).flatten(-2)
    return torch.cat([moves, lm[..., 4:]], -1)


def action_ids(opp: torch.Tensor) -> torch.Tensor:
    """[B, 2] ids into slot_actions (-1 when unlabelled)."""
    m, t = opp[..., 0], opp[..., 1]
    return torch.where(m < 0, -1, torch.where(m == 4, 12, 3 * m + t.clamp(min=0)))


def in_top(logp: torch.Tensor, label: torch.Tensor, k: int) -> torch.Tensor:
    return (logp.topk(min(k, logp.shape[-1]), -1).indices == label[..., None]).any(-1)


def joint_hit(a: torch.Tensor, b: torch.Tensor, la: torch.Tensor, lb: torch.Tensor, k: int) -> torch.Tensor:
    j = (a[:, :, None] + b[:, None, :]).flatten(1)
    return in_top(j, la * b.shape[-1] + lb, k)


@torch.no_grad()
def measure(model, data, idx, dev, batch: int) -> dict[str, torch.Tensor]:
    """Per-row hit flags (and which rows count) for every measure."""
    out: dict[str, list[torch.Tensor]] = {}

    def add(name, v):
        out.setdefault(name, []).append(v.cpu())

    for i in range(0, len(idx), batch):
        rows = idx[i:i + batch]
        b = {k: data[k][rows].to(dev) for k in INPUTS}
        opp = data["opp"][rows].long().to(dev)
        if model is None:
            # Equally likely moves and slot actions; tiny noise so ties break at random.
            lm = torch.randn(len(rows), 2, 5, device=dev) * 1e-3
            acts = torch.randn(len(rows), 2, 13, device=dev) * 1e-3
        else:
            mv, tg = model.opp_logits(model.encode(b), b)
            acts, lm = slot_actions(mv, tg), F.log_softmax(mv, -1)
        m, aid = opp[..., 0], action_ids(opp)
        add("slot_n", (m >= 0))
        add("move@1", in_top(lm, m, 1) & (m >= 0))
        add("move@2", in_top(lm, m, 2) & (m >= 0))
        add("action@1", in_top(acts, aid, 1) & (m >= 0))
        add("action@3", in_top(acts, aid, 3) & (m >= 0))
        both = (m >= 0).all(-1)
        add("joint_n", both)
        mc, ac = m.clamp(min=0), aid.clamp(min=0)
        add("jmove@6", joint_hit(lm[:, 0], lm[:, 1], mc[:, 0], mc[:, 1], 6) & both)
        add("jaction@6", joint_hit(acts[:, 0], acts[:, 1], ac[:, 0], ac[:, 1], 6) & both)
        add("jaction@12", joint_hit(acts[:, 0], acts[:, 1], ac[:, 0], ac[:, 1], 12) & both)
        add("rating", data["rating"][rows].long())
        add("later", data["ser_glob"][rows, 0] > 0)
    return {k: torch.cat(v) for k, v in out.items()}


def report(name: str, r: dict[str, torch.Tensor], sel: torch.Tensor) -> str:
    sn, jn = r["slot_n"][sel].sum().item(), r["joint_n"][sel].sum().item()
    pct = lambda k, n: f"{r[k][sel].sum().item() / max(n, 1):6.1%}"  # noqa: E731
    return (f"{name:<22} {pct('move@1', sn)} {pct('move@2', sn)} {pct('action@1', sn)} "
            f"{pct('action@3', sn)} | {pct('jmove@6', jn)} {pct('jaction@6', jn)} {pct('jaction@12', jn)}"
            f"   ({sn} slots, {jn} joint)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="+", type=Path)
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--batch", type=int, default=1024)
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load(args.data)
    if "opp" not in data:
        raise SystemExit("this dataset has no opponent labels; rebuild it with ai_vgc.nn.dataset")
    val = (data["game"].long() % 1000) < args.val_frac * 1000
    idx = torch.nonzero(val & (data["opp"][..., 0] >= 0).any(-1))[:, 0]
    print(f"{len(idx)} held-out decisions with a labelled opponent action", flush=True)
    print(f"{'':<22} {'move':>6} {'move':>6} {'act':>6} {'act':>6} | {'jmove':>6} {'jact':>6} {'jact':>6}")
    print(f"{'':<22} {'@1':>6} {'@2':>6} {'@1':>6} {'@3':>6} | {'@6':>6} {'@6':>6} {'@12':>6}")
    for path in [None, *args.models]:
        model = None
        if path is not None:
            model = load_policy(path, str(dev))
            if not model.config.get("aux"):
                print(f"{path.name}: no opponent-action head, skipped")
                continue
        r = measure(model, data, idx, dev, args.batch)
        name = "uniform" if path is None else path.stem
        print(report(name, r, torch.ones_like(r["rating"], dtype=torch.bool)), flush=True)
        for label, lo, hi in BUCKETS:
            sel = (r["rating"] >= lo) & (r["rating"] < hi)
            if sel.any():
                print(report(f"  {label}", r, sel))
        # Games 2-3 with the series context (series.py) vs game 1 and series missing a game.
        for label, sel in (("  game 2+ (context)", r["later"]), ("  no context", ~r["later"])):
            if sel.any():
                print(report(label, r, sel))


if __name__ == "__main__":
    main()
