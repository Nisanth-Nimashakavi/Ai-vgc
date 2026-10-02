"""Team preview as a matrix game (idea 6): score each side's candidate picks by playing them out.

    uv run python -m ai_vgc.nn.preview --model data/models/archive/mb-bo3-do-v1.pt --matchups 20
    uv run python -m ai_vgc.nn.preview --format gen9championsvgc2026regmc --port 8001 \\
        --showdown pokemon-showdown-mc --teams data/teams/reg_mc

For each matchup (two random teams from the pool):

1. A probe game lists each side's candidates, then forfeits: the policy's own preview first,
   then the most likely others by the policy's probability (ordered leads x unordered back
   pair, 180 in all), `--k` in total.
2. Every pair of candidates plays `--games` games with the policy doing the battling,
   which fills a k x k win-rate matrix for side A.
3. The matrix is solved for A's maximin mix (a linear program).
4. Fresh games measure what the solved mix is worth: A playing the mix vs B's normal
   policy preview, against A's normal policy preview vs the same B.

The gap in step 4, over many matchups, is what solving preview would add. If it is worth
having, the solved mixes become training targets for the preview head.
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from poke_env import ServerConfiguration
from poke_env.battle import DoubleBattle
from scipy.optimize import linprog

from ai_vgc.nn.encode import encode
from ai_vgc.nn.model import Policy, masked, slot_b_mask
from ai_vgc.nn.player import NNPlayer, load_policy


@torch.no_grad()
def _slot_logps(model: Policy, battle: DoubleBattle, rating: float) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Slot a's log-probs over team members, and slot b's given each slot-a choice."""
    obs = encode(battle, rating)
    b = {k: torch.from_numpy(np.ascontiguousarray(v))[None] for k, v in obs.items()}
    m = torch.from_numpy(obs["mask"])
    enc = model.encode(b)
    lp_a = F.log_softmax(masked(model.slot_logits(enc, b, 0), m[None, 0]), -1)[0].numpy()
    lp_b = {}
    for a0 in np.flatnonzero(obs["mask"][0]):
        prev = torch.tensor([a0])
        lp_b[a0] = F.log_softmax(masked(model.slot_logits(enc, b, 1, prev), slot_b_mask(m[None, 1], prev)),
                                 -1)[0].numpy()
    return lp_a, lp_b


def preview_candidates(model: Policy, battle: DoubleBattle, rating: float) -> list[tuple[str, float]]:
    """Every preview ("1234": ordered leads, then the back two) with the policy's log-prob,
    most likely first."""
    team = list(battle.team.values())
    for mon in team:
        mon._selected_in_teampreview = False
    lp_a, lp_b = _slot_logps(model, battle, rating)
    out = []
    for i, j in itertools.permutations(range(1, 7), 2):
        lead = lp_a[i] + lp_b[i][j]
        for mon in team:
            mon._selected_in_teampreview = False
        team[i - 1]._selected_in_teampreview = team[j - 1]._selected_in_teampreview = True
        back_a, back_b = _slot_logps(model, battle, rating)
        rest = [k for k in range(1, 7) if k not in (i, j)]
        for k, l in itertools.combinations(rest, 2):
            # The back pair is unordered: add up both orders.
            back = np.logaddexp(back_a[k] + back_b[k][l], back_a[l] + back_b[l][k])
            out.append((f"{i}{j}{k}{l}", float(lead + back)))
    for mon in team:
        mon._selected_in_teampreview = False
    return sorted(out, key=lambda c: -c[1])


class PreviewPlayer(NNPlayer):
    """NNPlayer whose preview is drawn from a fixed mix of orders ("1234" -> probability).
    With `probe_k`, it instead records its top candidates and forfeits."""

    def __init__(self, *args, mix: dict[str, float] | None = None, probe_k: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.mix = mix
        self.probe_k = probe_k
        self.candidates: list[str] = []

    async def _choose_preview(self, battle: DoubleBattle) -> str:
        if self.probe_k:
            # The policy's own (slot-by-slot greedy) pick first, so the matrix holds the baseline.
            own = (await super()._choose_preview(battle)).removeprefix("/team ")
            own = own[:2] + "".join(sorted(own[2:]))
            ranked = [o for o, _ in preview_candidates(self.model, battle, self.rating) if o != own]
            self.candidates = [own, *ranked[:self.probe_k - 1]]
            return "/forfeit"
        if self.mix is None:
            return await super()._choose_preview(battle)
        orders, probs = zip(*self.mix.items())
        return "/team " + random.choices(orders, probs)[0]


def maximin(M: np.ndarray) -> tuple[np.ndarray, float]:
    """Row player's maximin mix and value for the win-rate matrix M."""
    k, n = M.shape
    # Variables: x (k), v. Maximise v s.t. M^T x >= v, sum x = 1, x >= 0.
    c = np.zeros(k + 1)
    c[-1] = -1
    A_ub = np.hstack([-M.T, np.ones((n, 1))])
    res = linprog(c, A_ub=A_ub, b_ub=np.zeros(n), A_eq=[[1] * k + [0]], b_eq=[1],
                  bounds=[(0, None)] * k + [(None, None)], method="highs")
    return res.x[:k], res.x[-1]


async def _play(pairs: list[tuple[NNPlayer, NNPlayer]], n: int) -> list[float]:
    """Win rate of each pair's first player over n games."""
    await asyncio.gather(*(a.battle_against(b, n_battles=n) for a, b in pairs))
    rates = [a.n_won_battles / max(a.n_finished_battles, 1) for a, _ in pairs]
    for a, b in pairs:
        await a.ps_client.stop_listening()
        await b.ps_client.stop_listening()
    return rates


def main() -> None:
    from ai_vgc.showdown import account, ensure_server, wilson
    from ai_vgc.teams import TEAMS_DIR

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="data/models/archive/mb-bo3-do-v1.pt")
    ap.add_argument("--matchups", type=int, default=20)
    ap.add_argument("--k", type=int, default=6, help="candidate previews per side")
    ap.add_argument("--games", type=int, default=16, help="games per matrix cell")
    ap.add_argument("--eval-games", type=int, default=64, help="games per side of the step-4 comparison")
    ap.add_argument("--format", default="gen9championsvgc2026regmb")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--showdown", type=Path, default=None)
    ap.add_argument("--teams", type=Path, default=TEAMS_DIR)
    ap.add_argument("--rating", type=float, default=1700)
    ap.add_argument("--concurrency", type=int, default=4, help="battles at once per player pair")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.set_num_threads(1)
    random.seed(args.seed)

    proc = ensure_server(args.port, args.showdown)
    model = load_policy(args.model)
    teams = [p.read_text() for p in sorted(args.teams.glob("*.txt"))]
    common = dict(
        battle_format=args.format, max_concurrent_battles=args.concurrency, accept_open_team_sheet=True,
        log_level=40,
        server_configuration=ServerConfiguration(f"ws://localhost:{args.port}/showdown/websocket",
                                                 "https://play.pokemonshowdown.com/action.php?"),
    )

    def player(team: str, **kw) -> PreviewPlayer:
        return PreviewPlayer(model, args.rating, greedy=True, account_configuration=account("pv"),
                             team=team, **common, **kw)

    solved = policy = 0.0
    try:
        for mi in range(args.matchups):
            t = time.time()
            ta, tb = random.sample(teams, 2)
            # 1. Probe both sides' candidates.
            pa, pb = player(ta, probe_k=args.k), player(tb, probe_k=args.k)
            asyncio.run(_play([(pa, pb)], 1))
            ca, cb = pa.candidates, pb.candidates
            # 2. Fill the matrix.
            cells = list(itertools.product(range(len(ca)), range(len(cb))))
            rates = asyncio.run(_play([(player(ta, mix={ca[i]: 1}), player(tb, mix={cb[j]: 1}))
                                       for i, j in cells], args.games))
            M = np.zeros((len(ca), len(cb)))
            for (i, j), r in zip(cells, rates):
                M[i, j] = r
            # 3. Solve.
            x, v = maximin(M)
            mix = {ca[i]: float(p) for i, p in enumerate(x) if p > 1e-6}
            # 4. Fresh games: solved mix vs policy preview, both against B's policy preview.
            s, p = asyncio.run(_play([(player(ta, mix=mix), player(tb)),
                                      (player(ta), player(tb))], args.eval_games))
            solved += s
            policy += p
            top = ", ".join(f"{o} {q:.2f}" for o, q in sorted(mix.items(), key=lambda z: -z[1])[:3])
            print(f"matchup {mi + 1}: policy pick vs B's mix {M[0].mean():.2f}, maximin value {v:.2f} ({top}) | "
                  f"fresh: solved {s:.1%} vs policy {p:.1%} | {time.time() - t:.0f}s", flush=True)
        n = args.matchups * args.eval_games
        for name, rate in (("solved mix", solved), ("policy preview", policy)):
            lo, hi = wilson(round(rate * args.eval_games), n)
            print(f"{name}: {rate / args.matchups:.1%}  95% CI [{lo:.1%}, {hi:.1%}]  over {n} games")
    finally:
        if proc:
            proc.terminate()


if __name__ == "__main__":
    main()
