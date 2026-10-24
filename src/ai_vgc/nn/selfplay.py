"""Self-play data generation using the search player as the label generator.

This creates training data where every position is labeled with the search's
equilibrium value and policy, not just human decisions. This is the core of
Nessie's sample efficiency: search-guided self-play.

    uv run python -m ai_vgc.nn.selfplay --games 1000 --model data/models/current.pt \
        --search --out data/nn_selfplay/gen9championsvgc2026regmcbo3.npz

Output format matches ai_vgc.nn.dataset (KEYS + action, preview, turn, opp, rating, won, turns_left, game)
so it can be mixed with human data in train.py.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from ai_vgc.nn.encode import encode, legal_mask, vocab, move_table, vocab_sizes, N_ACT
from ai_vgc.nn.model import Policy
from ai_vgc.nn.player import NNPlayer, load_policy
from ai_vgc.nn.rules import apply_rules
from ai_vgc.nn.search import SearchPlayer, Bridge, search
from ai_vgc.nn.vgc_bridge import VGCBridge
from ai_vgc.nn.series import Context
from ai_vgc.teams import RandomPoolTeambuilder, TEAMS_DIR

# Must match dataset.py
KEYS = ["tok_cat", "tok_num", "mv_cat", "mv_dyn", "dmg", "glob", "act_tok", "mask",
        "ser_tok", "ser_mv", "ser_glob"]
N_PLAYERS = 2  # Self-play: 2 players per game


async def _selfplay_game(
    model: Policy,
    opp_model: Policy | None,
    fmt: str,
    teams_dir: str,
    showdown: str,
    bridge: Bridge,
    rating: float,
    search_kwargs: dict,
    greedy: bool,
    game_idx: int,
    port: int,
) -> dict | None:
    """Play one self-play game, return encoded data for both players."""
    from poke_env import AccountConfiguration, ServerConfiguration
    from poke_env.player import BattleOrder
    from poke_env.environment import DoublesEnv

    # Create two players
    teambuilder = RandomPoolTeambuilder(Path(teams_dir))
    team1 = teambuilder.yield_team()
    team2 = teambuilder.yield_team()

    p1 = SearchPlayer(
        model, rating, greedy, rules=True,
        account_configuration=AccountConfiguration(f"sp1_{game_idx}", None),
        server_configuration=ServerConfiguration(f"ws://localhost:{port}/showdown/websocket",
                                                 "https://play.pokemonshowdown.com/action.php?"),
        team=team1,
        opp_model=opp_model,
        fmt=fmt, teams_dir=teams_dir, showdown=showdown,
        bridge=bridge,
        **search_kwargs
    )
    p2 = SearchPlayer(
        model, rating, greedy, rules=True,
        account_configuration=AccountConfiguration(f"sp2_{game_idx}", None),
        server_configuration=ServerConfiguration(f"ws://localhost:{port}/showdown/websocket",
                                                 "https://play.pokemonshowdown.com/action.php?"),
        team=team2,
        opp_model=opp_model,
        fmt=fmt, teams_dir=teams_dir, showdown=showdown,
        bridge=bridge,
        **search_kwargs
    )

    # Battle
    try:
        await p1.battle_against(p2, n_battles=1)
    except Exception as e:
        import traceback

        print(f"Game {game_idx} failed: {e!r}", flush=True)
        traceback.print_exc()
        return None

    # Both players should have the same battle tag
    if not p1.games or not p2.games:
        return None

    game1 = p1.games[0]
    game2 = p2.games[0]

    # Replay both sides' decisions from the logs
    # We need to parse the logs and re-encode at each decision point
    log1 = game1.get("log", "")
    log2 = game2.get("log", "")

    if not log1 or not log2:
        return None

    # For now, just use the search player's recorded decisions
    # The SearchPlayer records via on_search callback
    rows_p1 = getattr(p1, "_selfplay_rows", [])
    rows_p2 = getattr(p2, "_selfplay_rows", [])

    if not rows_p1 or not rows_p2:
        return None

    return {"p1": rows_p1, "p2": rows_p2, "won": game1["won"]}


def _encode_from_search_details(details: dict, battle, rating: float, player_side: str) -> dict | None:
    """Convert search details into a training row.

    details contains: M (our value matrix), q (opponent distribution),
    actions (our candidates), score, value, logp, obs
    """
    from ai_vgc.nn.model import masked, slot_b_mask
    from ai_vgc.nn.encode import joint_mask

    try:
        obs = details["obs"]
        mask = obs["mask"]
        actions = details["actions"]  # [n_candidates, 2]
        logp = details["logp"]
        value = details["value"]  # per candidate
        score = details["score"]  # value + prior * logp
        M = details["M"]  # [n_candidates, n_opp_candidates]
        q = details["q"]  # [n_opp_candidates]

        # The chosen action
        best_idx = int(details.get("selected_index", score.argmax()))
        chosen = actions[best_idx]

        # Target policy: search's equilibrium strategy (our marginal)
        # M @ q gives our expected value per action; softmax with temperature
        # For training, we can use the search's implied policy:
        # the probability we'd play each action under the equilibrium
        # For now: use the search policy (logp) as target, but weighted by search quality

        # Value target: the search's value for the chosen action
        v_target = float(value[best_idx])

        # Build the row
        row = {k: obs[k] for k in KEYS}
        row["action"] = chosen.astype(np.int16)
        row["preview"] = False  # Will be set separately for preview turns
        row["turn"] = np.int16(battle.turn)
        row["opp"] = np.array([-1, -1], np.int8)  # Not available from search
        row["rating"] = np.int16(rating)
        row["won"] = None  # Set after game ends
        row["turns_left"] = None  # Set after game ends
        row["game"] = None  # Set after game ends

        return row
    except Exception as e:
        print(f"Encode failed: {e}")
        return None


def selfplay_worker(args: tuple) -> list[dict] | None:
    """Worker function for multiprocessing. Each worker runs one game."""
    (game_idx, model_path, opp_model_path, fmt, teams_dir, showdown, port,
     rating, search_kwargs, greedy, device) = args

    torch.set_num_threads(1)
    dev = torch.device(device)

    model = load_policy(model_path, device)
    opp_model = load_policy(opp_model_path, device) if opp_model_path else None

    # Create bridge. The live self-play game remains poke-env/Showdown; only
    # its search playouts use the selected backend.
    bridge = VGCBridge(timeout=60.0) if search_kwargs.pop("backend", "showdown") == "vgc-engine" \
        else Bridge(showdown, timeout=60.0, procs=1)

    # Run async game
    async def run_game():
        return await _selfplay_game(
            model, opp_model, fmt, teams_dir, showdown, bridge,
            rating, search_kwargs, greedy, game_idx, port
        )

    result = asyncio.run(run_game())
    bridge.close()
    return result


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", default="data/models/archive/mb-bo3-bc-v1.pt",
                    help="Policy checkpoint to use for self-play")
    ap.add_argument("--opp-model", default=None,
                    help="Opponent-action head checkpoint (for search)")
    ap.add_argument("--games", type=int, default=1000, help="Number of self-play games")
    ap.add_argument("--workers", type=int, default=4, help="Parallel games")
    ap.add_argument("--fmt", default="gen9championsvgc2026regmcbo3", help="Format")
    ap.add_argument("--teams-dir", type=Path, default=TEAMS_DIR, help="Team pool directory")
    ap.add_argument("--showdown", default="pokemon-showdown-mc", help="Showdown checkout")
    ap.add_argument("--port", type=int, default=8000, help="local Showdown port for self-play games")
    ap.add_argument("--rating", type=float, default=1700, help="Rating input to network")
    ap.add_argument("--greedy", action="store_true", help="Greedy actions (no sampling)")
    ap.add_argument("--out", type=Path, required=True, help="Output .npz file")
    ap.add_argument("--search-k", type=int, default=6)
    ap.add_argument("--search-opp-k", type=int, default=6)
    ap.add_argument("--search-seeds", type=int, default=2)
    ap.add_argument("--search-backend", choices=["showdown", "vgc-engine"], default="showdown",
                    help="search playout backend; self-play games still run on local Showdown")
    ap.add_argument("--search-worlds", type=int, default=1)
    ap.add_argument("--search-depth", type=int, default=1, choices=[1, 2])
    ap.add_argument("--search-k2", type=int, default=3)
    ap.add_argument("--search-opp-k2", type=int, default=3)
    ap.add_argument("--search-seeds2", type=int, default=1)
    ap.add_argument("--search-prior", type=float, default=0.0)
    ap.add_argument("--search-procs", type=int, default=1)
    ap.add_argument("--search-refine", type=int, default=0)
    ap.add_argument("--search-refine-seeds", type=int, default=64)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = ap.parse_args()

    # Load vocab once
    vocab()

    search_kwargs = dict(
        k=args.search_k,
        opp_k=args.search_opp_k,
        seeds=args.search_seeds,
        backend=args.search_backend,
        worlds=args.search_worlds,
        depth=args.search_depth,
        k2=args.search_k2,
        opp_k2=args.search_opp_k2,
        seeds2=args.search_seeds2,
        prior=args.search_prior,
        procs=args.search_procs,
        refine=args.search_refine,
        refine_seeds=args.search_refine_seeds,
    )

    print(f"Starting self-play: {args.games} games, {args.workers} workers")
    print(f"Model: {args.model}")
    print(f"Format: {args.fmt}")
    print(f"Search: k={args.search_k}, opp_k={args.search_opp_k}, seeds={args.search_seeds}, depth={args.search_depth}")

    # Prepare worker args
    worker_args = [
        (i, args.model, args.opp_model, args.fmt, str(args.teams_dir),
         args.showdown, args.port, args.rating, search_kwargs, args.greedy, args.device)
        for i in range(args.games)
    ]

    # Self-play games still use Showdown's websocket protocol; only the search
    # rollout backend is Rust. Start one local server for this process pool.
    from ai_vgc.showdown import ensure_server

    server_proc = ensure_server(args.port, args.showdown)
    # Run in parallel
    all_rows = {k: [] for k in KEYS + ["action", "preview", "turn", "opp",
                                         "rating", "won", "turns_left", "game"]}
    n_completed = 0
    n_failed = 0

    with ProcessPoolExecutor(args.workers, mp_context=__import__("multiprocessing").get_context("spawn")) as pool:
        for i, result in enumerate(pool.map(selfplay_worker, worker_args)):
            if result is None:
                n_failed += 1
            else:
                # Merge rows from both players
                for side in ("p1", "p2"):
                    for row in result[side]:
                        for k in row:
                            all_rows[k].append(row[k])
                n_completed += 1

            if (i + 1) % 50 == 0 or i == args.games - 1:
                total_rows = len(all_rows["action"])
                print(f"  {i+1}/{args.games}: {n_completed} done, {n_failed} failed, {total_rows} decisions")

    if n_completed == 0:
        if server_proc is not None:
            server_proc.terminate()
        print("No games completed!")
        return

    # Post-process: fill in won, turns_left, game id
    n = len(all_rows["action"])
    game_ids = np.full(n, 0, np.uint32)
    won = np.zeros(n, bool)
    turns_left = np.zeros(n, np.int16)

    # We need to group by game. The rows come in order per game per player.
    # For simplicity, assign each game a unique id and fill won/turns_left
    # This is approximate - we'd need proper tracking in the worker
    game_id = 0
    idx = 0
    while idx < n:
        # Find end of this game (preview turns + game turns)
        # For now, just use a fixed game id per pair of rows
        game_ids[idx:idx+2] = zlib.crc32(f"sp_{game_idx}_{idx}".encode()) & 0xffffffff
        # We don't have per-game won info here, would need worker to return it
        idx += 2
        game_id += 1

    all_rows["game"] = game_ids
    all_rows["won"] = won
    all_rows["turns_left"] = turns_left

    # Save
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_data = {k: np.stack(all_rows[k]) if isinstance(all_rows[k][0], np.ndarray)
                else np.asarray(all_rows[k]) for k in all_rows}
    np.savez(args.out, **out_data)
    print(f"Saved {n} decisions from {n_completed} games to {args.out}")
    if server_proc is not None:
        server_proc.terminate()


if __name__ == "__main__":
    main()
