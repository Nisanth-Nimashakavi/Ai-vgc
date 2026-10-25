"""Public Showdown Ladder & Challenge Runner for Model 19 with Dynamic Opponent Rebuilding & Depth-2 Search.

Usage:
  # Ladder 10 matches on public Showdown:
  python scripts/ladder_model19.py --games 10 --search --depth 2

  # Wait for challenges on Showdown:
  python scripts/ladder_model19.py --accept --search --depth 2
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import signal
import sys
import time
from typing import Any

import numpy as np
import torch
from poke_env import AccountConfiguration, ShowdownServerConfiguration, ServerConfiguration
from poke_env.data.normalize import to_id_str
from poke_env.player import Player
from poke_env.teambuilder import Teambuilder

from ai_vgc.nn.dynamic_team import DynamicOpponentBuilder, create_dynamic_preview_pool
from ai_vgc.nn.model19 import Transformer19, architecture_for_state
from ai_vgc.nn.player import _public_login
from ai_vgc.nn.search import _core
from ai_vgc.teams import RandomPoolTeambuilder

# Ensure project root and scripts dir are on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from eval_showdown_stateful_shadow import StatefulShadow19Player


def parse_args():
    p = argparse.ArgumentParser(description="Model 19 Showdown Ladder Runner")
    default_model = "data/models/model19_best_1054m_ema.pt"
    if not Path(default_model).exists() and Path("runs/heuristic19-blackwell-760236/main/model19_sp_1054343k_ema.pt").exists():
        default_model = "runs/heuristic19-blackwell-760236/main/model19_sp_1054343k_ema.pt"
    p.add_argument("--model-path", type=str, default=default_model,
                   help="Path to Model 19 checkpoint")
    p.add_argument("--team", type=str, default="data/teams/reg_mc_top_meta/MC371.txt",
                   help="Path to our Showdown team paste")
    p.add_argument("--teams-dir", type=Path, help="Random team per battle from this directory")
    p.add_argument("--seed", type=int, default=20261018)
    p.add_argument("--username", type=str, default=os.environ.get("BOT", "nimnimbot"),
                   help="Showdown username")
    p.add_argument("--password", type=str, default=os.environ.get("PS_PASSWORD", ""),
                   help="Showdown password")
    p.add_argument("--format", type=str, default="gen9championsvgc2026regmc",
                   help="Battle format")
    p.add_argument("--games", type=int, default=10,
                   help="Number of ladder games to play")
    p.add_argument("--accept", action="store_true",
                   help="Accept incoming challenges instead of queueing on ladder")
    p.add_argument("--search", action="store_true", default=True,
                   help="Enable simultaneous-move lookahead search (default True)")
    p.add_argument("--no-search", dest="search", action="store_false",
                   help="Disable search (use raw policy)")
    p.add_argument("--depth", type=int, default=1,
                   help="Search lookahead depth (1 or 2, default 1)")
    p.add_argument("--search-k", type=int, default=20,
                   help="Turn 1 candidate actions for our side")
    p.add_argument("--search-m", type=int, default=40,
                   help="Turn 1 candidate actions for opponent")
    p.add_argument("--search-k2", type=int, default=2,
                   help="Turn 2 candidate actions for our side")
    p.add_argument("--search-m2", type=int, default=2,
                   help="Turn 2 candidate actions for opponent")
    p.add_argument("--search-blend", type=float, default=0.8,
                   help="Blend weight for search equilibrium vs net prior")
    p.add_argument("--search-beta", type=float, default=15.0,
                   help="Search advantage multiplier beta for mixed move sampling (default 15.0)")
    p.add_argument("--search-gamma", type=float, default=0.91,
                   help="Prior exponent gamma for mixed move sampling (default 0.91)")
    p.add_argument("--deterministic", action="store_true", default=True,
                   help="Play argmax deterministic moves instead of sampling from mixed strategy (default True)")
    p.add_argument("--sample", dest="deterministic", action="store_false",
                   help="Sample moves stochastically from mixed strategy distribution")
    p.add_argument("--save-dir", type=str, default="data/ladder_model19/live_games",
                   help="Directory to save game logs and replay files")
    p.add_argument("--local", action="store_true",
                   help="Connect to local Showdown test server instead of public Showdown")
    p.add_argument("--port", type=int, default=18791,
                   help="Port for local server")
    p.add_argument("--device", type=str, default="auto",
                   help="PyTorch device: auto, cpu or cuda")
    return p.parse_args()


class LadderModel19Bot(StatefulShadow19Player):
    """Subclass of StatefulShadow19Player with ladder battle management and logging."""

    def __init__(self, *args, save_dir: Path | None = None,
                 rotating_builder=None, teams_dir=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_dir = save_dir
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
        self.timer = True
        self.games_history = []
        self.stop_after_match = False
        self.rotating_builder = rotating_builder
        self.teams_dir = teams_dir
        self.battle_teams = {}
        self.team = getattr(self, "_team", None)

    def _busy(self) -> bool:
        """True if any battle is currently active/unfinished."""
        return any(not b.finished for b in self._battles.values())

    async def _ladder(self, n_games: int):
        await self.ps_client.logged_in.wait()
        start_time = time.perf_counter()

        for _ in range(n_games):
            if self.stop_after_match:
                break
            async with self._battle_start_condition:
                await self.ps_client.search_ladder_game(self._format, self.next_team)
                # Wait for battle to start, periodically checking stop_after_match
                while not self.stop_after_match:
                    try:
                        await asyncio.wait_for(
                            self._battle_start_condition.wait(),
                            timeout=1.0,
                        )
                        break
                    except asyncio.TimeoutError:
                        continue
                if self.stop_after_match:
                    try:
                        await self.ps_client.send_message("/cancelsearch", "")
                    except Exception:
                        pass
                    break

                while self._battle_count_queue.full() and not self.stop_after_match:
                    async with self._battle_end_condition:
                        try:
                            await asyncio.wait_for(
                                self._battle_end_condition.wait(),
                                timeout=1.0,
                            )
                        except asyncio.TimeoutError:
                            continue
                if self.stop_after_match:
                    break
                await self._battle_semaphore.acquire()

            # Wait while active battle plays out
            while self._busy() and not self.stop_after_match:
                await asyncio.sleep(0.5)

        # Wait for any active battle that started before stop was requested
        try:
            await self._battle_count_queue.join()
        except Exception:
            pass

        self.logger.info(
            "Laddering (%d battles) finished in %fs",
            n_games,
            time.perf_counter() - start_time,
        )

    async def _accept_challenges(
        self,
        opponent: Any,
        n_challenges: int,
        packed_team: Any = None,
    ):
        if opponent:
            if isinstance(opponent, list):
                opponent = [to_id_str(o) for o in opponent]
            else:
                opponent = to_id_str(opponent)
        await self.ps_client.logged_in.wait()

        for _ in range(n_challenges):
            if self.stop_after_match:
                break
            team = packed_team or self.next_team
            while not self.stop_after_match:
                try:
                    raw_user = await asyncio.wait_for(self._challenge_queue.get(), timeout=1.0)
                    username = to_id_str(raw_user)
                except asyncio.TimeoutError:
                    continue
                if (
                    (opponent is None)
                    or (opponent == username)
                    or (isinstance(opponent, list) and (username in opponent))
                ):
                    await self.ps_client.accept_challenge(username, team)
                    await self._battle_semaphore.acquire()
                    break

        try:
            await self._battle_count_queue.join()
        except Exception:
            pass

    async def teampreview(self, battle: Any) -> str:
        if self.rotating_builder is not None:
            name = self.rotating_builder.last_name
            self.t_us = read_team_json(self.teams_dir / f"{name}.txt")
            self.battle_teams[battle.battle_tag] = name
            print(f"[Random Team] {name}", flush=True)
        tag = getattr(battle, "battle_tag", "")
        opp = getattr(battle, "opponent_username", "Unknown")
        opp_r = getattr(battle, "opponent_rating", "?")
        clean_tag = tag.lstrip(">")
        print(f"\n[Battle Started] Tag: {clean_tag} | Opponent: {opp} (Rating: {opp_r})", flush=True)
        print(f"  Live Battle URL: https://play.pokemonshowdown.com/{clean_tag}", flush=True)

        # Enforce Showdown battle timer & tactical inference flags
        battle._spd_on = True
        battle._dmg_on = True
        battle._guess = True
        if self.timer:
            try:
                await self.ps_client.send_message("/timer on", tag)
            except Exception:
                pass
        return super().teampreview(battle)

    def choose_move(self, battle: Any) -> Any:
        battle._spd_on = True
        battle._dmg_on = True
        battle._guess = True
        order = super().choose_move(battle)
        tag = getattr(battle, "battle_tag", "").lstrip(">")
        print(f"  [{tag} Turn {battle.turn}] Choice: {getattr(order, 'message', str(order))}", flush=True)
        return order

    def _battle_finished_callback(self, battle: Any):
        super()._battle_finished_callback(battle)
        tag = getattr(battle, "battle_tag", "")
        clean_id = tag.replace(">battle-", "").replace("battle-", "").replace(">", "")
        replay_url = f"https://replay.pokemonshowdown.com/{clean_id}"



        won = getattr(battle, "won", False)
        record = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "battle_tag": tag,
            "won": won,
            "turns": battle.turn,
            "opponent": battle.opponent_username,
            "rating": battle.rating,
            "opponent_rating": battle.opponent_rating,
            "replay_url": replay_url,
            "team": self.battle_teams.pop(tag, None),
            "search_diagnostics": dict(self.search_diagnostics),
        }
        self.games_history.append(record)
        print(f"[Match Completed] Opponent: {battle.opponent_username} | Result: {'WIN' if won else 'LOSS'} | Turns: {battle.turn} | Rating: {battle.rating} | Replay: {replay_url}\n", flush=True)

        if self.save_dir:
            hist_file = self.save_dir / "games.jsonl"
            battle.save_replay(self.save_dir / f"{clean_id}.html")
            with open(hist_file, "a") as f:
                f.write(json.dumps(record) + "\n")


def read_team_json(path):
    parsed = Teambuilder.parse_showdown_team(Path(path).read_text("utf-8"))
    return [dict(species=_core(to_id_str(m.species or m.nickname)), level=50, ability=m.ability or "",
                 item=m.item or "", nature=m.nature or "Serious", moves=m.moves,
                 evs=m.evs or {}, ivs=m.ivs or {}) for m in parsed]


def main():
    args = parse_args()
    torch.set_num_threads(4)

    # Check password from env if omitted
    password = args.password or os.environ.get("PS_PASSWORD", "")
    if not args.local and not password:
        env_file = Path.home() / ".config/ai-vgc/showdown.env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.startswith("export PS_PASSWORD=") or line.startswith("PS_PASSWORD="):
                    password = line.split("=", 1)[1].strip("\"' ")
                    break

    print("=" * 70)
    print("MODEL 19 SHOWDOWN LADDER RUNNER")
    print("=" * 70)
    print(f"Account:   {args.username}")
    print(f"Format:    {args.format}")
    print(f"Mode:      {'Accepting Challenges' if args.accept else f'Ladder ({args.games} games)'}")
    print(f"Model:     {args.model_path}")
    print(f"Search:    {args.search} (depth={args.depth}, k={args.search_k}, m={args.search_m}, k2={args.search_k2}, m2={args.search_m2})")
    print(f"Team:      {args.teams_dir or args.team}")
    print(f"Server:    {'Local (localhost:' + str(args.port) + ')' if args.local else 'Public Showdown'}")
    print("-" * 70)

    if args.teams_dir:
        tb = RandomPoolTeambuilder(args.teams_dir, seed=args.seed)
        tb._announce = True
        team_json = read_team_json(sorted(args.teams_dir.glob("*.txt"))[0])
    else:
        tb = RandomPoolTeambuilder(Path(args.team).parent, names=[Path(args.team).stem])
        team_json = read_team_json(args.team)

    # Load Model 19
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device in (None, "auto") else torch.device(args.device)
    print(f"Loading Model 19 checkpoint: {args.model_path} on {device}...")
    ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
    m19 = architecture_for_state(ckpt["model_state_dict"])().to(device)
    m19.load_state_dict({k.removeprefix("module."): v for k, v in ckpt["model_state_dict"].items()})
    m19.eval()

    # Dynamic opponent builder with meta priors from full reg_mc library
    teams_meta_dir = "data/teams/reg_mc" if Path("data/teams/reg_mc").exists() else str(Path(args.team).parent)
    dynamic_builder = DynamicOpponentBuilder(
        teams_dir=teams_meta_dir,
        fmt=args.format,
    )

    # Server configuration
    if args.local:
        config = ServerConfiguration(f"ws://localhost:{args.port}/showdown/websocket",
                                     "https://play.pokemonshowdown.com/action.php?")
    else:
        config = ShowdownServerConfiguration

    account_config = AccountConfiguration(args.username, password if not args.local else None)

    save_path = Path(args.save_dir)
    priors_file = "data/public-priors.json" if Path("data/public-priors.json").exists() else "runs/model19-beliefs-20261016/data/munchstats-priors.json"
    bot = LadderModel19Bot(
        model=m19,
        pool=None,
        t_us=team_json,
        t_them=None,
        sets_dir=teams_meta_dir,
        dynamic_opp=True,
        dynamic_builder=dynamic_builder,
        priors_path=priors_file if Path(priors_file).exists() else None,
        device=str(device),
        team=tb,
        account_configuration=account_config,
        battle_format=args.format,
        accept_open_team_sheet=False,
        max_concurrent_battles=1,
        server_configuration=config,
        use_search=args.search,
        search_depth=args.depth,
        search_k=args.search_k,
        search_m=args.search_m,
        search_k2=args.search_k2,
        search_m2=args.search_m2,
        search_blend=args.search_blend,
        search_beta=args.search_beta,
        search_gamma=args.search_gamma,
        deterministic=args.deterministic,
        rotating_builder=tb if args.teams_dir else None,
        teams_dir=args.teams_dir,
        save_dir=save_path,
        start_listening=args.local,
        ping_interval=20,
        ping_timeout=60,
    )

    # Signal handling for clean exit
    first_sigint = []

    def sig_handler(sig, frame):
        now = time.time()
        bot.stop_after_match = True

        if first_sigint and (now - first_sigint[0] > 0.5):
            print("\nForce exit requested, quitting immediately...", flush=True)
            try:
                loop = getattr(bot.ps_client, "loop", None)
                for b_tag, b in list(getattr(bot, "_battles", {}).items()):
                    if not b.finished and loop and loop.is_running():
                        asyncio.run_coroutine_threadsafe(
                            bot.ps_client.send_message("/forfeit", b_tag),
                            loop,
                        )
            except Exception:
                pass
            os._exit(130)

        first_sigint.append(now)

        if not bot._busy():
            print("\nCtrl-C received: no active battle, stopping immediately...", flush=True)
            try:
                loop = getattr(bot.ps_client, "loop", None)
                if loop and loop.is_running():
                    asyncio.run_coroutine_threadsafe(
                        bot.ps_client.send_message("/cancelsearch", ""),
                        loop,
                    )
            except Exception:
                pass
            os._exit(0)

        print("\nCtrl-C received: finishing current battle and shutting down (press Ctrl-C again to forfeit and quit immediately)...", flush=True)
        try:
            loop = getattr(bot.ps_client, "loop", None)
            if loop and loop.is_running():
                asyncio.run_coroutine_threadsafe(
                    bot.ps_client.send_message("/cancelsearch", ""),
                    loop,
                )
        except Exception:
            pass

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    if not args.local:
        print(f"Logging in to Showdown as {args.username}...")
        _public_login(bot, password)

    try:
        if args.accept:
            print("Listening for incoming challenges (Ctrl-C to stop)...", flush=True)
            asyncio.run(bot.accept_challenges(None, 100000))
        else:
            print(f"Starting ladder queue for {args.games} games...", flush=True)
            asyncio.run(bot.ladder(args.games))
    except KeyboardInterrupt:
        print("\nSession stopped by user.", flush=True)

    tot = bot.n_finished_battles
    w = bot.n_won_battles
    print("=" * 70)
    print(f"SESSION COMPLETE: {w} wins / {tot} finished battles ({w / max(1, tot) * 100:.1f}%)")
    print("=" * 70)


if __name__ == "__main__":
    main()
