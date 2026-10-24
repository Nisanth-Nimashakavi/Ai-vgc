"""Head-to-head evaluation benchmark for 17-token Transformer models and search configurations.

Allows comparing:
1. Transformer17 + 1-Turn Simultaneous Search vs Raw Transformer17 (no search)
2. Transformer17 + Search vs Heuristic Bot
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
import numpy as np
import torch

import vgc_engine
from ai_vgc.nn.model17 import Transformer17
from ai_vgc.nn.player17 import Player17


def load_teams(teams_file: str) -> list[str]:
    """Load JSON teams list."""
    path = Path(teams_file)
    if path.is_file():
        data = json.loads(path.read_text())
        if isinstance(data, list):
            # If it's a list of teams, serialize each team to JSON string
            return [json.dumps(team["pokemon"] if "pokemon" in team else team) for team in data]
    return []


SAMPLE_TEAMS = [
    # Team 1: Sun / Booster Fast Offense
    [
        {"species": "fluttermane", "item": "boosterenergy", "ability": "protosynthesis", "moves": ["moonblast", "shadowball", "dazzlinggleam", "protect"], "nature": "Timid", "evs": {"hp": 4, "spa": 252, "spe": 252}},
        {"species": "incineroar", "item": "sitrusberry", "ability": "intimidate", "moves": ["flareblitz", "knockoff", "partingshot", "fakeout"], "nature": "Careful", "evs": {"hp": 252, "def": 156, "spd": 100}},
        {"species": "urshifurapidstrike", "item": "choiceband", "ability": "unseenfist", "moves": ["surgingstrikes", "closecombat", "aquajet", "uturn"], "nature": "Jolly", "evs": {"atk": 252, "spe": 252, "hp": 4}},
        {"species": "rillaboom", "item": "assaultvest", "ability": "grassysurge", "moves": ["grassyglide", "woodhammer", "highhorsepower", "fakeout"], "nature": "Adamant", "evs": {"hp": 252, "atk": 252, "spd": 4}},
        {"species": "ogerponhearthflame", "item": "hearthflamemask", "ability": "moldbreaker", "moves": ["ivycudgel", "hornleech", "spikyshield", "followme"], "nature": "Jolly", "evs": {"atk": 252, "spe": 252, "hp": 4}},
        {"species": "landorustherian", "item": "choicescarf", "ability": "intimidate", "moves": ["stompingtantrum", "rockslide", "uturn", "earthquake"], "nature": "Adamant", "evs": {"atk": 252, "spe": 252, "hp": 4}}
    ],
    # Team 2: Tailwind Hyper Offense
    [
        {"species": "tornadus", "item": "covertcloak", "ability": "prankster", "moves": ["tailwind", "bleakwindstorm", "taunt", "protect"], "nature": "Timid", "evs": {"hp": 252, "spa": 4, "spe": 252}},
        {"species": "urshifurapidstrike", "item": "mysticwater", "ability": "unseenfist", "moves": ["surgingstrikes", "closecombat", "aquajet", "protect"], "nature": "Adamant", "evs": {"atk": 252, "spe": 252, "hp": 4}},
        {"species": "fluttermane", "item": "choicespecs", "ability": "protosynthesis", "moves": ["moonblast", "shadowball", "dazzlinggleam", "powergem"], "nature": "Timid", "evs": {"hp": 4, "spa": 252, "spe": 252}},
        {"species": "amoonguss", "item": "rockyhelmet", "ability": "regenerator", "moves": ["spore", "ragepowder", "pollenpuff", "clearsmog"], "nature": "Relaxed", "evs": {"hp": 252, "def": 156, "spd": 100}},
        {"species": "incineroar", "item": "safetygoggles", "ability": "intimidate", "moves": ["flareblitz", "knockoff", "partingshot", "fakeout"], "nature": "Careful", "evs": {"hp": 252, "def": 156, "spd": 100}},
        {"species": "kingambit", "item": "blackglasses", "ability": "defiant", "moves": ["kowtowcleave", "suckerpunch", "ironhead", "protect"], "nature": "Adamant", "evs": {"hp": 252, "atk": 252, "def": 4}}
    ]
]


def play_game(
    p1: Player17,
    p2: Player17,
    team1_raw: list[dict],
    team2_raw: list[dict],
    max_turns: int = 30,
) -> tuple[int, int]:
    """Play a full battle between p1 and p2.

    Returns (winner, turns): +1 for p1 win, -1 for p2 win, 0 for tie.
    """
    t1_json = json.dumps(team1_raw)
    t2_json = json.dumps(team2_raw)

    # 1. Preview phase: load 6-mon teams
    preview_b = vgc_engine.Battle.from_teams(t1_json, t2_json, format="gen9championsvgc2026regmc", stat_points=True)
    picks_1 = p1.choose_preview(preview_b, 0)
    picks_2 = p2.choose_preview(preview_b, 1)

    # 2. Select chosen 4 for each side
    chosen_1 = [team1_raw[i] for i in picks_1]
    chosen_2 = [team2_raw[i] for i in picks_2]

    # 3. Start actual 4v4 battle
    battle = vgc_engine.Battle.from_teams(
        json.dumps(chosen_1), json.dumps(chosen_2), format="gen9championsvgc2026regmc", stat_points=True
    )

    for turn in range(1, max_turns + 1):
        c1 = p1.choose_turn(battle, 0)
        c2 = p2.choose_turn(battle, 1)

        res = battle.step_move(c1, c2)
        if res == "p1_win":
            return (1, turn)
        elif res == "p2_win":
            return (-1, turn)
        elif res == "tie":
            return (0, turn)

    # Turns exhausted
    return (0, max_turns)


def evaluate_matchup(
    ckpt_path: str,
    n_games: int = 10,
    device: str = "cpu",
):
    """Benchmark Search (Player 1) vs Raw Policy (Player 2)."""
    print(f"Loading model checkpoint from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location=device)
    model = Transformer17()
    model.load_state_dict(ckpt["model"])
    model.eval()

    p1_search = Player17(model, use_search=True, k=6, m=5, blend_alpha=0.5, device=device)
    p2_raw = Player17(model, use_search=False, device=device)

    p1_wins = 0
    p2_wins = 0
    ties = 0

    print(f"\nStarting {n_games} games: Player 1 (Search k=6, m=5) vs Player 2 (Raw Policy)...")
    t0 = time.time()

    for g in range(1, n_games + 1):
        # Alternate sides to remove side advantage
        if g % 2 == 1:
            team1, team2 = SAMPLE_TEAMS[0], SAMPLE_TEAMS[1]
            winner, turns = play_game(p1_search, p2_raw, team1, team2)
            if winner == 1:
                p1_wins += 1
                res_str = "Search (P1) WON"
            elif winner == -1:
                p2_wins += 1
                res_str = "Raw (P2) WON"
            else:
                ties += 1
                res_str = "TIE"
        else:
            team1, team2 = SAMPLE_TEAMS[1], SAMPLE_TEAMS[0]
            winner, turns = play_game(p2_raw, p1_search, team1, team2)
            if winner == -1:
                p1_wins += 1
                res_str = "Search (P2) WON"
            elif winner == 1:
                p2_wins += 1
                res_str = "Raw (P1) WON"
            else:
                ties += 1
                res_str = "TIE"

        print(f"  Game {g:2d}/{n_games}: {res_str} in {turns} turns (Search wins: {p1_wins}, Raw wins: {p2_wins})")

    elapsed = time.time() - t0
    winrate = p1_wins / max(p1_wins + p2_wins, 1) * 100
    print("\n" + "=" * 50)
    print(f"EVALUATION RESULTS ({n_games} games, {elapsed:.1f}s):")
    print(f"  Search Wins:   {p1_wins}")
    print(f"  Raw Wins:      {p2_wins}")
    print(f"  Ties:          {ties}")
    print(f"  Search Winrate: {winrate:.1f}%")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="data/models/model17_imitation_1m.pt")
    parser.add_argument("--games", type=int, default=6)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    evaluate_matchup(args.model, n_games=args.games, device=args.device)
