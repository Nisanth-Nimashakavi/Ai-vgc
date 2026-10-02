"""Add up team_rank / team_test CSVs (one per shard) and sort them.

    uv run python scripts/team_sum.py data/team_ranks/392600/*.csv
    uv run python scripts/team_sum.py data/team_ranks/392600/*.csv --top 20 --copy-from data/teams/reg_mc \\
        --copy-to data/teams/reg_mc_top

team_rank rows are ranked teams (best first); team_test rows are opponent teams (worst matchups
first). --copy-to copies the --top teams' files (team_rank only) into their own folder, for
`player --teams`.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ai_vgc.showdown import wilson  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csvs", nargs="+", type=Path)
    ap.add_argument("--top", type=int, default=20, help="rows to print (and copy)")
    ap.add_argument("--copy-from", type=Path, default=None, help="folder with the ranked teams' .txt files")
    ap.add_argument("--copy-to", type=Path, default=None)
    args = ap.parse_args()

    rows: dict[str, dict] = {}
    key = None
    for path in args.csvs:
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                key = key or ("team" if "team" in r else "opponent_team")
                row = rows.setdefault(r[key], {"wins": 0, "games": 0, "mons": r.get("mons") or r.get("their_mons", "")})
                row["wins"] += int(r["wins"])
                row["games"] += int(r["games"])
    if not rows:
        raise SystemExit("no rows")
    ranking = key == "team"
    order = sorted(rows.items(), key=lambda kv: kv[1]["wins"] / max(kv[1]["games"], 1), reverse=ranking)
    wins, games = sum(r["wins"] for r in rows.values()), sum(r["games"] for r in rows.values())
    lo, hi = wilson(wins, games)
    what = "teams ranked" if ranking else "opponent teams (worst matchups first)"
    print(f"{len(rows)} {what}; {games} games, overall {wins / max(games, 1):.1%} [{lo:.1%}, {hi:.1%}]")
    for name, r in order[:args.top]:
        lo, hi = wilson(r["wins"], r["games"])
        print(f"  {r['wins'] / max(r['games'], 1):6.1%} [{lo:5.1%}, {hi:5.1%}] {r['wins']:>4}/{r['games']:<4} "
              f"{name}: {r['mons']}")
    if args.copy_to:
        if not (ranking and args.copy_from):
            raise SystemExit("--copy-to needs team_rank CSVs and --copy-from")
        args.copy_to.mkdir(parents=True, exist_ok=True)
        for name, _ in order[:args.top]:
            shutil.copy(args.copy_from / f"{name}.txt", args.copy_to / f"{name}.txt")
        print(f"copied the top {min(args.top, len(order))} to {args.copy_to}")


if __name__ == "__main__":
    main()
