"""Collect whole teams from the open team sheets in human replays, for search's team-level guesses
at hidden sets under closed team sheets (`ai_vgc.nn.search.team_library` reads the output).

    uv run python scripts/team_library.py gen9championsvgc2026regmc

Team preview shows all six opposing species, and popular teams are shared pastes, so the six
together pin down sets far better than each species' usage alone. Reads
data/battle_logs/logs_<format>*.json (only games with |showteam| lines count) and writes
data/teams/team_library_<format>.json: [[[[species, item, ability, [moves], nature], ...], count], ...],
most common first. Sheets don't show stat points, so there are none.
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

from poke_env.data import to_id_str

fmt = sys.argv[1].removesuffix("bo3")
teams: Counter = Counter()
games = 0
for path in sorted(Path("data/battle_logs").glob(f"logs_{fmt}*.json")):
    for _, log in json.loads(path.read_text()).values():
        sheets = re.findall(r"^\|showteam\|p[12]\|(.*)$", log, re.M)
        games += bool(sheets)
        for sheet in sheets:
            team = []
            for mon in sheet.split("]"):
                f = mon.split("|")
                if len(f) < 6:
                    continue
                moves = tuple(sorted(to_id_str(m) for m in f[4].split(",") if m))
                team.append((to_id_str(f[1] or f[0]), to_id_str(f[2]), to_id_str(f[3]), moves, f[5] or ""))
            if len(team) == 6:
                teams[tuple(sorted(team))] += 1
out = [[[[sp, i, a, list(m), nat] for sp, i, a, m, nat in team], n] for team, n in teams.most_common()]
dest = Path(f"data/teams/team_library_{fmt}.json")
dest.write_text(json.dumps(out))
print(f"{dest}: {len(out)} distinct teams ({sum(teams.values())} sheets) from {games} games")
