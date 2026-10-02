"""Count the sets people run, from the open team sheets in human replays, for search's guesses at
hidden sets under closed team sheets (`ai_vgc.nn.search.set_counts` reads the output).

    uv run python scripts/set_usage.py gen9championsvgc2026regmc

Reads data/battle_logs/logs_<format>*.json (Bo1 and Bo3; only games with |showteam| lines count)
and writes data/teams/set_usage_<format>.json: {species: [[item, ability, [moves], count], ...]}.
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

from poke_env.data import to_id_str

fmt = sys.argv[1].removesuffix("bo3")
counts: dict[str, Counter] = {}
games = 0
for path in sorted(Path("data/battle_logs").glob(f"logs_{fmt}*.json")):
    for _, log in json.loads(path.read_text()).values():
        sheets = re.findall(r"^\|showteam\|p[12]\|(.*)$", log, re.M)
        games += bool(sheets)
        for sheet in sheets:
            for mon in sheet.split("]"):
                f = mon.split("|")
                if len(f) < 5:
                    continue
                species = to_id_str(f[1] or f[0])
                moves = tuple(sorted(to_id_str(m) for m in f[4].split(",") if m))
                counts.setdefault(species, Counter())[(to_id_str(f[2]), to_id_str(f[3]), moves)] += 1
out = {sp: [[i, a, list(m), n] for (i, a, m), n in c.most_common()] for sp, c in sorted(counts.items())}
dest = Path(f"data/teams/set_usage_{fmt}.json")
dest.write_text(json.dumps(out))
print(f"{dest}: {sum(sum(c.values()) for c in counts.values())} sets of {len(out)} species from {games} games")
