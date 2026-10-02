"""How humans change team preview between games of a Bo3, from the scraped logs:
    python scripts/human_bo3.py data/battle_logs/logs_*.json
Per player and game N > 1 of a series: did they lead the same pair / bring the same four as
in game N - 1, split by whether they won or lost that game.
"""
import json
import re
import sys
from collections import defaultdict

GAME = re.compile(r'Game (\d)</strong> of <a href="/(game-bestof\d-[^"]+)"')

series = defaultdict(dict)  # series id -> game number -> {player: (leads, four), "winner": name}
for path in sys.argv[1:]:
    for _, (_, log) in json.load(open(path)).items():
        g = GAME.search(log)
        if not g:
            continue
        names, leads, four, winner = {}, defaultdict(list), defaultdict(set), None
        for line in log.split("\n"):
            p = line.split("|")
            if len(p) > 3 and p[1] == "player" and p[3]:
                names[p[2]] = p[3]
            elif len(p) > 3 and p[1] in ("switch", "drag") and p[2][:2] in ("p1", "p2"):
                side, sp = p[2][:2], p[3].split(",")[0].split("-Mega")[0]
                four[side].add(sp)
                if len(leads[side]) < 2 and "|turn|1" not in log[:log.find(line)]:
                    leads[side].append(sp)
            elif len(p) > 2 and p[1] == "win":
                winner = p[2]
        if winner and len(names) == 2:
            series[g.group(2)][int(g.group(1))] = {
                "winner": winner, **{names[s]: (frozenset(leads[s]), frozenset(four[s])) for s in names}}

stats = defaultdict(lambda: [0, 0])
for games in series.values():
    for n in sorted(games):
        if n - 1 not in games:
            continue
        prev, cur = games[n - 1], games[n]
        for name in (k for k in cur if k != "winner"):
            if name not in prev:
                continue
            res = "after win" if prev["winner"] == name else "after loss"
            for what, i in (("same leads", 0), ("same four", 1)):
                s = stats[f"{what} {res}"]
                s[0] += prev[name][i] == cur[name][i]
                s[1] += 1
print(f"{len(series)} series")
for k, (a, n) in sorted(stats.items()):
    print(f"  {k}: {a}/{n} = {a / max(n, 1):.0%}")

# Does the game N - 1 loser win game N more often when they change leads?
comeback = defaultdict(lambda: [0, 0])
for games in series.values():
    for n in sorted(games):
        if n - 1 not in games:
            continue
        prev, cur = games[n - 1], games[n]
        loser = next((k for k in prev if k not in ("winner", prev["winner"])), None)
        if loser not in cur or prev["winner"] not in cur:
            continue
        changed = cur[loser][0] != prev[loser][0]
        winner_same = cur[prev["winner"]][0] == prev[prev["winner"]][0]
        key = f"loser {'changed' if changed else 'kept'} leads, winner {'kept' if winner_same else 'changed'}"
        comeback[key][0] += cur["winner"] == loser
        comeback[key][1] += 1
        comeback["all"][0] += cur["winner"] == loser
        comeback["all"][1] += 1
print("previous game's loser wins the next:")
for k, (a, n) in sorted(comeback.items()):
    print(f"  {k}: {a}/{n} = {a / max(n, 1):.1%}")
