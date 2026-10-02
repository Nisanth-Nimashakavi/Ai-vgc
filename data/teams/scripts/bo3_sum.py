"""Add up the per-copy lines of a Bo3 run (scripts/slurm/nn_bo3.sh):
    python scripts/bo3_sum.py logs/vgc-nn-bo3-<job>.out
"""
import re
import sys

tot = {}
for line in open(sys.argv[1]):
    for key, pat in [("games", r": (\d+)/(\d+) = "), ("series", r"series (\d+)/(\d+)"),
                     ("game 1", r"game 1 (\d+)/(\d+)"), ("game 2", r"game 2 (\d+)/(\d+)"),
                     ("game 3", r"game 3 (\d+)/(\d+)")]:
        m = re.search(pat, line)
        if m and (key != "games" or " vs " in line):
            w, n = tot.get(key, (0, 0))
            tot[key] = (w + int(m.group(1)), n + int(m.group(2)))
print("  ".join(f"{k} {w}/{n} = {w / max(n, 1):.1%}" for k, (w, n) in tot.items()))
