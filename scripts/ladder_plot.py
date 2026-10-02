"""Plot ladder rating by team: python scripts/ladder_plot.py [format] [out.png]"""
import json, re, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

fmt = sys.argv[1] if len(sys.argv) > 1 else "regmc"
out = sys.argv[2] if len(sys.argv) > 2 else "docs/ladder_by_team.png"
rows = []
for f in sorted(Path("data/live_games").glob("*/games.jsonl")):
    for line in f.open():
        g = json.loads(line)
        log = g["log"]
        if "|rated|" not in log or g["format"].replace("gen9championsvgc2026", "") != fmt:
            continue
        me = g.get("user") or "nimnimbot"
        ch = re.search(rf"\|raw\|{re.escape(me)}'s rating: (\d+) &rarr; <strong>(\d+)", log)
        mine = re.search(rf"\|player\|p\d\|{re.escape(me)}\|[^|]*\|(\d+)", log)
        rows.append({"time": pd.Timestamp(g["time"]), "won": g["result"] == "win",
                     "before": int(ch[1]) if ch else (int(mine[1]) if mine else None),
                     "after": int(ch[2]) if ch else None,
                     "model": g.get("model") or "general",
                     "team": g.get("team") or "older (untagged)"})
d = pd.DataFrame(rows).sort_values("time").reset_index(drop=True)
d["after"] = d["after"].fillna(d["before"].shift(-1))
d = d.dropna(subset=["after"])
teams = [t for t, n in d.team.value_counts().items() if n >= 5]
# Commands as typed (from the user); other teams were laddered with older general models.
CMDS = {
    "MC301": "scripts/bot.sh --ladder --cts --use MC301 --model data/models/mc-cts-rnad-v7-MC301.pt "
             "--opp-model data/models/mc-cts-opp-v6.pt --search-team-mass 0.8 --search-depth 2 --watch "
             "--set-guess --damage-inference",
    "MC378": "scripts/bot.sh --ladder 25 --cts --use MC378 --model data/models/mc-cts-rnad-v7-MC378.pt "
             "--opp-model data/models/mc-cts-opp-v6.pt --search-team-mass 0.8 --search-depth 2 --watch",
}
SHORT = {"MC301": "v7, depth 2, +set-guess +damage-inf", "MC378": "v7, depth 2"}
fig, axes = plt.subplots(1, 2, figsize=(14, 7), gridspec_kw={"width_ratios": [2, 1]})
cm = plt.get_cmap("tab10")
for i, t in enumerate(teams):
    x = d[d.team == t]
    axes[0].plot(x.time, x.after, ".-", ms=3, lw=0.7, color=cm(i % 10),
                 label=f"{t} [{SHORT.get(t, x.model.mode()[0])}] ({len(x)}g, {x.won.mean():.0%}, peak {int(x.after.max())})")
axes[0].set_title(f"Ladder rating after each game ({fmt}), by team")
axes[0].set_ylabel("rating"); axes[0].legend(fontsize=8); axes[0].grid(alpha=.3)
fig.autofmt_xdate()
net = d.assign(delta=d.after - d.before).groupby("team").delta.sum().reindex(teams)
axes[1].barh(net.index, net.values, color=["#59a14f" if v >= 0 else "#e15759" for v in net.values])
axes[1].set_title("Net rating change per team (sum of game deltas)")
axes[1].axvline(0, color="k", lw=.8)
foot = "\n".join(f"{t}: {c}" for t, c in CMDS.items() if t in teams)
plt.tight_layout(rect=(0, 0.14, 1, 1))
fig.text(0.01, 0.01, foot, family="monospace", fontsize=6.5, va="bottom", wrap=True)
plt.savefig(out, dpi=130)
print(d.groupby("team").agg(games=("won", "size"), win=("won", "mean"), peak=("after", "max"), last=("after", "last")))
print(net)
