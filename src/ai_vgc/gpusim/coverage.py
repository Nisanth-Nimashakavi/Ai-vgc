"""What share of the real team pool the sim can play, and which abilities/items/moves block the most sets.

usage: uv run --extra gpusim python -m ai_vgc.gpusim.coverage [teams_dir]
"""
import re
import sys
from collections import Counter
from pathlib import Path

from . import tables as T

TEAMS = Path(__file__).parents[3] / "data/teams/reg_mc"
# implemented effects: ability/item ids the sim models (extend as they land); items/abilities with no battle
# effect in the sim's scope are listed as supported so they don't count as blockers
ABILITIES: set = set()
ITEMS: set = set()
FIELD_MOVES = {"sunnyday", "raindance", "sandstorm", "snowscape", "trickroom", "tailwind", "reflect",
               "lightscreen", "auroraveil", "electricterrain", "grassyterrain", "mistyterrain", "psychicterrain"}


def parse_team(text):
    """Showdown paste -> list of dict(species, item, ability, moves)."""
    mons = []
    for block in re.split(r"\n\s*\n", text.strip()):
        lines = [l.strip() for l in block.splitlines() if l.strip()]
        if not lines:
            continue
        head = lines[0].split("@")
        who = re.sub(r"\s*\((M|F)\)\s*$", "", head[0].strip())  # drop the gender tag
        inner = re.search(r"\(([^()]+)\)\s*$", who)  # "Nickname (Species)" -> Species
        name = inner.group(1) if inner else who
        m = dict(species=T.to_id(name), item=T.to_id(head[1]) if len(head) > 1 else "", ability="", moves=[])
        for l in lines[1:]:
            if l.startswith("Ability:"):
                m["ability"] = T.to_id(l.split(":", 1)[1])
            elif l.startswith("-"):
                m["moves"].append(T.to_id(l[1:]))
        mons.append(m)
    return mons


def move_supported(t, mid):
    mv = t.raw["moves"][t.move_ix[mid]] if mid in t.move_ix else None
    if mv is None:
        return False
    if mid in FIELD_MOVES:
        return True
    ok_sec = not mv.get("secondaries") or len(mv["secondaries"]) == 1
    plain = not mv["handlers"] and ok_sec and not any(mv.get(k) for k in (
        "multihit", "drain", "recoil", "heal", "damage", "ohko", "selfdestruct", "forceSwitch", "selfSwitch",
        "volatileStatus", "sideCondition", "weather", "terrain", "pseudoWeather", "overrideOffensiveStat",
        "overrideDefensiveStat", "breaksProtect", "isProtect"))
    return plain and mv["target"] in ("normal", "any", "allAdjacentFoes", "allAdjacent", "self") and (mv["critRatio"] or 1) == 1


def report(teams_dir=TEAMS, top=25):
    t = T.load()
    mons = [m for f in sorted(Path(teams_dir).glob("*.txt")) for m in parse_team(f.read_text())]
    blockers = {"ability": Counter(), "item": Counter(), "move": Counter(), "species": Counter()}
    full = 0
    for m in mons:
        miss = []
        if m["species"] not in t.species_ix:
            miss.append(("species", m["species"]))
        if m["ability"] not in ABILITIES:
            miss.append(("ability", m["ability"]))
        if m["item"] and m["item"] not in ITEMS:
            miss.append(("item", m["item"]))
        miss += [("move", x) for x in m["moves"] if not move_supported(t, x)]
        for kind, name in miss:
            blockers[kind][name] += 1
        full += not miss
    print(f"{len(mons)} sets from {teams_dir}; fully supported: {full} ({100 * full / len(mons):.1f}%)")
    for kind, c in blockers.items():
        print(f"\nTop {kind} blockers ({sum(c.values())} total blocks, {len(c)} distinct):")
        print("  " + ", ".join(f"{n} {k}" for k, n in c.most_common(top)))
    return blockers


if __name__ == "__main__":
    report(*sys.argv[1:2])
