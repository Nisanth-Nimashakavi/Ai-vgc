"""Which abilities and items the sim implements, and their parameters.

Adding an entry here (plus the logic in sim.py) is only half the job: the id must also be covered by the
differential test, then listed in ledger.py. `python -m ai_vgc.gpusim.effects` writes data/gpusim/implemented.json,
which the Showdown case generator reads so its random teams exercise exactly what is implemented.
"""
import json
from pathlib import Path

AB_WEATHER = {"drought": 1, "drizzle": 2, "sandstream": 3, "snowwarning": 4}  # sun, rain, sand, snow
AB_TERRAIN = {"electricsurge": 1, "grassysurge": 2, "mistysurge": 3, "psychicsurge": 4}
AB_INTIMIDATE = {"intimidate"}

IT_WEATHER_EXT = {"heatrock": 1, "damprock": 2, "smoothrock": 3, "icyrock": 4}  # weather lasts 8 turns
IT_SPE_MOD = {"choicescarf": 6144}
IT_DMG_MOD = {"lifeorb": 5324}
IT_LEFTOVERS = {"leftovers"}
IT_SASH = {"focussash"}
IT_SITRUS = {"sitrusberry"}
IT_TERRAIN_EXT = {"terrainextender"}
IT_CLAY = {"lightclay"}  # screens last 8 turns

ABILITIES = set(AB_WEATHER) | set(AB_TERRAIN) | AB_INTIMIDATE
ITEMS = (set(IT_WEATHER_EXT) | set(IT_SPE_MOD) | set(IT_DMG_MOD) | IT_LEFTOVERS | IT_SASH | IT_SITRUS
         | IT_TERRAIN_EXT | IT_CLAY)


def dump(path=None):
    path = path or Path(__file__).parents[3] / "data/gpusim/implemented.json"
    Path(path).write_text(json.dumps({"abilities": sorted(ABILITIES), "items": sorted(ITEMS)}))


if __name__ == "__main__":
    dump()
