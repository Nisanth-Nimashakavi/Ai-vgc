"""Pack data/gpusim/tables.json (exported from Showdown) into numpy arrays for the sim."""
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

TABLES = Path(__file__).parents[3] / "data/gpusim/tables.json"
CATEGORY = {"Physical": 0, "Special": 1, "Status": 2}
# target codes: how a move picks its targets
TARGET = {"normal": 0, "any": 0, "allAdjacentFoes": 1, "allAdjacent": 2, "self": 3, "all": 4, "allySide": 4, "foeSide": 4}
STATS = ("hp", "atk", "def", "spa", "spd", "spe")
BOOSTS = STATS[1:]  # stat stages tracked in the sim (accuracy/evasion come later)
STATUS = {None: 0, "brn": 1, "par": 2, "psn": 3, "tox": 4}
NATURES = {  # nature -> (plus, minus); neutral natures have none
    "Adamant": ("atk", "spa"), "Modest": ("spa", "atk"), "Bold": ("def", "atk"),
    "Calm": ("spd", "atk"), "Jolly": ("spe", "spa"), "Timid": ("spe", "atk"),
    "Impish": ("def", "spa"), "Careful": ("spd", "spa"), "Hardy": (None, None),
}


def to_id(name):
    """Showdown id: lowercase alphanumerics only."""
    return "".join(c for c in name.lower() if c.isalnum())


class Tables:
    def __init__(self, path=TABLES):
        t = json.loads(Path(path).read_text())
        self.raw = t
        self.types = t["types"]
        self.chart = np.array(t["typeChart"], np.float32)  # [attacker, defender]
        self.species_ix = {s["id"]: i for i, s in enumerate(t["species"])}
        self.move_ix = {m["id"]: i for i, m in enumerate(t["moves"])}
        sp, mv = t["species"], t["moves"]
        tix = {n: i for i, n in enumerate(self.types)}
        self.base = np.array([[s["baseStats"][k] for k in STATS] for s in sp], np.int32)
        self.sp_types = np.array([[tix[x] for x in s["types"]] + [-1] * (2 - len(s["types"])) for s in sp], np.int32)
        self.mv_type = np.array([tix[m["type"]] for m in mv], np.int32)
        self.mv_cat = np.array([CATEGORY[m["category"]] for m in mv], np.int32)
        self.mv_bp = np.array([m["basePower"] for m in mv], np.int32)
        self.mv_acc = np.array([100 if m["accuracy"] is True else m["accuracy"] for m in mv], np.int32)
        self.mv_prio = np.array([m["priority"] for m in mv], np.int32)
        self.mv_target = np.array([TARGET.get(m["target"], 0) for m in mv], np.int32)

        def boosts(d):
            d = d or {}
            return [d.get(k, 0) for k in BOOSTS]

        def sec(m):  # the first (only supported) secondary effect, else a no-op
            return (m.get("secondaries") or [{}])[0]

        self.mv_boost = np.array([boosts(m.get("boosts")) for m in mv], np.int32)
        self.mv_boost_self = np.array([m["target"] == "self" for m in mv], bool)
        self.mv_self_boost = np.array([boosts((m.get("self") or m.get("selfBoost") or {}).get("boosts")) for m in mv], np.int32)
        self.mv_status = np.array([STATUS.get(m.get("status"), 0) for m in mv], np.int32)
        self.mv_sec_chance = np.array([sec(m).get("chance", 0) for m in mv], np.int32)
        self.mv_sec_status = np.array([STATUS.get(sec(m).get("status"), 0) for m in mv], np.int32)
        self.mv_sec_boost = np.array([boosts(sec(m).get("boosts")) for m in mv], np.int32)
        self.mv_sec_self_boost = np.array([boosts((sec(m).get("self") or {}).get("boosts")) for m in mv], np.int32)
        self.mv_ign_def = np.array([bool(m.get('ignoreDefensive')) for m in mv], bool)
        self.mv_ign_off = np.array([bool(m.get('ignoreOffensive')) for m in mv], bool)
        self.mv_powder = np.array([bool((m.get('flags') or {}).get('powder')) for m in mv], bool)
        # field effect codes: 1-4 weather (sun, rain, sand, snow), 5 Trick Room, 6 Tailwind, 7 Reflect,
        # 8 Light Screen, 9 Aurora Veil, 10-13 terrain (electric, grassy, misty, psychic)
        fld = {"sunnyday": 1, "raindance": 2, "sandstorm": 3, "snowscape": 4, "trickroom": 5, "tailwind": 6,
               "reflect": 7, "lightscreen": 8, "auroraveil": 9, "electricterrain": 10, "grassyterrain": 11,
               "mistyterrain": 12, "psychicterrain": 13}
        self.mv_field = np.array([fld.get(m["id"], 0) for m in mv], np.int32)
        self.mv_halve_grassy = np.array([m["id"] in ("earthquake", "bulldoze", "magnitude") for m in mv], bool)
        # damaging moves always check type immunity; status moves only when ignoreImmunity is false
        self.mv_imm = np.array([m["category"] != "Status" or m.get("ignoreImmunity") is False for m in mv], bool)


@lru_cache
def load():
    return Tables()


def calc_stats(base, evs, nature, level=50):
    """Champions stat formula (data/mods/champions/scripts.ts statModify, level-clause branch)."""
    plus, minus = NATURES.get(nature, (None, None))
    out = []
    for k, b, e in zip(STATS, base, evs):
        core = (2 * b + 31 + max(2 * e - 1, 0)) * level // 100
        if k == "hp":
            out.append(core + level + 10)
            continue
        s = core + 5
        if k == plus:
            s = s * 110 // 100
        elif k == minus:
            s = s * 90 // 100
        out.append(s)
    return out
