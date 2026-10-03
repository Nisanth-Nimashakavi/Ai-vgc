"""Which abilities and items the sim implements, and their parameters.

Adding an entry here (plus the logic in sim.py) is only half the job: the id must also be covered by the
differential test, then listed in ledger.py. `python -m ai_vgc.gpusim.effects` writes data/gpusim/implemented.json,
which the Showdown case generator reads so its random teams exercise exactly what is implemented.
"""
import json
from pathlib import Path

AB_WEATHER = {"drought": 1, "drizzle": 2, "sandstream": 3, "snowwarning": 4, "orichalcumpulse": 1}  # sun, rain, sand, snow
AB_TERRAIN = {"electricsurge": 1, "grassysurge": 2, "mistysurge": 3, "psychicsurge": 4, "hadronengine": 1}
AB_INTIMIDATE = {"intimidate"}

IT_WEATHER_EXT = {"heatrock": 1, "damprock": 2, "smoothrock": 3, "icyrock": 4}  # weather lasts 8 turns
IT_SPE_MOD = {"choicescarf": 6144}
IT_LIFEORB = {"lifeorb"}
IT_LEFTOVERS = {"leftovers"}
IT_SASH = {"focussash"}
IT_SITRUS = {"sitrusberry"}
IT_TERRAIN_EXT = {"terrainextender"}
IT_CLAY = {"lightclay"}  # screens last 8 turns

# ---------------------------------------------------------------------------------------------------------------
# Damage modifier rules. RULES[kind][event][id] = [(priority, (cond, cond), mult4096), ...] (at most 2 per id/event).
# kind: "ab" attacker's ability, "it" attacker's item, "dab"/"dit" defender's ability/item, "aab" an ability held
# by the attacker or its ally, "any" an ability held by anyone on the field (auras).
# events: bp (BasePower), atk/spa (the attacker's stat), def/spd (the defender's stat), dmg (ModifyDamage chain).
# Showdown applies each event's handlers as one chainModify product, highest priority first.
# ---------------------------------------------------------------------------------------------------------------
RULES: dict = {}
TYPES = ["Bug", "Dark", "Dragon", "Electric", "Fairy", "Fighting", "Fire", "Flying", "Ghost", "Grass", "Ground", "Ice",
         "Normal", "Poison", "Psychic", "Rock", "Steel", "Water"]


def rule(kind, event, ids, prio, conds, mult):
    conds = (conds,) if isinstance(conds, str) else tuple(conds)
    conds = (conds + ("always", "always"))[:2]
    for i in ([ids] if isinstance(ids, str) else ids):
        RULES.setdefault(kind, {}).setdefault(event, {}).setdefault(i, []).append((prio, conds, mult))


for _id, _c, _m, _p in [("technician", "bp<=60", 6144, 30), ("ironfist", "punch", 4915, 23), ("strongjaw", "bite", 6144, 19),
                        ("toughclaws", "contact", 5325, 21), ("sharpness", "slicing", 6144, 19),
                        ("megalauncher", "pulse", 6144, 19), ("sheerforce", "sheerforce", 5325, 21),
                        ("analytic", "analytic", 5325, 21), ("sandforce", "sand_rgs", 5325, 21),
                        ("flareboost", "spec_burn", 6144, 19), ("toxicboost", "phys_psn", 6144, 19),
                        ("punkrock", "sound", 5325, 7), ("reckless", "recoil", 4915, 23)]:
    rule("ab", "bp", _id, _p, _c, _m)
rule("ab", "bp", "rivalry", 24, "rivalry_same", 5120)
rule("ab", "bp", "rivalry", 24, "rivalry_diff", 3072)
rule("ab", "bp", ["aerilate", "pixilate", "refrigerate", "galvanize", "normalize", "dragonize"], 23, "retyped", 4915)
rule("dab", "bp", "dryskin", 17, "mt:Fire", 5120)
rule("aab", "bp", "battery", 22, ["spec", "not_self"], 5325)
rule("aab", "bp", "powerspot", 22, "not_self", 5325)
rule("aab", "bp", "steelyspirit", 22, "mt:Steel", 6144)

# retyping abilities: id -> (kind, new type); kind 1 = only Normal moves, 2 = every move, 3 = sound moves
RETYPE = {"aerilate": (1, "Flying"), "pixilate": (1, "Fairy"), "refrigerate": (1, "Ice"), "galvanize": (1, "Electric"),
          "dragonize": (1, "Dragon"), "normalize": (2, "Normal"), "liquidvoice": (3, "Water")}

for _id, _t in [("blaze", "Fire"), ("overgrow", "Grass"), ("torrent", "Water"), ("swarm", "Bug")]:
    for _e in ("atk", "spa"):
        rule("ab", _e, _id, 5, ["mt:" + _t, "hp_third"], 6144)
for _e in ("atk", "spa"):
    rule("ab", _e, "defeatist", 5, "hp_half", 2048)
    rule("ab", _e, "dragonsmaw", 5, "mt:Dragon", 6144)
    rule("ab", _e, "steelworker", 5, "mt:Steel", 6144)
    rule("ab", _e, "transistor", 5, "mt:Electric", 5325)
    rule("ab", _e, "rockypayload", 5, "mt:Rock", 6144)
    rule("ab", _e, "firemane", 5, "mt:Fire", 6144)
    rule("ab", _e, "waterbubble", 5, "mt:Water", 8192)
    rule("ab", _e, "stakeout", 5, "fresh_target", 8192)
    rule("dab", _e, "waterbubble", 5, "mt:Fire", 2048)
    rule("dab", _e, "heatproof", 6, "mt:Fire", 2048)
    rule("dab", _e, "thickfat", 6, "mt:Ice", 2048)
    rule("dab", _e, "thickfat", 6, "mt:Fire", 2048)
    rule("dab", _e, "purifyingsalt", 6, "mt:Ghost", 2048)
rule("ab", "atk", ["hugepower", "purepower"], 5, "always", 8192)
rule("ab", "atk", "guts", 5, "status", 6144)
rule("ab", "atk", "hustle", 5, "always", 6144)
rule("ab", "atk", "gorillatactics", 1, "always", 6144)
rule("ab", "atk", "orichalcumpulse", 5, "sun", 5461)
rule("ab", "spa", "hadronengine", 5, "electric_terrain", 5461)
rule("ab", "spa", "solarpower", 5, "sun", 6144)
rule("ab", "spa", ["plus", "minus"], 5, "ally_plus_minus", 6144)
rule("it", "atk", "lightball", 1, "pikachu", 8192)
rule("it", "spa", "lightball", 1, "pikachu", 8192)
rule("ab", "def", "furcoat", 6, "always", 8192)
rule("ab", "def", "marvelscale", 6, "dstatus", 6144)
rule("ab", "def", "grasspelt", 6, "grassy_terrain", 6144)

# accuracy chain: the attacker's SourceModifyAccuracy, the defender's ModifyAccuracy, anyone's AnyModifyAccuracy
rule("ab", "acc", "hustle", -1, "phys", 3277)
rule("ab", "acc", "compoundeyes", -1, "always", 5325)
rule("it", "acc", "widelens", -2, "always", 4505)
rule("it", "acc", "zoomlens", -2, "target_moved", 4915)
rule("dab", "acc", "sandveil", -1, "sandstorm", 3277)
rule("dab", "acc", "snowcloak", -1, "snow", 3277)
rule("dab", "acc", "tangledfeet", -1, "confused", 2048)
rule("dit", "acc", "brightpowder", -2, "always", 3686)

# status immunity: ability -> statuses it blocks (1 brn, 2 par, 3 psn, 4 tox)
STATUS_BLOCK = {"immunity": (3, 4), "limber": (2,), "waterveil": (1,), "waterbubble": (1,), "thermalexchange": (1,),
                "pastelveil": (3, 4), "purifyingsalt": (1, 2, 3, 4), "comatose": (1, 2, 3, 4)}
SUN_BLOCK = {"leafguard"}  # blocks every status in sun
ALLY_STATUS_BLOCK = {"pastelveil": (3, 4)}

# weather residuals: fractions are 1/n of max HP
SAND_IMMUNE = {"sandforce", "sandrush", "sandveil", "overcoat", "magicguard"}
SUN_DAMAGE = {"solarpower": 8, "dryskin": 8}
RAIN_HEAL = {"dryskin": 8, "raindish": 16}
SNOW_HEAL = {"icebody": 16}
MAGIC_GUARD = {"magicguard"}
HEATPROOF = {"heatproof"}
POWDER_IMMUNE = {"overcoat"}

# TryHit absorption: ability -> (matches, effect). matches: a type name, or a flag among
# wind/bullet/sound/status/ally/notse. effect: ("heal",) 1/4 max HP | ("boost", stat, n) | ("flashfire",) | ("immune",)
ABSORB = {
    "waterabsorb": ("Water", ("heal",)), "dryskin": ("Water", ("heal",)), "voltabsorb": ("Electric", ("heal",)),
    "eartheater": ("Ground", ("heal",)), "lightningrod": ("Electric", ("boost", "spa", 1)),
    "stormdrain": ("Water", ("boost", "spa", 1)), "motordrive": ("Electric", ("boost", "spe", 1)),
    "sapsipper": ("Grass", ("boost", "atk", 1)), "wellbakedbody": ("Fire", ("boost", "def", 2)),
    "windrider": ("wind", ("boost", "atk", 1)), "flashfire": ("Fire", ("flashfire",)),
    "levitate": ("Ground", ("immune",)), "goodasgold": ("status", ("immune",)), "bulletproof": ("bullet", ("immune",)),
    "soundproof": ("sound", ("immune",)), "telepathy": ("ally", ("immune",)), "wonderguard": ("notse", ("immune",)),
}
REDIRECT = {"lightningrod": "Electric", "stormdrain": "Water"}
UNGROUNDED = {"levitate"}
IGNORE_REDIRECT = {"stalwart", "propellertail"}  # the move can not be redirected
MOLD_BREAKER = {"moldbreaker", "teravolt", "turboblaze"}  # the move ignores the target's ability
rule("ab", "atk", "flashfire", 5, "ff_fire", 6144)
rule("ab", "spa", "flashfire", 5, "ff_fire", 6144)

# speed abilities: id -> (condition, multiplier). Unburden's condition is "the Pokemon has lost its item".
SPEED = {"sandrush": ("sand", 8192), "swiftswim": ("rain", 8192), "chlorophyll": ("sun", 8192),
         "slushrush": ("snow", 8192), "surgesurfer": ("eterrain", 8192), "quickfeet": ("status", 6144),
         "unburden": ("noitem", 8192)}
SPEED_WHEN = ["", "sun", "rain", "sand", "snow", "eterrain", "status", "noitem"]
SPEED_BOOST = {"speedboost"}  # +1 Spe at the end of every turn once it has been out a turn

# crit ratio stage added by the attacker's ability/item (Merciless: always crits a poisoned target)
CRIT_PLUS = {"superluck": 1, "scopelens": 1}
CRIT_LEEK = {"leek": 2}  # only Farfetch'd / Sirfetch'd
MERCILESS = {"merciless"}

# the final damage chain: attacker ModifyDamage, defender SourceModifyDamage, anyone's AnyModifyDamage
rule("ab", "dmg", "tintedlens", 0, "resisted", 8192)
rule("ab", "dmg", "neuroforce", 0, "se", 5120)
rule("ab", "dmg", "sniper", 0, "crit", 6144)
rule("dab", "dmg", ["multiscale", "shadowshield"], 0, "full_hp", 2048)
rule("dab", "dmg", ["solidrock", "filter", "prismarmor"], 0, "se", 3072)
rule("dab", "dmg", "fluffy", 0, "contact", 2048)
rule("dab", "dmg", "fluffy", 0, "mt:Fire", 8192)
rule("dab", "dmg", "icescales", 0, "spec", 2048)
rule("dab", "dmg", "punkrock", 0, "sound", 2048)

TYPE_BOOST_ITEMS = {"charcoal": "Fire", "mysticwater": "Water", "magnet": "Electric", "miracleseed": "Grass",
                    "nevermeltice": "Ice", "blackbelt": "Fighting", "poisonbarb": "Poison", "softsand": "Ground",
                    "sharpbeak": "Flying", "twistedspoon": "Psychic", "silverpowder": "Bug", "hardstone": "Rock",
                    "spelltag": "Ghost", "dragonfang": "Dragon", "blackglasses": "Dark", "metalcoat": "Steel",
                    "silkscarf": "Normal", "fairyfeather": "Fairy"}
for _i, _t in TYPE_BOOST_ITEMS.items():
    rule("it", "bp", _i, 15, "mt:" + _t, 4915)
rule("it", "bp", "muscleband", 16, "phys", 4505)
rule("it", "bp", "wiseglasses", 16, "spec", 4505)
rule("it", "dmg", "expertbelt", 0, "se", 4915)
rule("it", "dmg", "lifeorb", 0, "always", 5324)

# resist berries: halve a super-effective hit of their type (Chilan: any Normal hit) and are consumed
RESIST_BERRIES = {"occaberry": "Fire", "passhoberry": "Water", "wacanberry": "Electric", "rindoberry": "Grass",
                  "yacheberry": "Ice", "chopleberry": "Fighting", "kebiaberry": "Poison", "shucaberry": "Ground",
                  "cobaberry": "Flying", "payapaberry": "Psychic", "tangaberry": "Bug", "chartiberry": "Rock",
                  "kasibberry": "Ghost", "habanberry": "Dragon", "colburberry": "Dark", "babiriberry": "Steel",
                  "roseliberry": "Fairy", "chilanberry": "Normal"}
for _i, _t in RESIST_BERRIES.items():
    rule("dit", "dmg", _i, 0, ["mt:" + _t, "normal_hit" if _i == "chilanberry" else "se"], 2048)

# only part of these is modelled so far (burn immunity without the on-hit boost, ...): not counted as done
PARTIAL = {"thermalexchange", "pastelveil"}

ABILITIES = (set(AB_WEATHER) | set(AB_TERRAIN) | AB_INTIMIDATE | set(RETYPE) | {"adaptability"} | set(STATUS_BLOCK) | SUN_BLOCK | SAND_IMMUNE | set(SUN_DAMAGE)
             | set(RAIN_HEAL) | set(SNOW_HEAL) | HEATPROOF | POWDER_IMMUNE | {"noguard", "victorystar", "wonderskin"} | set(ABSORB) | MOLD_BREAKER | set(SPEED) | SPEED_BOOST | IGNORE_REDIRECT | {"superluck", "merciless"}
             | {i for k in ("ab", "dab", "aab", "any") for ev in RULES.get(k, {}).values() for i in ev}) - PARTIAL
ITEMS = (set(IT_WEATHER_EXT) | set(IT_SPE_MOD) | IT_LIFEORB | IT_LEFTOVERS | IT_SASH | IT_SITRUS
         | IT_TERRAIN_EXT | IT_CLAY | {"scopelens", "leek"} | {i for k in ("it", "dit") for ev in RULES.get(k, {}).values() for i in ev})


# Per-turn coin flips the differential test forces in Showdown ("n/d" -> bool) and replays here. A chance Showdown
# rolls that is not listed fails the test, so a new mechanic cannot slip in unnoticed. 1/8 is both Champions'
# paralysis and a stage-2 crit; 1/24 is the base crit chance.
CHANCE_KEYS = ["1/8", "1/24", "1/2", "1/3", "1/4", "1/5", "3/10", "1/10"] + [f"1/{3 ** n}" for n in range(2, 7)]


def dump(path=None):
    path = path or Path(__file__).parents[3] / "data/gpusim/implemented.json"
    Path(path).write_text(json.dumps({"abilities": sorted(ABILITIES), "items": sorted(ITEMS)}))


if __name__ == "__main__":
    dump()


COND_NAMES = (["always", "bp<=60", "punch", "bite", "contact", "slicing", "pulse", "sound", "recoil", "sheerforce",
               "analytic", "rivalry_same", "rivalry_diff", "sand_rgs", "spec_burn", "phys_psn", "status", "dstatus",
               "hp_third", "hp_half", "phys", "spec", "full_hp", "se", "resisted", "crit", "fresh_target", "sun",
               "electric_terrain", "grassy_terrain", "pikachu", "retyped", "not_self", "ally_plus_minus", "normal_hit",
               "target_moved", "sandstorm", "snow", "confused", "ff_fire"]
              + ["mt:" + t for t in TYPES])
