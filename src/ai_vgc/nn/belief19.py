"""Public MunchStats marginals -> finite CTS set hypotheses, never opponent sets.

The source supplies separate marginals, not jointly observed complete sets. These
particles are approximate hypotheses; missing species use a broad public-dex prior.
Champions spreads are Stat Points (at most 66 total and 32 per stat), not EVs.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import re

import vgc_engine

STATS = ("hp", "atk", "def", "spa", "spd", "spe")
SOURCE = "https://www.munchstats.com/champions/doubles/"
CONTRACT = "public-beliefs-19-v1"


def slug(value):
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _stratified(options, n, rng):
    options = [(value, float(weight)) for value, weight in options if float(weight) > 0]
    total = sum(weight for _, weight in options)
    if not total:
        raise ValueError("A public prior marginal is empty")
    result = []
    for i in range(n):
        threshold = (i + rng.random()) * total / n
        cumulative = 0
        for value, weight in options:
            cumulative += weight
            if cumulative >= threshold:
                result.append(value)
                break
    rng.shuffle(result)
    return result


def compile_public_priors(cache, dex, vocab, *, particles=16, seed=20261019):
    """Only public tables accepted; there is deliberately no team argument."""
    if particles < 2:
        raise ValueError("Beliefs require at least two hypotheses")
    known = {key: set(values) for key, values in vocab.items()}
    recipes = {}
    usage_species = 0
    for species_id, name in enumerate(vocab["species"]):
        info = dex.get(name, {})
        base = slug(info.get("baseSpecies", name))
        entry = cache.get(name, cache.get(base, {}))
        usage_species += bool(entry)
        # Per-species seeds avoid dependence on cache dictionary order or team order.
        digest = hashlib.sha256(f"{seed}:{name}".encode()).digest()
        rng = random.Random(int.from_bytes(digest[:8], "little"))
        def marginal(key, fallback):
            values = [(slug(v), w) for v, w in entry.get(key, []) if slug(v) in known[key]]
            return values or [(slug(v), 1) for v in fallback if slug(v) in known[key]]
        possible = list(info.get("abilities", {}).values())
        abilities = marginal("abilities", possible or ["No Ability"])
        if not abilities:
            abilities = [("noability", 1)]
        items = marginal("items", ["Choice Scarf", "Assault Vest", "Sitrus Berry", "Life Orb", "Focus Sash", "Leftovers"])
        spreads = []
        for text, weight in entry.get("spreads", []):
            try:
                points = tuple(map(int, text.split("/")))
            except (TypeError, ValueError):
                continue
            if len(points) == 6 and min(points) >= 0 and max(points) <= 32 and sum(points) <= 66:
                spreads.append((points, weight))
        if not spreads:
            # Broad legal hypotheses, independent of the actual opponent's values.
            spreads = [(v, 1) for v in [(32, 32, 2, 0, 0, 0), (2, 32, 0, 0, 0, 32),
                (32, 0, 2, 32, 0, 0), (2, 0, 0, 32, 0, 32), (32, 0, 16, 0, 18, 0),
                (16, 0, 10, 0, 8, 32), (32, 0, 0, 0, 2, 32), (0, 0, 0, 0, 0, 0)]]
        natures = [(slug(v), w) for v, w in entry.get("natures", []) if slug(v) in NATURES]
        if not natures:
            natures = [(v, 1) for v in ["hardy", "adamant", "modest", "jolly", "timid", "bold", "careful", "brave", "quiet", "relaxed", "sassy"]]
        moves = marginal("moves", ["Protect", "Tackle", "Helping Hand", "Substitute"])
        draws = [_stratified(options, particles, rng) for options in (items, abilities, spreads, natures)]
        sets = []
        for k in range(particles):
            remaining = list(moves)
            selected = []
            while remaining and len(selected) < 4:
                move = rng.choices([v for v, _ in remaining], [w for _, w in remaining])[0]
                selected.append(move)
                remaining = [(v, w) for v, w in remaining if v != move]
            sets.append({"species": name, "level": 50, "item": draws[0][k], "ability": draws[1][k],
                         "evs": dict(zip(STATS, draws[2][k])), "nature": draws[3][k], "moves": selected})
        recipes[species_id] = sets
    return recipes, {"species": len(recipes), "usage_species_including_forms": usage_species,
                     "particles_per_species": particles, "seed": seed,
                     "prior_semantics": "independently sampled public usage marginals; public-dex fallback"}


NATURES = {"hardy", "lonely", "brave", "adamant", "naughty", "bold", "docile", "relaxed", "impish", "lax",
           "timid", "hasty", "serious", "jolly", "naive", "modest", "mild", "quiet", "bashful", "rash",
           "calm", "gentle", "sassy", "careful", "quirky"}


class PublicBeliefBank:
    def __init__(self, path, *, particles=16, dex_dir=None):
        import os
        path = Path(path)
        raw = path.read_bytes()
        dex_root = Path(dex_dir or os.environ.get("VGC_DEX_DIR", ".localdex/data"))
        dex = json.loads((dex_root / "pokedex.json").read_text())
        recipes, meta = compile_public_priors(json.loads(raw), dex, vgc_engine.policy_vocab(), particles=particles)
        compiled = json.dumps(recipes, separators=(",", ":"), sort_keys=True)
        self.native = vgc_engine.BeliefPriors19(compiled)
        self.metadata = {**meta, "contract": CONTRACT, "source": SOURCE,
                         "cache_sha256": hashlib.sha256(raw).hexdigest(),
                         "compiled_sha256": hashlib.sha256(compiled.encode()).hexdigest(),
                         "public_dex_sha256": hashlib.sha256((dex_root / "pokedex.json").read_bytes()).hexdigest()}

    def tracker(self):
        return self.native.tracker()
