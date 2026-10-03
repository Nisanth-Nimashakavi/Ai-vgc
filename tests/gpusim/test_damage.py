"""JAX damage vs Showdown. Cases: node scripts/gpusim/damage_cases.js 3000 > data/gpusim/damage_cases.json"""
import json
from pathlib import Path

import jax
import numpy as np
import pytest

from ai_vgc.gpusim.damage import damage

CASES = Path(__file__).parents[2] / "data/gpusim/damage_cases.json"


@pytest.mark.skipif(not CASES.exists(), reason="generate damage_cases.json first")
def test_damage_matches_showdown():
    cases = json.loads(CASES.read_text())
    col = lambda k, dt=np.int32: np.array([c[k] for c in cases], dtype=dt)
    got = np.asarray(jax.jit(damage)(
        col("level"), col("power"), col("atk"), col("dfn"), col("roll"), col("crit", bool),
        col("stab"), col("type_exp"), col("burned", bool), col("spread", bool),
        np.full(len(cases), 4096, np.int32), np.full(len(cases), 4096, np.int32)))
    want = col("expected")
    bad = np.nonzero(got != want)[0]
    for i in bad[:10]:
        print(cases[i], "got", got[i])
    assert len(bad) == 0, f"{len(bad)}/{len(cases)} mismatches"
