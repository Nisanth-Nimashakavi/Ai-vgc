"""One-turn JAX sim vs Showdown. Cases: node scripts/gpusim/turn_cases.js 1000 > data/gpusim/turn_cases.json"""
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ai_vgc.gpusim import sim

CASES = Path(__file__).parents[2] / "data/gpusim/turn_cases.json"


def run(tab, case):
    state = sim.make_state(case["teams"])
    actions = jnp.array([[a["slot"], a["foe"]] for a in case["actions"]], jnp.int32)
    rolls = jnp.full((4, 4), case["roll"], jnp.int32)
    return state, np.asarray(sim.step(tab, state, actions, rolls, jnp.zeros((4, 4), bool),
                                      jnp.ones(4, bool), jnp.zeros(4))["hp"])


@pytest.mark.skipif(not CASES.exists(), reason="generate turn_cases.json first")
def test_turn_matches_showdown():
    tab, cases = sim.make_tables(), json.loads(CASES.read_text())
    step = jax.jit(sim.step)
    bad_stats = bad_hp = 0
    for c in cases:
        state = sim.make_state(c["teams"])
        if (state["stats"] != np.array(c["stats"])).any():
            bad_stats += 1
            continue
        actions = jnp.array([[a["slot"], a["foe"]] for a in c["actions"]], jnp.int32)
        got = np.asarray(step(tab, state, actions, jnp.full((4, 4), c["roll"], jnp.int32),
                              jnp.zeros((4, 4), bool), jnp.ones(4, bool), jnp.zeros(4))["hp"])
        if (got != np.array(c["hp"])).any():
            bad_hp += 1
            if bad_hp <= 5:
                print("HP MISMATCH", [c["teams"][s][j]["species"] + ":" + c["teams"][s][j]["moves"][c["actions"][s*2+j]["slot"]] for s in (0, 1) for j in (0, 1)], c["actions"], "roll", c["roll"], "stats", c["stats"], "want", c["hp"], "got", got.tolist())
    assert bad_stats == 0 and bad_hp == 0, f"stats {bad_stats}, hp {bad_hp} of {len(cases)}"
