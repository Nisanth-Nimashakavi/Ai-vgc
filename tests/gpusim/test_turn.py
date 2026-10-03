"""One-turn JAX sim vs Showdown. Cases: node scripts/gpusim/turn_cases.js 2000 > data/gpusim/turn_cases.json"""
import json
import re
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ai_vgc.gpusim import sim
from ai_vgc.gpusim.tables import STATUS

CASES = Path(__file__).parents[2] / "data/gpusim/turn_cases.json"


def forced_rng(c):
    """The RNG draws turn_cases.js forces in Showdown: fixed roll, no crits, secondary roll = the same value."""
    return dict(rolls=jnp.full((4, 4), c["roll"], jnp.int32), crits=jnp.zeros((4, 4), bool),
                hits=jnp.ones(4, bool), sec=jnp.full((4, 4), c["roll"], jnp.int32),
                skip=jnp.full(4, c["parSkip"], bool), tiebreak=tiebreak(c))


def tiebreak(c):
    """Speed ties are broken by Showdown's PRNG (not forcible), so read the order it chose from its log."""
    who = re.findall(r"^\|(?:move|cant)\|p([12])([ab]):", c["log"], re.M)
    tb = np.zeros(4)
    for pos, (side, slot) in enumerate(who[:4]):
        tb[(int(side) - 1) * 2 + "ab".index(slot)] = (3 - pos) / 4 + 0.01
    return jnp.asarray(tb)


@pytest.mark.skipif(not CASES.exists(), reason="generate turn_cases.json first")
def test_turn_matches_showdown():
    tab, cases = sim.make_tables(), json.loads(CASES.read_text())
    step = jax.jit(sim.step)
    bad = 0
    for c in cases:
        state = sim.make_state(c["teams"], status=[STATUS[s] for s in c["status"]])
        assert (state["stats"] == np.array(c["stats"])).all(), "stat formula mismatch"
        actions = jnp.array([[a["slot"], a["foe"]] for a in c["actions"]], jnp.int32)
        out = step(tab, state, actions, forced_rng(c))
        got = dict(hp=np.asarray(out["hp"]).tolist(), status=np.asarray(out["status"]).tolist(),
                   boosts=np.asarray(out["boosts"]).tolist())
        want = dict(hp=c["hp"], status=[STATUS.get(s, 0) for s in c["after"]["status"]], boosts=c["after"]["boosts"])
        for k in range(4):  # fainted Pokemon: status and boosts no longer matter
            if want["hp"][k] == 0:
                got["status"][k] = want["status"][k] = 0
                got["boosts"][k] = want["boosts"][k] = [0] * 5
        if got != want:
            bad += 1
            if bad <= 5:
                names = [c["teams"][s][j]["species"] + ":" + c["teams"][s][j]["moves"][c["actions"][s * 2 + j]["slot"]]
                         for s in (0, 1) for j in (0, 1)]
                print("MISMATCH", names, "start status", c["status"], "parSkip", c["parSkip"], "roll", c["roll"])
                print("  want", want, "\n  got ", got, "\n", c["log"])
    assert bad == 0, f"{bad} of {len(cases)} turns differ"
