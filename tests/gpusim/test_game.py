"""Multi-turn JAX sim vs Showdown. Cases: node scripts/gpusim/game_cases.js 300 > data/gpusim/game_cases.json"""
import json
import re
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ai_vgc.gpusim import sim
from ai_vgc.gpusim.tables import STATUS

CASES = Path(__file__).parents[2] / "data/gpusim/game_cases.json"
SHOWN = ("weather", "terrain")


def tiebreak(log):
    """Speed ties are broken by Showdown's PRNG (not forcible), so read the order it chose from its log."""
    body = log.split("\n|t:|", 1)[-1].split("\n|\n", 1)[0]  # the action lines before the end-of-turn residuals
    who = re.findall(r"^\|(?:move|cant|switch)\|p([12])([ab]):", body, re.M)
    who = [w for i, w in enumerate(who) if i == 0 or w != who[i - 1]]  # switches are logged twice (split lines)
    tb = np.zeros(4)
    for pos, (side, slot) in enumerate(who[:4]):
        tb[(int(side) - 1) * 2 + "ab".index(slot)] = (3 - pos) / 4 + 0.01
    return jnp.asarray(tb)


def rng_for(turn):
    return dict(rolls=jnp.full((4, 4), turn["roll"], jnp.int32), crits=jnp.zeros((4, 4), bool),
                acc=jnp.zeros((4, 4), jnp.int32), stall=jnp.zeros(4, bool), sec=jnp.full((4, 4), turn["roll"], jnp.int32),
                skip=jnp.full(4, turn["parSkip"], bool), tiebreak=tiebreak(turn["log"]))


def summarize(st):
    """The comparable part of a state (benched/fainted Pokemon have no meaningful stages or status)."""
    hp = np.asarray(st["hp"])
    active = np.asarray(st["active"])
    boosts = np.zeros((8, 5), int)
    for mon in active:
        if hp[mon] > 0:
            boosts[mon] = np.asarray(st["boosts"])[mon]
    status = np.where(hp > 0, np.asarray(st["status"]), 0)
    pres = lambda k: (np.asarray(st[k]) > 0).astype(int).tolist()
    return dict(hp=hp.tolist(), status=status.tolist(), boosts=boosts.tolist(), active=active.tolist(),
                weather=int(st["weather"]), terrain=int(st["terrain"]), tr=int(st["tr_t"] > 0),
                tw=pres("tw_t"), reflect=pres("reflect_t"), ls=pres("ls_t"), av=pres("av_t"))


def expected(after):
    hp = np.array(after["hp"])
    status = np.array([STATUS.get(s or None, 0) for s in after["status"]])
    boosts = np.zeros((8, 5), int)
    for mon in after["active"]:
        if hp[mon] > 0:
            boosts[mon] = after["boosts"][mon]
    return dict(hp=hp.tolist(), status=np.where(hp > 0, status, 0).tolist(), boosts=boosts.tolist(),
                active=after["active"], weather=after["weather"], terrain=after["terrain"], tr=after["tr"],
                tw=after["tw"], reflect=after["reflect"], ls=after["ls"], av=after["av"])


@pytest.mark.skipif(not CASES.exists(), reason="generate game_cases.json first")
def test_games_match_showdown():
    tab, games = sim.make_tables(), json.loads(CASES.read_text())
    step, replace = jax.jit(sim.step), jax.jit(sim.replace)
    bad = n = 0
    for g in games:
        st = sim.make_state(g["teams"])
        assert (st["stats"] == np.array(g["stats"])).all(), "stat formula mismatch"
        for t, turn in enumerate(g["turns"]):
            n += 1
            st = step(tab, st, jnp.array(turn["actions"], jnp.int32), rng_for(turn))
            st = replace(st, jnp.array(turn["replace"], jnp.int32))
            got, want = summarize(st), expected(turn["after"])
            if got != want:
                bad += 1
                if bad <= 3:
                    print("MISMATCH turn", t, "actions", turn["actions"], "roll", turn["roll"])
                    for k in want:
                        if got[k] != want[k]:
                            print("  ", k, "want", want[k], "\n      got ", got[k])
                    print(turn["log"][:2500])
                break  # later turns depend on this one
    assert bad == 0, f"{bad} of {len(games)} games diverge ({n} turns checked)"
