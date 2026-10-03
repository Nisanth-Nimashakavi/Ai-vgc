# GPU simulator (JAX) for Reg M-C

`src/ai_vgc/gpusim/`: a batched JAX re-implementation of the Pokemon Showdown VGC (`gen9championsvgc2026regmc`) battle
engine. Target: millions of turns/s on a GPU for RL training. Showdown (`pokemon-showdown-mc/`) is the oracle.

## How correctness is checked (read this first)

Everything is differential. `scripts/gpusim/game_cases.js` plays random 4v4 doubles games in real Showdown with its RNG
forced (damage roll, accuracy roll, per-turn coin flips by "n/d" key), and records the state after every turn.
`tests/gpusim/test_game.py` replays the same actions through the JAX sim and diffs HP, status, stat stages, PP, items,
active Pokemon, weather/terrain/Trick Room/Tailwind/screens, and each turn's action speeds.

```
node scripts/gpusim/game_cases.js 400 14 > data/gpusim/game_cases.json     # SEED=n for another seed
uv run --extra gpusim --extra dev pytest tests/gpusim -q                   # ~45 s (XLA opt level 0, see conftest)
bash scripts/gpusim/focus.sh id id ...                                     # one ability/item at a time
uv run python -m ai_vgc.gpusim.effects                                     # regenerate data/gpusim/implemented.json
uv run --extra gpusim python -m ai_vgc.gpusim.ledger                       # which abilities/items are done
uv run --extra gpusim python -m ai_vgc.gpusim.coverage                     # share of real team-pool sets playable
```

An ability/item is DONE only if it is in `effects.py`, appears in the random games (`test_every_implemented_effect_is_exercised`)
and the replay matches. Any "n/d" chance Showdown rolls that the sim does not know fails the test, so new mechanics cannot slip in.

## Layout

- `tables.py`: packs `data/gpusim/tables.json` (from `scripts/gpusim/export_tables.js`, the Champions mod) into arrays.
- `damage.py`: the damage formula. `effects.py`: rule registry (damage/accuracy modifier rules, absorb table, speed, ...).
- `sim.py`: state pytree, `start`, `step` (one turn), `replace` (KO refills), `winner`, `action_speeds`.
- `ledger.py` / `coverage.py`: progress reports.

## Status (see the ledger for the live numbers)

Done: damage incl. crits/ratio stages, stat stages, brn/par/psn/tox, status moves, secondaries, switching + replacement, PP,
weather/terrain/Trick Room/Tailwind/screens, Protect/Detect/Fake Out/flinch, ~118 abilities and ~56 items (damage/speed/accuracy
modifiers, absorb/immunity family, weather/terrain setters, Intimidate, Focus Sash, Sitrus, resist berries, ...).

NOT done (the real remaining work; the coverage report says 0% of real team sets are fully playable yet):
- Statuses/volatiles: sleep, freeze, confusion, attract, disable, taunt, encore, charge, perish, substitute, ...
- On-hit abilities/items (Rough Skin, Static, Defiant, Rocky Helmet, ...), AfterEachBoost (Defiant/Competitive), trace/copy,
  Mega Evolution (81 stones, species change mid-battle), berries that cure status, seeds, Eject Button/Red Card, Unburden etc.
- Moves with handlers (~160), multi-hit, drain/recoil, U-turn/switch moves, Helping Hand, Follow Me, Wide Guard, hazards, weather ball...
- Pivots/forced switches and the choice-lock rule (the sim trusts the actions it is given; a legal-action mask is still to write).
- RL interface (`reset`/`step`/mask/obs), throughput numbers on a GPU (this laptop has none; CPU compile of the vmapped step is slow
  without `XLA_FLAGS=--xla_backend_optimization_level=0`).

## Showdown quirks found (all are in the code with comments)

Dynamic speed re-sort each action; paralysis = 1/8 skip and speed applied after the chain with floor; TR speed is `-speed`
in this mod; TryHit order: Protect/Psychic Terrain, then abilities (absorb), then type immunity, then accuracy; Flash Fire
sets accuracy true; Toxic never misses for Poison users; battle ends mid-residual when a side is wiped; Victory Star counts
per holder; selfBoost is skipped on a battle-ending KO; Champions PP = (min(pp,20)/5+1)*4.
