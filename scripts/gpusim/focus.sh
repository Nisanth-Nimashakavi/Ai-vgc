#!/usr/bin/env bash
# Test each given ability/item on its own against Showdown: scripts/gpusim/focus.sh id id ...
# prints one line per id: ok, or the number of diverging games.
cd "$(dirname "$0")/../.."
for id in "$@"; do
  FOCUS=$id SEED=${SEED:-5} node scripts/gpusim/game_cases.js ${GAMES:-120} ${TURNS:-12} > data/gpusim/game_cases.json
  out=$(uv run --extra gpusim --extra dev pytest tests/gpusim/test_game.py -k test_games -x -q 2>&1 | grep -E "games diverge|passed|Error" | head -1)
  echo "$id: $out"
done
