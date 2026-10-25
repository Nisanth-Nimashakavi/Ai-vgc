#!/usr/bin/env bash
# Runner for Model 19 on Public Showdown with High-Speed Chance Search
set -euo pipefail

if [[ -f "$HOME/.config/ai-vgc/showdown.env" ]]; then
  source "$HOME/.config/ai-vgc/showdown.env"
fi

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT:$PROJECT_ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

TEAM=${1:-"data/teams/reg_mc_top_meta/MC371.txt"}
GAMES=${2:-10}
DEVICE=${DEVICE:-auto}
DEPTH=${DEPTH:-1}
SEARCH_K=${SEARCH_K:-20}
SEARCH_M=${SEARCH_M:-40}

PY_BIN="$PROJECT_ROOT/.venv/bin/python"
if [[ ! -x "$PY_BIN" ]]; then
  PY_BIN="$(command -v python3 || echo python)"
fi

EXTRA_ARGS=()
if [[ "$TEAM" == "pool" || -d "$TEAM" ]]; then
  TEAMS_DIR="$TEAM"
  if [[ "$TEAM" == "pool" ]]; then
    TEAMS_DIR="data/teams/pool_5600/holdout_clean"
    if [[ ! -d "$TEAMS_DIR" ]]; then
      TEAMS_DIR="data/teams/reg_mc"
    fi
  fi
  EXTRA_ARGS+=(--teams-dir "$TEAMS_DIR")
  TEAM_DESC="Random Pool ($TEAMS_DIR)"
else
  EXTRA_ARGS+=(--team "$TEAM")
  TEAM_DESC="$TEAM"
fi

echo "======================================================================"
echo " MODEL 19 SHOWDOWN LADDER RUNNER"
echo " Team:        $TEAM_DESC"
echo " Games:       $GAMES"
echo " Search:      Depth $DEPTH Chance-Branching (K=$SEARCH_K, M=$SEARCH_M)"
echo " Device:      $DEVICE"
echo " Move Policy: Deterministic Argmax (beta=15.0, gamma=0.91)"
echo " Username:    ${BOT:-nimnimbot}"
echo "======================================================================"

exec "$PY_BIN" scripts/ladder_model19.py \
  "${EXTRA_ARGS[@]}" \
  --games "$GAMES" \
  --search --depth "$DEPTH" \
  --search-k "$SEARCH_K" --search-m "$SEARCH_M" \
  --search-beta 15.0 --search-gamma 0.91 \
  --deterministic \
  --device "$DEVICE" "${@:3}"
