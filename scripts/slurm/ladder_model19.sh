#!/bin/bash -l
#SBATCH --job-name=vgc19-ladder
#SBATCH --output=logs/%x-%j.out
#SBATCH --partition=unlimited
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4

set -euo pipefail
cd ~/ai-vgc
mkdir -p logs

export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/src:$PWD:$PWD/scripts"

if [[ -f "$HOME/.config/ai-vgc/showdown.env" ]]; then
  source "$HOME/.config/ai-vgc/showdown.env"
fi

TEAM=${1:-"data/teams/reg_mc_top_meta/MC371.txt"}
GAMES=${2:-10}

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
  echo "Starting Model 19 Ladder session with generalist pool ($TEAMS_DIR) for $GAMES games on GPU..."
else
  EXTRA_ARGS+=(--team "$TEAM")
  echo "Starting Model 19 Ladder session with $TEAM for $GAMES games on GPU..."
fi

.venv/bin/python scripts/ladder_model19.py \
  "${EXTRA_ARGS[@]}" \
  --games "$GAMES" \
  --search --depth 1 \
  --search-k 20 --search-m 40 \
  --search-beta 15.0 --search-gamma 0.91 \
  --device auto "${@:3}"
