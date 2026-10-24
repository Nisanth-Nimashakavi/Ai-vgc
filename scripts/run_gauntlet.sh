#!/bin/bash
# Run Method B (The Gauntlet Benchmark):
# Tests candidate models against the pool of 500+ tournament teams piloted by v6 on Showdown.
#
# Usage:
#   bash scripts/run_gauntlet.sh             # Run both v7f specialist and Model 19 generalist (500 games each)
#   bash scripts/run_gauntlet.sh v7f         # Run only v7f specialist
#   bash scripts/run_gauntlet.sh model19     # Run only Model 19 generalist
#   GAMES=50 bash scripts/run_gauntlet.sh    # Fast test with 50 games

set -e
cd ~/ai-vgc

GAMES="${GAMES:-500}"
TARGET="${1:-both}"
POOL="${POOL:-data/teams/reg_mc}"

# Locate Model 19 Checkpoint
MODEL_19="data/models/model19_best_1054m_ema.pt"
if [[ ! -f "$MODEL_19" ]]; then
  if [[ -f "runs/heuristic19-blackwell-760236/main/model19_sp_1054343k_ema.pt" ]]; then
    MODEL_19="runs/heuristic19-blackwell-760236/main/model19_sp_1054343k_ema.pt"
  else
    LATEST=$(ls -t runs/heuristic19-blackwell-760236/main/model19_sp_*_ema.pt 2>/dev/null | head -n 1 || true)
    if [[ -n "$LATEST" && -f "$LATEST" ]]; then
      MODEL_19="$LATEST"
    fi
  fi
fi

# Locate v7f Checkpoint
MODEL_V7F="data/models/mc-cts-rnad-v7f-MC378.pt"

# Locate v6 Opponent Checkpoint
MODEL_V6="data/models/mc-cts-rnad-v6.pt"

# Locate Priors
PRIORS="data/public-priors.json"
if [[ ! -f "$PRIORS" ]]; then
  if [[ -f "runs/model19-beliefs-20261016/data/munchstats-priors.json" ]]; then
    PRIORS="runs/model19-beliefs-20261016/data/munchstats-priors.json"
  fi
fi

echo "======================================================================"
echo "THE GAUNTLET BENCHMARK: 500-TEAM POOL vs v6 GENERALIST"
echo "======================================================================"
echo "Total Games per Model: $GAMES (side-balanced: half P1, half P2)"
echo "Opponent Policy:       $MODEL_V6"
echo "Team Pool:             $POOL"
echo "Format:                gen9championsvgc2026regmc (CTS Bo1)"
echo "Target:                $TARGET"
echo "======================================================================"

# 1. Run v7f Specialist Gauntlet
if [[ "$TARGET" == "v7f" || "$TARGET" == "both" ]]; then
  echo ""
  echo ">>> [1/2] Benchmarking Specialist (v7f-MC378) against the Pool ($POOL)..."
  ~/.local/bin/uv run python scripts/eval_showdown_gauntlet_v7f.py \
    --games "$GAMES" \
    --model "$MODEL_V7F" \
    --opp-model "$MODEL_V6" \
    --team "MC378" \
    --pool "$POOL" \
    --concurrency 4
fi

# 2. Run Model 19 Generalist Gauntlet
if [[ "$TARGET" == "model19" || "$TARGET" == "both" ]]; then
  echo ""
  echo ">>> [2/2] Benchmarking Generalist (Model 19) against the Pool ($POOL)..."
  EXTRA_FLAGS=()
  if [[ "${SEARCH:-0}" == "1" ]]; then
    EXTRA_FLAGS+=(--search --depth 1 --search-k 20 --search-m 40 --search-k2 2 --search-m2 2)
    echo "  (Chance-branching search enabled: K=20, M=40)"
  fi

  ~/.local/bin/uv run python scripts/eval_showdown_stateful_shadow.py \
    --games "$GAMES" \
    --model-path "$MODEL_19" \
    --my-team "MC378" \
    --opp-team "pool" \
    --pool-dir "$POOL" \
    --opp-model "$MODEL_V6" \
    --belief-priors "$PRIORS" \
    "${EXTRA_FLAGS[@]}"
fi

echo ""
echo "======================================================================"
echo "Gauntlet evaluation completed successfully!"
echo "Results saved to showdown_gauntlet_*.json and showdown_eval_*.json"
echo "======================================================================"
