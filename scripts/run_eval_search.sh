#!/usr/bin/env bash
set -e

# Auto-locate model checkpoint if not in data/models/
if [ ! -f "data/models/model19_best_1054m_ema.pt" ]; then
    echo "Copying 1054M EMA checkpoint to data/models/model19_best_1054m_ema.pt..."
    mkdir -p data/models
    if [ -f "runs/heuristic19-blackwell-760236/data/models/model19_batched/model19_sp_1054343k_ema.pt" ]; then
        cp runs/heuristic19-blackwell-760236/data/models/model19_batched/model19_sp_1054343k_ema.pt data/models/model19_best_1054m_ema.pt
    fi
fi

# Auto-locate belief priors if not in data/
if [ ! -f "data/public-priors.json" ]; then
    echo "Copying public priors to data/public-priors.json..."
    mkdir -p data
    if [ -f "runs/model19-beliefs-20261016/data/munchstats-priors.json" ]; then
        cp runs/model19-beliefs-20261016/data/munchstats-priors.json data/public-priors.json
    fi
fi

OPP_MODEL="${1:-data/models/mc-cts-rnad-v7f-MC378.pt}"
OPP_TEAM="${2:-MC378}"
GAMES="${3:-50}"

echo "=========================================================="
echo "Running Model 19 Chance-Branching Search vs ${OPP_MODEL} (${GAMES} games)"
echo "=========================================================="

exec uv run python scripts/eval_showdown_stateful_shadow.py \
    --model-path data/models/model19_best_1054m_ema.pt \
    --opp-model "$OPP_MODEL" \
    --opp-team "$OPP_TEAM" \
    --belief-priors data/public-priors.json \
    --search \
    --depth 1 \
    --search-k 20 \
    --search-m 40 \
    --games "$GAMES"
