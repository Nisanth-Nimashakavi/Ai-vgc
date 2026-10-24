#!/bin/bash -l
#SBATCH --job-name=vgc19-spec-mc378
#SBATCH --output=/common/home/nsn44/ai-vgc/runs/model19-specialist-mc378-regmc/logs/%x-%j.out
#SBATCH --partition=unlimited
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00

set -euo pipefail

RUN_DIR="/common/home/nsn44/ai-vgc/runs/model19-specialist-mc378-regmc"
PROJECT_ROOT="/common/home/nsn44/ai-vgc"
EXPLOITERS="$RUN_DIR/data/models/exploiters"
MODELS_DIR="$RUN_DIR/data/models"
LOGS_DIR="$RUN_DIR/logs"

mkdir -p "$LOGS_DIR" "$EXPLOITERS" "$MODELS_DIR"
cd "$PROJECT_ROOT"

export PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export VGC_DEX_DIR="$PROJECT_ROOT/.localdex/data"
export VGC_RAYON_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PY="$PROJECT_ROOT/.venv/bin/python"

WARMSTART="${WARMSTART:-/common/home/nsn44/ai-vgc/runs/heuristic19-blackwell-760236/main/model19_sp_1054343k_ema.pt}"
if [[ ! -f "$WARMSTART" ]]; then
  WARMSTART="/common/home/nsn44/ai-vgc/data/models/model19_best_1054m_ema.pt"
fi

TRAIN_POOL="/common/home/nsn44/ai-vgc/data/teams/pool_5600/train"
HOLDOUT_POOL="/common/home/nsn44/ai-vgc/data/teams/pool_5600/holdout_clean"
if [[ ! -d "$HOLDOUT_POOL" ]]; then
  HOLDOUT_POOL="/common/home/nsn44/ai-vgc/data/teams/pool_5600/holdout"
fi
SPECIALIST_TEAM="${SPECIALIST_TEAM:-/common/home/nsn44/ai-vgc/data/teams/reg_mc/MC378.txt}"
PRIORS="/common/home/nsn44/ai-vgc/data/public-priors.json"
if [[ ! -f "$PRIORS" ]]; then
  PRIORS="/common/home/nsn44/ai-vgc/runs/model19-beliefs-20261016/data/munchstats-priors.json"
fi

# Check for existing checkpoint to resume
LATEST_CKPT=$(ls -t "$MODELS_DIR"/model19_sp_*.pt 2>/dev/null | grep -v "_ema.pt" | head -n 1 || true)
if [[ -n "$LATEST_CKPT" ]]; then
    INIT_FLAG="--resume-from $LATEST_CKPT"
    echo "Found existing specialist checkpoint, resuming from: $LATEST_CKPT"
else
    INIT_FLAG="--warmstart-from $WARMSTART"
    echo "Fresh Model 19 specialist initialization, warmstarting from: $WARMSTART"
fi

echo "=== VGC 2026 Model 19 Specialist Training on MC378 vs 512 VGC Pastes ==="
echo "Node:            $(hostname)"
echo "GPU Allocation:  ${CUDA_VISIBLE_DEVICES:-auto}"
echo "Belief Priors:   $PRIORS"
echo "KL Anchor:       $WARMSTART"
echo "Specialist Team: $SPECIALIST_TEAM"
echo "Train Pool:      $TRAIN_POOL (512 tournament pastes)"
echo "Exploiters Dir:  $EXPLOITERS"
"$PY" -c 'import torch; print(f"PyTorch {torch.__version__}, GPU: {torch.cuda.get_device_name(0)}")' || true

exec "$PY" -m ai_vgc.nn.rl18 \
  --train-pool "$TRAIN_POOL" \
  --holdout-pool "$HOLDOUT_POOL" \
  --specialist-team "$SPECIALIST_TEAM" \
  --out-dir "$MODELS_DIR" \
  --belief-priors "$PRIORS" \
  --belief-particles 16 \
  $INIT_FLAG \
  --kl-anchor "$WARMSTART" \
  --kl-coef 0.05 \
  --timesteps 25000000 \
  --target-games 5000000 \
  --num-envs 2048 \
  --rollout-len 32 \
  --batch-size 2048 \
  --accum 2 \
  --ppo-epochs 1 \
  --lr 0.0001 \
  --entropy 0.02 \
  --aux-coef 0.25 \
  --ema-decay 0.999 \
  --save-interval 1000000 \
  --eval-interval 500000 \
  --eval-games 64 \
  --max-turns 60 \
  --batched \
  --no-double-buffer \
  --exploiter-dir "$EXPLOITERS" \
  --exploiter-prob 0.3 \
  --seed 20261018 \
  --device cuda:0
