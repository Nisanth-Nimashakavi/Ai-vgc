#!/bin/bash -l
# Measure persistent native VGC rollout throughput on one accelerator.
# Example:
#   sbatch --gres=gpu:5000_Blackwell:1 scripts/slurm/benchmark_native_persistent.sh
#SBATCH --job-name=vgc-native-bench
#SBATCH --output=logs/%x-%j.out
#SBATCH --mem=96G
#SBATCH --time=2:00:00

set -euo pipefail
cd ~/ai-vgc
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PATH="$HOME/.cargo/bin:$PATH"

# iLab login shells do not load Cargo or consistently expose the local Dex.
# Resolve the data directory in the allocated job before rebuilding the Pyo3
# extension, and fail with useful diagnostics rather than benchmark stale code.
if [[ -z "${VGC_DEX_DIR:-}" ]]; then
  for candidate in "$PWD/.localdex/data" "$HOME/Dev/localdex/data" \
                   "$HOME/localdex/data" /common/home/nsn44/Dev/localdex/data; do
    if [[ -f "$candidate/moves.json" ]]; then
      export VGC_DEX_DIR="$candidate"
      break
    fi
  done
fi
: "${VGC_DEX_DIR:?set VGC_DEX_DIR to the localdex data directory}"
[[ -f "$VGC_DEX_DIR/moves.json" ]]

nvidia-smi -L
~/.local/bin/uv run --extra nn python -c 'import torch; print(f"torch={torch.__version__} cuda={torch.version.cuda} device={torch.cuda.get_device_name(0)}")'
~/.local/bin/uv run --extra nn maturin develop --release \
  --manifest-path vgc_engine_repo/crates/vgc-engine-py/Cargo.toml
~/.local/bin/uv run --extra nn python - <<'PY'
import vgc_engine
for name in ("reset_slots", "select_teams_at", "step_policy_action_array"):
    assert hasattr(vgc_engine.BattleBatch, name), name
print("native binding API ready")
PY

MODEL="${MODEL:-data/models/mc-cts-rnad-v7-MC301.pt}"
TEAMS="${TEAMS:-data/teams/reg_mb}"
FORMAT="${FORMAT:-gen9championsvgc2026regmb}"
# Start with a bounded diagnostic sweep. Override these at submission only
# after its results prove that a larger shape fits the target accelerator.
SLOTS="${SLOTS:-256 512 1024}"
HORIZON="${HORIZON:-16}"
WARMUP="${WARMUP:-1}"
REPEATS="${REPEATS:-2}"
MAX_TURNS="${MAX_TURNS:-128}"

common=(
  --device cuda --model "$MODEL" --teams "$TEAMS" --format "$FORMAT"
  --team-preview --slots $SLOTS --horizon "$HORIZON" --max-turns "$MAX_TURNS"
  --warmup "$WARMUP" --repeats "$REPEATS"
)

echo "===== eager ====="
~/.local/bin/uv run --extra nn python scripts/benchmark_native_persistent.py "${common[@]}"

echo "===== torch.compile ====="
~/.local/bin/uv run --extra nn python scripts/benchmark_native_persistent.py "${common[@]}" --compile
