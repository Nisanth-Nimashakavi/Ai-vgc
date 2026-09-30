#!/bin/bash -l
# Team preview matrix-game test (ai_vgc.nn.preview), SHARDS copies at once with different
# matchups, each on its own Showdown server. Extra arguments go to every copy:
#   cd ~/ai-vgc && sbatch scripts/slurm/nn_preview.sh --model data/models/archive/mb-bo3-do-v1.pt --matchups 10 --games 64
#   cd ~/ai-vgc && sbatch scripts/slurm/nn_preview.sh --format gen9championsvgc2026regmc \
#     --showdown pokemon-showdown-mc --teams data/teams/reg_mc --games 64
# Results: logs/vgc-nn-preview-<job>.out (a summary per copy; add them up).
#SBATCH --job-name=vgc-nn-preview
#SBATCH --output=logs/%x-%j.out
#SBATCH --mem=64G
#SBATCH --time=8:00:00

cd ~/ai-vgc
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
# Ports from the job id, so two jobs on one node never share (and stop) a Showdown server (RL uses 20000-29999; ports above 30000 fail to start on ilab).
BASE=$((10000 + SLURM_JOB_ID % 100 * 100))
SHARDS=${SHARDS:-8}
for s in $(seq 1 "$SHARDS"); do
  ~/.local/bin/uv run --extra nn python -m ai_vgc.nn.preview --seed "$s" --port $((BASE + s)) "$@" \
    2>/dev/null | grep --line-buffered -E "^(matchup|solved|policy)" | sed -u "s/^/[$s] /" &
done
wait
