#!/bin/bash -l
# Bo3 adaptation (idea 8): SHARDS copies of `player` in a Bo3 format, N series in all against
# OPP. Extra arguments go to every copy; the comparison is one run with --adapt, one without:
#   cd ~/ai-vgc && sbatch scripts/slurm/nn_bo3.sh --model data/models/archive/mb-bo3-rnad-v3.pt --adapt
#   cd ~/ai-vgc && sbatch scripts/slurm/nn_bo3.sh --model data/models/archive/mb-bo3-rnad-v3.pt
# Results: logs/vgc-nn-bo3-<job>.out; add them up with scripts/bo3_sum.py. Keep SHARDS small: next to
# other jobs on one node, 20 copies (Python, Showdown and sim bridge each) hit the user process limit.
#SBATCH --job-name=vgc-nn-bo3
#SBATCH --output=logs/%x-%j.out
#SBATCH --mem=64G
#SBATCH --time=8:00:00

cd ~/ai-vgc
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
# Ports from the job id, so two jobs on one node never share a Showdown server (ports above 30000 fail on ilab).
BASE=$((10000 + SLURM_JOB_ID % 100 * 100))
SHARDS=${SHARDS:-8}
N=${N:-500}
OPP=${OPP:-nn:data/models/all-bo3-bc-v2.pt}
FORMAT=${FORMAT:-gen9championsvgc2026regmbbo3}
for s in $(seq 1 "$SHARDS"); do
  ~/.local/bin/uv run --extra nn python -m ai_vgc.nn.player --format "$FORMAT" --opponent "$OPP" \
    --n $((N / SHARDS)) --port $((BASE + s)) --concurrency 8 "$@" 2>"logs/vgc-nn-bo3-$SLURM_JOB_ID-$s.err" \
    | grep --line-buffered " vs \|Bo3:\|adapted" &
done
wait
for s in $(seq 1 "$SHARDS"); do
  grep -q . "logs/vgc-nn-bo3-$SLURM_JOB_ID-$s.err" 2>/dev/null || rm -f "logs/vgc-nn-bo3-$SLURM_JOB_ID-$s.err"
done
