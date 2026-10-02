#!/bin/bash -l
# Expert iteration data (idea 7 step 4): SHARDS copies of `ai_vgc.nn.exit gen`, N search games in
# all against OPP (comma-separated; split evenly), each copy writing data/exit/$ROUND/<job>-<copy>.npz.
# Extra arguments go to every copy:
#   cd ~/ai-vgc && sbatch --exclude=ilab1 scripts/slurm/nn_exit_gen.sh --model data/models/archive/mb-bo3-rnad-v3.pt
# Then train on them with scripts/slurm/nn_exit_train.sh. Keep SHARDS small (user process limit).
#SBATCH --job-name=vgc-nn-exit-gen
#SBATCH --output=logs/%x-%j.out
#SBATCH --mem=64G
#SBATCH --time=12:00:00

cd ~/ai-vgc
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
# Ports from the job id, so two jobs on one node never share a Showdown server (ports above 30000 fail on ilab).
BASE=$((10000 + SLURM_JOB_ID % 100 * 100))
SHARDS=${SHARDS:-8}
N=${N:-4000}
ROUND=${ROUND:-r1}
OPP=${OPP:-nn:data/models/all-bo3-bc-v2.pt,nn:data/models/archive/mb-bo3-rnad-v3.pt}
for s in $(seq 1 "$SHARDS"); do
  ~/.local/bin/uv run --extra nn python -m ai_vgc.nn.exit gen --opponent "$OPP" --n $((N / SHARDS)) \
    --port $((BASE + s)) --out "data/exit/$ROUND/$SLURM_JOB_ID-$s.npz" "$@" \
    2>"logs/vgc-nn-exit-gen-$SLURM_JOB_ID-$s.err" | grep --line-buffered "^vs \|searched decisions" &
done
wait
for s in $(seq 1 "$SHARDS"); do
  grep -q . "logs/vgc-nn-exit-gen-$SLURM_JOB_ID-$s.err" 2>/dev/null || rm -f "logs/vgc-nn-exit-gen-$SLURM_JOB_ID-$s.err"
done
