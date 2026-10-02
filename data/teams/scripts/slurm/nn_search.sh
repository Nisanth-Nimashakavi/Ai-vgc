#!/bin/bash -l
# Search vs no search (idea 7): SHARDS copies of `player --search`, each on its own Showdown
# server, N games in all against OPP (heuristic or nn:<checkpoint>). Extra arguments go to
# every copy:
#   cd ~/ai-vgc && sbatch scripts/slurm/nn_search.sh --model data/models/archive/mb-bo3-do-v1.pt --opp-model data/models/all-bo3-bc-v2.pt
#   cd ~/ai-vgc && OPP=heuristic sbatch scripts/slurm/nn_search.sh --model data/models/archive/mb-bo3-do-v1.pt
# Results: logs/vgc-nn-search-<job>.out (one line per copy; add them up). A copy that prints
# nothing left its errors in logs/vgc-nn-search-<job>-<copy>.err.
#SBATCH --job-name=vgc-nn-search
#SBATCH --output=logs/%x-%j.out
#SBATCH --mem=64G
#SBATCH --time=8:00:00

cd ~/ai-vgc
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
# Ports from the job id, so two jobs on one node never share (and stop) a Showdown server (RL uses 20000-29999; ports above 30000 fail to start on ilab).
BASE=$((10000 + SLURM_JOB_ID % 100 * 100))
SHARDS=${SHARDS:-20}
N=${N:-1000}
OPP=${OPP:-nn:data/models/archive/mb-bo3-do-v1.pt}
for s in $(seq 1 "$SHARDS"); do
  ~/.local/bin/uv run --extra nn python -m ai_vgc.nn.player --search --greedy --opponent "$OPP" \
    --n $((N / SHARDS)) --port $((BASE + s)) "$@" 2>"logs/vgc-nn-search-$SLURM_JOB_ID-$s.err" | grep --line-buffered " vs " &
done
wait
for s in $(seq 1 "$SHARDS"); do
  grep -q . "logs/vgc-nn-search-$SLURM_JOB_ID-$s.err" 2>/dev/null || rm -f "logs/vgc-nn-search-$SLURM_JOB_ID-$s.err"
done
