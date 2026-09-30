#!/bin/bash -l
# Evaluate several checkpoints at once on CPUs (ilab has no CPU limit), each match on its own Showdown server.
# Arguments are the models; each plays --n greedy games vs the heuristic and vs each OPPS model:
#   cd ~/ai-vgc && sbatch scripts/slurm/nn_eval.sh data/models/archive/all-bo3-bc-v2-{base,aug,awr,aux}.pt
#   cd ~/ai-vgc && OPPS="data/models/archive/mb-bo3-do-v1.pt" N=2000 sbatch scripts/slurm/nn_eval.sh data/models/archive/mb-bo3-rnad-v3.pt
# EXTRA goes to every match, e.g. Reg M-C:
#   cd ~/ai-vgc && OPPS=data/models/archive/mb-bo3-do-v1.pt EXTRA="--format gen9championsvgc2026regmc --showdown pokemon-showdown-mc --teams data/teams/reg_mc" sbatch scripts/slurm/nn_eval.sh data/models/mc-bo1-rnad-v3.pt
# Results: logs/vgc-nn-eval-<job>.out (one line per match).
#SBATCH --job-name=vgc-nn-eval
#SBATCH --output=logs/%x-%j.out
#SBATCH --mem=64G
#SBATCH --time=4:00:00

cd ~/ai-vgc
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
OPPS=${OPPS:-data/models/archive/mb-bo3-bc-v1.pt}
N=${N:-1000}
# Ports from the job id, so two jobs on one node never share (and stop) a Showdown server (RL uses 20000-29999; ports above 30000 fail to start on ilab).
port=$((10000 + SLURM_JOB_ID % 100 * 100))
for model in "$@"; do
  for opp in heuristic $OPPS; do
    [[ $opp == heuristic ]] || opp=nn:$opp
    ~/.local/bin/uv run --extra nn python -m ai_vgc.nn.player --model "$model" --opponent "$opp" \
      --n "$N" --greedy --port $port $EXTRA 2>/dev/null | grep " vs " &
    port=$((port + 1))
  done
done
wait
