#!/bin/bash -l
# Specialise one model per top team (Slurm array: task i trains TEAMS[i]) with rl.py --my-team,
# Bo3 as on the ladder (changing leads after a loss), against the whole Reg M-C pool:
#   cd ~/ai-vgc && sbatch --array=0-9%5 --exclude=ilab1 scripts/slurm/nn_team_rl.sh
# INIT (default mc-bo3-rnad-v5) is the start and first KL anchor; extra arguments go to ai_vgc.nn.rl
# (e.g. --resume after a timeout). Out: data/models/mc-bo3-rnad-v6-<team>.pt. Then scripts/slurm/nn_team_rr.sh.
#SBATCH --job-name=vgc-nn-team-rl
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH -G 1
#SBATCH --mem=96G
#SBATCH --time=24:00:00

cd ~/ai-vgc
# reg_mc_top best first (team_rank round 2); keep in step with scripts/bot.sh and nn_team_rr.sh.
TEAMS=(MC196 MC147 MC371 MC378 MC358 MC408 MC337 MC321 MC41 MC4)
TEAM=${TEAMS[$SLURM_ARRAY_TASK_ID]}
INIT=${INIT:-data/models/mc-bo3-rnad-v5.pt}
echo "team $TEAM from $INIT"
nvidia-smi -L
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
~/.local/bin/uv run --extra nn python -m ai_vgc.nn.rl --init "$INIT" --ref "$INIT" \
  --out "data/models/mc-bo3-rnad-v6-$TEAM.pt" --my-team "data/teams/reg_mc_top/$TEAM.txt" \
  --reg-every 20 --iters 100 --change-after-loss \
  --format gen9championsvgc2026regmcbo3 --teams data/teams/reg_mc --showdown pokemon-showdown-mc "$@"
