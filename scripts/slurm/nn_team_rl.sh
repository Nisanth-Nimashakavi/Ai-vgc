#!/bin/bash -l
# Team specialists, one per top team (Slurm array: task i trains TEAMS[i]), in the CTS Bo1 format.
# Stage 1: each team against the whole Reg M-C pool, from the CTS model:
#   cd ~/ai-vgc && sbatch --array=0-9%3 --exclude=ilab1 scripts/slurm/nn_team_rl.sh
# Stage 2: each stage-1 specialist against the others (each on its own team, with its own model),
# plus some pool games so it doesn't forget the field:
#   cd ~/ai-vgc && STAGE=2 sbatch --array=0-9%3 --exclude=ilab1 scripts/slurm/nn_team_rl.sh
# Out: data/models/mc-cts-rnad-v7-<team>.pt (stage 1), mc-cts-rnad-v8-<team>.pt (stage 2).
# Then scripts/slurm/nn_team_rr.sh ranks the teams. Extra arguments go to ai_vgc.nn.rl.
# %3: at most three at once; more RL jobs on one node hit ilab's per-user process limit.
#SBATCH --job-name=vgc-nn-team-rl
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH -G 1
#SBATCH --mem=96G
#SBATCH --time=24:00:00

cd ~/ai-vgc
# reg_mc_top best first (team_rank round 2); keep in step with scripts/bot.sh and nn_team_rr.sh.
TEAMS=(MC196 MC147 MC371 MC378 MC358 MC408 MC337 MC321 MC41 MC4)
TEAM=${TEAMS[$SLURM_ARRAY_TASK_ID]}
STAGE=${STAGE:-1}
MC=(--format gen9championsvgc2026regmc --teams data/teams/reg_mc --showdown pokemon-showdown-mc --closed-sheets)
if [[ $STAGE == 1 ]]; then
  INIT=${INIT:-data/models/mc-cts-rnad-v6.pt}
  OUT=data/models/mc-cts-rnad-v7-$TEAM.pt
  extra=(--iters 100)
else
  INIT=data/models/mc-cts-rnad-v7-$TEAM.pt
  OUT=data/models/mc-cts-rnad-v8-$TEAM.pt
  specs=()
  for t in "${TEAMS[@]}"; do
    [[ $t == "$TEAM" ]] || specs+=("data/models/mc-cts-rnad-v7-$t.pt::data/teams/reg_mc_top/$t.txt")
  done
  extra=(--iters 60 --mix spec:0.6,self:0.2,heuristic:0.1,bc:0.1 --specialists "${specs[@]}")
fi
echo "stage $STAGE: team $TEAM from $INIT -> $OUT"
nvidia-smi -L
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
~/.local/bin/uv run --extra nn python -m ai_vgc.nn.rl --init "$INIT" --ref "$INIT" --out "$OUT" \
  --my-team "data/teams/reg_mc_top/$TEAM.txt" --reg-every 20 "${extra[@]}" "${MC[@]}" "$@"
