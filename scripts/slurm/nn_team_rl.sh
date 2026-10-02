#!/bin/bash -l
# Team specialists, one per top team (Slurm array: task i trains TEAMS[i]), in the CTS Bo1 format.
# Stage 1: each team against the whole Reg M-C pool, from the CTS model:
#   cd ~/ai-vgc && sbatch --array=0-9%3 --exclude=ilab1 scripts/slurm/nn_team_rl.sh
# Stage 2: each stage-1 specialist against the others (each on its own team, with its own model),
# plus some pool games so it doesn't forget the field:
#   cd ~/ai-vgc && STAGE=2 sbatch --array=0-9%3 --exclude=ilab1 scripts/slurm/nn_team_rl.sh
# STAGE=f: continue v7 with Speed and damage inference and set guesses in the features -> mc-cts-rnad-v7f-<team>.pt
# STAGE=s: continue the v7 specialist with Speed inference on (ai_vgc.speed) -> data/models/mc-cts-rnad-v7s-<team>.pt
# STAGE=h: stage 1 with mc-cts-opp-v6 (imitation of human closed-sheet play) as a training opponent
#   -> data/models/mc-cts-rnad-v7h-<team>.pt
# Out: data/models/mc-cts-rnad-v7-<team>.pt (stage 1), mc-cts-rnad-v8-<team>.pt (stage 2).
# Then scripts/slurm/nn_team_rr.sh ranks the teams. Extra arguments go to ai_vgc.nn.rl.
# %3: at most three at once. Slurm can still put all three on one node, and three at the default 32 workers
# hit ilab's per-user process limit ("RuntimeError: Resource temporarily unavailable"); ilab rejects
# --cpus-per-task, so each job runs 16 workers x 16 battles instead (same battles at once, half the processes).
#SBATCH --job-name=vgc-nn-team-rl
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH -G 1
#SBATCH --mem=96G
#SBATCH --time=24:00:00

cd ~/ai-vgc
# reg_mc_top best first (team_rank round 2); keep in step with scripts/bot.sh and nn_team_rr.sh.
# TEAMS="MC408 MC365 ..." overrides the list (array index i trains the i-th; files from data/teams/reg_mc).
TEAMS=(${TEAMS:-MC196 MC147 MC371 MC378 MC358 MC408 MC337 MC321 MC41 MC4})
TEAM=${TEAMS[$SLURM_ARRAY_TASK_ID]}
STAGE=${STAGE:-1}
MC=(--format gen9championsvgc2026regmc --teams data/teams/reg_mc --showdown pokemon-showdown-mc --closed-sheets)
if [[ $STAGE == 1 ]]; then
  INIT=${INIT:-data/models/mc-cts-rnad-v6.pt}
  OUT=data/models/mc-cts-rnad-v7-$TEAM.pt
  extra=(--iters 100)
elif [[ $STAGE == s ]]; then  # continue v7 with Speed inferred from turn order in the features
  INIT=${INIT:-data/models/mc-cts-rnad-v7-$TEAM.pt}
  OUT=data/models/mc-cts-rnad-v7s-$TEAM.pt
  extra=(--iters 60 --speed-inference)
elif [[ $STAGE == f ]]; then  # continue with all the closed-sheet features: Speed and damage inference, set guesses
  INIT=${INIT:-data/models/mc-cts-rnad-v7-$TEAM.pt}
  OUT=data/models/mc-cts-rnad-v7f-$TEAM.pt
  extra=(--iters 100 --speed-inference --damage-inference --set-guess)
elif [[ $STAGE == h ]]; then  # stage 1 with human-like play: mc-cts-opp-v6 as a training opponent
  INIT=${INIT:-data/models/mc-cts-rnad-v6.pt}
  OUT=data/models/mc-cts-rnad-v7h-$TEAM.pt
  extra=(--iters 100 --mix self:0.4,human:0.3,bc:0.15,heuristic:0.15 --human data/models/mc-cts-opp-v6.pt)
else
  INIT=data/models/mc-cts-rnad-v7-$TEAM.pt
  OUT=data/models/mc-cts-rnad-v8-$TEAM.pt
  specs=()
  for t in "${TEAMS[@]}"; do
    [[ $t == "$TEAM" ]] || specs+=("data/models/mc-cts-rnad-v7-$t.pt::data/teams/reg_mc/$t.txt")
  done
  extra=(--iters 60 --mix spec:0.6,self:0.2,heuristic:0.1,bc:0.1 --specialists "${specs[@]}")
fi
echo "stage $STAGE: team $TEAM from $INIT -> $OUT"
nvidia-smi -L
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
~/.local/bin/uv run --extra nn python -m ai_vgc.nn.rl --init "$INIT" --ref "$INIT" --out "$OUT" \
  --my-team "data/teams/reg_mc/$TEAM.txt" --reg-every 20 --workers 16 --concurrency 16 "${extra[@]}" "${MC[@]}" "$@"
