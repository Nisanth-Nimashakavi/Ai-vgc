#!/bin/bash
# Closed-sheet feature + value pipeline for one team (run on ilab, not via sbatch):
#   1. RL with the new closed-sheet features (STAGE=f: speed and damage inference, set guesses) -> v7f
#   2. search-value data: games of v7f + search against mc-cts-opp-v6, every searched decision with
#      search's value for each candidate                                        (after 1)
#   3. train v7f's value head towards search's values -> v7fv                     (after 2)
#   cd ~/ai-vgc && TEAM=MC378 bash scripts/slurm/value_pipeline.sh
# Job ids go to logs/value_jobs.txt.
set -e
cd ~/ai-vgc
TEAM=${TEAM:-MC378}
MC=(--format gen9championsvgc2026regmc --teams data/teams/reg_mc --showdown pokemon-showdown-mc --closed-sheets)
FEAT=(--speed-inference --damage-inference --set-guess)
rl=$(env -u POOL TEAMS=$TEAM STAGE=f sbatch --parsable --array=0 --exclude=ilab1 scripts/slurm/nn_team_rl.sh)
gen=$(env -u POOL SHARDS=${SHARDS:-8} N=${N:-1600} ROUND=v-$TEAM OPP=nn:data/models/mc-cts-opp-v6.pt \
  sbatch --parsable --exclude=ilab1 --dependency=afterok:$rl scripts/slurm/nn_exit_gen.sh \
  --model data/models/mc-cts-rnad-v7f-$TEAM.pt --opp-model data/models/mc-cts-opp-v6.pt \
  --my-team data/teams/reg_mc/$TEAM.txt --search-worlds 4 --concurrency 2 "${MC[@]}" "${FEAT[@]}")
tr=$(sbatch --parsable --dependency=afterok:$gen scripts/slurm/nn_exit_train.sh \
  --init data/models/mc-cts-rnad-v7f-$TEAM.pt --data data/exit/v-$TEAM --out data/models/mc-cts-rnad-v7fv-$TEAM.pt \
  --value-only --search-value 0.7 --epochs 4 --lr 1e-4)
{
  echo "$(date '+%F %T') team=$TEAM"
  echo "rl=$rl gen=$gen train=$tr"
} | tee -a logs/value_jobs.txt
