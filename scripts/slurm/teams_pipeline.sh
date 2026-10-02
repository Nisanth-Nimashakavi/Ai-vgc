#!/bin/bash
# Queue the team fine-tuning pipeline in one go (CTS Bo1, Reg M-C, top 10 teams):
#   cd ~/ai-vgc && bash scripts/slurm/teams_pipeline.sh
#
#   stage1    each team's specialist vs the whole pool           -> mc-cts-rnad-v7-<team>   (now)
#   stage2    each specialist vs the other nine specialists      -> mc-cts-rnad-v8-<team>   (after stage1)
#   rr        round robin of the v8 specialists: the best team                              (after stage2)
#   baseline  round robin with one model (mc-cts-rnad-v6) on every team, for comparison      (now)
#
# Job ids go to logs/team_jobs.txt. Check progress with:
#   bash scripts/slurm/teams_status.sh
set -euo pipefail
cd ~/ai-vgc
mkdir -p logs

for f in data/models/mc-cts-rnad-v6.pt data/teams/reg_mc_top/MC196.txt; do
  [[ -f $f ]] || { echo "missing $f (rsync from the laptop first)"; exit 1; }
done

stage1=$(sbatch --parsable --array=0-9%3 --exclude=ilab1 scripts/slurm/nn_team_rl.sh)
stage2=$(STAGE=2 sbatch --parsable --dependency=afterok:$stage1 --array=0-9%3 --exclude=ilab1 \
  scripts/slurm/nn_team_rl.sh)
rr=$(sbatch --parsable --dependency=afterok:$stage2 scripts/slurm/nn_team_rr.sh)
baseline=$(MODEL=data/models/mc-cts-rnad-v6.pt sbatch --parsable scripts/slurm/nn_team_rr.sh)

{
  echo "$(date '+%F %T')"
  echo "stage1=$stage1 stage2=$stage2 rr=$rr baseline=$baseline"
} | tee -a logs/team_jobs.txt
