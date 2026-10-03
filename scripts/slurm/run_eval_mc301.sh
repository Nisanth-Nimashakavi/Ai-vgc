#!/bin/bash
# Eval Reg M-C MC301 models without search (v7f flags), 2000 games vs heuristic, opp-v6 and bc. Usage:
#   cd ~/ai-vgc && bash scripts/slurm/run_eval_mc301.sh data/models/mc-cts-rnad-v7fv-MC301.pt
# Small search eval (N games, extra player flags in MORE):
#   cd ~/ai-vgc && N=300 MORE=--search bash scripts/slurm/run_eval_mc301.sh data/models/mc-cts-rnad-v7fv-MC301.pt
export OPPS="data/models/mc-cts-opp-v6.pt data/models/all-bo3-bc-v2.pt"
export N=${N:-2000}
export EXTRA="--teams data/teams/reg_mc --only MC301 --opponent-teams data/teams/reg_mc --closed-sheets --format gen9championsvgc2026regmc --showdown pokemon-showdown-mc --speed-inference --damage-inference --set-guess $MORE"
sbatch scripts/slurm/nn_eval.sh "$@"
