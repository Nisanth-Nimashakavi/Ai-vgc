#!/bin/bash
# Queue the whole CTS (Bo1, closed team sheets) run in one go, each job waiting on what it needs:
#   cd ~/ai-vgc && bash scripts/slurm/cts_overnight.sh
#
#   dataset  closed-sheet data from every log              -> data/nn_cts
#   rl       self-play from mc-bo3-rnad-v5, closed sheets  -> mc-cts-rnad-v6        (now)
#   opp1     opponent model on every regulation            -> all-cts-opp-v6        (after dataset)
#   opp2     fine-tuned on Reg M-C                         -> mc-cts-opp-v6         (after opp1)
#   eval     mc-cts-rnad-v6 vs heuristic and v5, 1000 CTS games each              (after rl)
#   worlds   search with 1 vs 4 guesses at hidden sets, 400 CTS games each        (now)
#   final    mc-cts-rnad-v6 + search (mc-cts-opp-v6, 4 guesses) vs v5, 400 games  (after rl, opp2)
# Job ids are printed and saved to logs/cts_jobs.txt; results land in logs/vgc-nn-*-<id>.out.
set -euo pipefail
cd ~/ai-vgc
mkdir -p logs
MC=(--format gen9championsvgc2026regmc --teams data/teams/reg_mc --showdown pokemon-showdown-mc)
OPP_ARGS=(--aux-coef 0.2 --aux-mt --aug)
sub() { sbatch --parsable "$@"; }

dataset=$(sub scripts/slurm/nn_dataset.sh --closed --out data/nn_cts)
rl=$(sub --exclude=ilab1 scripts/slurm/nn_rl.sh --init data/models/mc-bo3-rnad-v5.pt \
  --out data/models/mc-cts-rnad-v6.pt --closed-sheets --reg-every 20 --iters 150 \
  --pool-init data/models/mc-bo1-rnad-v3.pt data/models/mc-bo3-rnad-v5.pt "${MC[@]}")
opp1=$(sub --dependency=afterok:$dataset scripts/slurm/nn_train.sh --init data/models/all-bo3-opp-v2.pt \
  --data data/nn_cts --out data/models/all-cts-opp-v6.pt --epochs 5 --lr 1e-4 "${OPP_ARGS[@]}")
opp2=$(sub --dependency=afterok:$opp1 scripts/slurm/nn_train.sh --init data/models/all-cts-opp-v6.pt \
  --data data/nn_cts --formats gen9championsvgc2026regmc gen9championsvgc2026regmcbo3 \
  --out data/models/mc-cts-opp-v6.pt --epochs 3 --lr 5e-5 "${OPP_ARGS[@]}")
eval=$(OPPS=data/models/mc-bo3-rnad-v5.pt EXTRA="${MC[*]} --closed-sheets" \
  sub --dependency=afterok:$rl scripts/slurm/nn_eval.sh data/models/mc-cts-rnad-v6.pt)
worlds=""
for w in 1 4; do
  worlds+=" $(N=400 OPP=nn:data/models/mc-bo3-rnad-v5.pt sub scripts/slurm/nn_search.sh \
    --model data/models/mc-bo3-rnad-v5.pt --opp-model data/models/all-bo3-opp-v2.pt --search-prior 0.1 \
    --search-worlds $w --closed-sheets "${MC[@]}")"
done
final=$(N=400 OPP=nn:data/models/mc-bo3-rnad-v5.pt sub --dependency=afterok:$rl:$opp2 scripts/slurm/nn_search.sh \
  --model data/models/mc-cts-rnad-v6.pt --opp-model data/models/mc-cts-opp-v6.pt --search-prior 0.1 \
  --search-worlds 4 --closed-sheets "${MC[@]}")

{
  echo "$(date '+%F %T')"
  echo "dataset $dataset  rl $rl  opp1 $opp1  opp2 $opp2  eval $eval  worlds(1,4)$worlds  final $final"
} | tee -a logs/cts_jobs.txt
