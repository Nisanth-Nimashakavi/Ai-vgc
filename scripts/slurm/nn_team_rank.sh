#!/bin/bash -l
# Rank every team in POOL (ai_vgc.nn.team_rank): SHARDS copies, each ranking every SHARDS-th team,
# GAMES games per team against random pool teams. Extra arguments go to every copy:
#   cd ~/ai-vgc && POOL=data/teams/reg_mc sbatch --exclude=ilab1 scripts/slurm/nn_team_rank.sh --model data/models/mc-bo1-rnad-v3.pt --format gen9championsvgc2026regmc --showdown pokemon-showdown-mc
#   cd ~/ai-vgc && sbatch --exclude=ilab1 scripts/slurm/nn_team_rank.sh --model data/models/archive/mb-bo3-rnad-v3.pt
# Results: data/team_ranks/<job>/*.csv, summed at the end into logs/vgc-nn-team-rank-<job>.out.
#SBATCH --job-name=vgc-nn-team-rank
#SBATCH --output=logs/%x-%j.out
#SBATCH --mem=64G
#SBATCH --time=8:00:00

cd ~/ai-vgc
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
# Ports from the job id, so two jobs on one node never share a Showdown server (ports above 30000 fail on ilab).
BASE=$((10000 + SLURM_JOB_ID % 100 * 100))
SHARDS=${SHARDS:-8}
GAMES=${GAMES:-100}
POOL=${POOL:-data/teams/reg_mb}
OUT=data/team_ranks/$SLURM_JOB_ID
for s in $(seq 0 $((SHARDS - 1))); do
  ~/.local/bin/uv run --extra nn python -m ai_vgc.nn.team_rank --pool "$POOL" --games "$GAMES" \
    --shard "$s/$SHARDS" --port $((BASE + s + 1)) --out "$OUT/$s.csv" "$@" \
    >"logs/vgc-nn-team-rank-$SLURM_JOB_ID-$s.log" 2>&1 &
done
wait
~/.local/bin/uv run python scripts/team_sum.py "$OUT"/*.csv --top 40
