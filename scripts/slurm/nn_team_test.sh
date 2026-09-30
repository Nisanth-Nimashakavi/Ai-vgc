#!/bin/bash -l
# Test one team against every team in POOL (ai_vgc.nn.team_test): SHARDS copies split the pool,
# GAMES games against each pool team. The first argument is the team file; the rest go to every copy:
#   cd ~/ai-vgc && POOL=data/teams/reg_mc sbatch --exclude=ilab1 scripts/slurm/nn_team_test.sh data/teams/reg_mc/MC101.txt --model data/models/mc-bo1-rnad-v3.pt --format gen9championsvgc2026regmc --showdown pokemon-showdown-mc
# Results: data/team_tests/<job>/*.csv, summed at the end (worst matchups first) into logs/vgc-nn-team-test-<job>.out.
#SBATCH --job-name=vgc-nn-team-test
#SBATCH --output=logs/%x-%j.out
#SBATCH --mem=64G
#SBATCH --time=4:00:00

cd ~/ai-vgc
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
BASE=$((10000 + SLURM_JOB_ID % 100 * 100))
SHARDS=${SHARDS:-8}
GAMES=${GAMES:-8}
POOL=${POOL:-data/teams/reg_mb}
TEAM=$1; shift
OUT=data/team_tests/$SLURM_JOB_ID
for s in $(seq 0 $((SHARDS - 1))); do
  ~/.local/bin/uv run --extra nn python -m ai_vgc.nn.team_test "$TEAM" --pool "$POOL" --games "$GAMES" \
    --shard "$s/$SHARDS" --port $((BASE + s + 1)) --out "$OUT/$s.csv" "$@" \
    >"logs/vgc-nn-team-test-$SLURM_JOB_ID-$s.log" 2>&1 &
done
wait
echo "$TEAM $*"
~/.local/bin/uv run python scripts/team_sum.py "$OUT"/*.csv --top 25
