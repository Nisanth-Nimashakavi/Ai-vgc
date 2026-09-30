#!/bin/bash -l
# Round robin of the team specialists (ai_vgc.nn.team_rr): every pair of top teams plays SERIES Bo3
# series, each piloted by its own mc-bo3-rnad-v6-<team>.pt, both sides with the ladder options. SHARDS copies
# split the 45 pairs; the summed table (series win rate per team, head to head) ends the log:
#   cd ~/ai-vgc && sbatch scripts/slurm/nn_team_rr.sh
# MODEL=data/models/mc-bo3-rnad-v5.pt pilots every team with one model instead (the untrained baseline).
#SBATCH --job-name=vgc-nn-team-rr
#SBATCH --output=logs/%x-%j.out
#SBATCH --mem=64G
#SBATCH --time=8:00:00

cd ~/ai-vgc
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
TEAMS=(MC196 MC147 MC371 MC378 MC358 MC408 MC337 MC321 MC41 MC4)
BASE=$((10000 + SLURM_JOB_ID % 100 * 100))
SHARDS=${SHARDS:-8}
SERIES=${SERIES:-100}
entries=()
for t in "${TEAMS[@]}"; do entries+=("$t=${MODEL:-data/models/mc-bo3-rnad-v6-$t.pt}"); done
OUT=data/team_rr/$SLURM_JOB_ID
for s in $(seq 0 $((SHARDS - 1))); do
  ~/.local/bin/uv run --extra nn python -m ai_vgc.nn.team_rr --entries "${entries[@]}" --series "$SERIES" \
    --shard "$s/$SHARDS" --port $((BASE + s + 1)) --showdown pokemon-showdown-mc --out "$OUT/$s.csv" "$@" \
    >"logs/vgc-nn-team-rr-$SLURM_JOB_ID-$s.log" 2>&1 &
done
wait
~/.local/bin/uv run --extra nn python -m ai_vgc.nn.team_rr --sum "$OUT"/*.csv
