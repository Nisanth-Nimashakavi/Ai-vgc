#!/bin/bash -l
# Round robin of the team specialists (ai_vgc.nn.team_rr): every pair of top teams plays SERIES
# CTS Bo1 games, each team piloted by its own specialist. SHARDS copies split the 45 pairs; the summed
# table (win rate per team, head to head) ends the log:
#   cd ~/ai-vgc && sbatch scripts/slurm/nn_team_rr.sh                 # stage-2 specialists (v8)
#   cd ~/ai-vgc && GEN=v7 sbatch scripts/slurm/nn_team_rr.sh          # stage-1 specialists
#   cd ~/ai-vgc && GEN="v7 v8" sbatch scripts/slurm/nn_team_rr.sh     # both, 20 entries (MC196@v7 ...):
#     includes every v7-vs-v8 pairing, mirrors too, so the table shows which stage pilots better
# MODEL=data/models/mc-cts-rnad-v6.pt pilots every team with one model instead (the baseline).
# The whole pool (378 teams, 71k pairs), one model, a few games per pair:
#   cd ~/ai-vgc && POOL=data/teams/reg_mc SERIES=4 SHARDS=16 sbatch scripts/slurm/nn_team_rr.sh
#SBATCH --job-name=vgc-nn-team-rr
#SBATCH --output=logs/%x-%j.out
#SBATCH --mem=64G
#SBATCH --time=8:00:00

cd ~/ai-vgc
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
teams_set=${TEAMS:-}
TEAMS=(${TEAMS:-MC196 MC147 MC371 MC378 MC358 MC408 MC337 MC321 MC41 MC4})  # TEAMS="..." overrides
BASE=$((10000 + SLURM_JOB_ID % 100 * 100))
SHARDS=${SHARDS:-6}
SERIES=${SERIES:-100}
entries=()
# TEAMS wins over POOL: an `export POOL=...` left over from nn_team_test.sh once turned a 19-team run
# into the whole 378-team pool at 100 series per pair.
if [[ -n ${POOL:-} && -z $teams_set ]]; then  # the whole pool, one model: POOL=data/teams/reg_mc MODEL=...
  entries=(--all "${MODEL:-data/models/mc-cts-rnad-v6.pt}" --teams "$POOL")
else
  entries=(--teams data/teams/reg_mc --entries)
  gens=(${GEN:-v8})
  for g in "${gens[@]}"; do
    for t in "${TEAMS[@]}"; do
      name=$t; (( ${#gens[@]} > 1 )) && name=$t@$g
      entries+=("$name=${MODEL:-data/models/mc-cts-rnad-$g-$t.pt}")
    done
  done
fi
OUT=data/team_rr/$SLURM_JOB_ID
for s in $(seq 0 $((SHARDS - 1))); do
  ~/.local/bin/uv run --extra nn python -m ai_vgc.nn.team_rr "${entries[@]}" --series "$SERIES" \
    --shard "$s/$SHARDS" --port $((BASE + s + 1)) --showdown pokemon-showdown-mc --format gen9championsvgc2026regmc --closed-sheets \
    --out "$OUT/$s.csv" "$@" \
    >"logs/vgc-nn-team-rr-$SLURM_JOB_ID-$s.log" 2>&1 &
done
wait
~/.local/bin/uv run --extra nn python -m ai_vgc.nn.team_rr --sum "$OUT"/*.csv
