#!/bin/bash -l
#SBATCH --job-name=vgc-nn-rl
#SBATCH --output=logs/%x-%j.out
#SBATCH -G 12
#SBATCH --mem=192G
#SBATCH --time=24:00:00

cd ~/ai-vgc
nvidia-smi -L
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
~/.local/bin/uv run --extra nn python -m ai_vgc.nn.rl "$@"
rc=$?

# rl.py exits 42 when a GPU got close to --mem-limit: it already checkpointed this
# iteration, so resubmit --resume with fewer --vector-workers/--servers instead of
# risking an OOM crash later in the run.
if [ "$rc" -eq 42 ]; then
    info=$(scontrol show job "$SLURM_JOB_ID")
    gpus=$(echo "$info" | grep -oP 'gres/gpu=\K[0-9]+' | head -1)
    mem=$(echo "$info" | grep -oP 'Mem[A-Za-z]*=\K[0-9]+[A-Za-z]*' | head -1)
    nodelist=$(echo "$info" | grep -oP 'BatchHost=\K\S+' | head -1)
    timelimit=$(echo "$info" | grep -oP 'TimeLimit=\K\S+' | head -1)
    jobname=$(echo "$info" | grep -oP 'JobName=\K\S+' | head -1)

    args=("$@")
    new_args=()
    has_resume=0
    for i in "${!args[@]}"; do
        case "${args[$i]}" in
            --vector-workers)
                old=${args[$((i+1))]}
                new=$(( old * 7 / 10 ))
                [ "$new" -lt 1 ] && new=1
                new_args+=("--vector-workers" "$new")
                ;;
            --servers)
                old=${args[$((i+1))]}
                new=$(( old * 7 / 10 ))
                [ "$new" -lt 1 ] && new=1
                new_args+=("--servers" "$new")
                ;;
            --native-actors)
                old=${args[$((i+1))]}
                new=$(( old * 7 / 10 ))
                [ "$new" -lt 1 ] && new=1
                new_args+=("--native-actors" "$new")
                ;;
            --resume) has_resume=1; new_args+=("${args[$i]}") ;;
            *)
                # skip values already consumed above
                prev="${args[$((i-1))]}"
                [ "$prev" = "--vector-workers" ] && continue
                [ "$prev" = "--servers" ] && continue
                [ "$prev" = "--native-actors" ] && continue
                new_args+=("${args[$i]}")
                ;;
        esac
    done
    [ "$has_resume" -eq 0 ] && new_args+=("--resume")

    echo "mem-limit hit: resubmitting with reduced workers (${new_args[*]})"
    sbatch -G "$gpus" --mem="$mem" --time="$timelimit" --nodelist="$nodelist" \
        --job-name="$jobname" --output="logs/${jobname}-%j.out" \
        "$0" "${new_args[@]}"
    exit 0
fi

exit $rc
