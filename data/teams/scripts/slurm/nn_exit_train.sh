#!/bin/bash -l
# Expert iteration training (idea 7 step 4) on one GPU. Extra arguments go to `ai_vgc.nn.exit train`:
#   cd ~/ai-vgc && sbatch scripts/slurm/nn_exit_train.sh --init data/models/archive/mb-bo3-rnad-v3.pt --data data/exit/r1 --out data/models/archive/mb-bo3-exit-v1.pt
#SBATCH --job-name=vgc-nn-exit-train
#SBATCH --output=logs/%x-%j.out
#SBATCH -G 1
#SBATCH --mem=64G
#SBATCH --time=6:00:00

cd ~/ai-vgc
nvidia-smi -L
~/.local/bin/uv run --extra nn python -m ai_vgc.nn.exit train "$@"
