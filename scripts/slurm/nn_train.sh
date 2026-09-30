#!/bin/bash -l
# Train the policy network on one GPU. Extra arguments go to ai_vgc.nn.train:
#   cd ~/ai-vgc && sbatch scripts/slurm/nn_train.sh
#   cd ~/ai-vgc && sbatch scripts/slurm/nn_train.sh --d 384 --layers 6 --out data/models/all-bo3-bc-big.pt
#SBATCH --job-name=vgc-nn-train
#SBATCH --output=logs/%x-%j.out
#SBATCH -G 1
#SBATCH --mem=64G
#SBATCH --time=24:00:00

cd ~/ai-vgc
nvidia-smi -L
~/.local/bin/uv run --extra nn python -m ai_vgc.nn.train "$@"
