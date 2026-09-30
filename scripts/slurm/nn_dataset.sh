#!/bin/bash -l
# Build data/nn/*.npz from every log in data/battle_logs (CPU only). Extra arguments go to
# ai_vgc.nn.dataset, e.g. --out data/nn_ser for a copy with the Bo3 series context.
#   cd ~/ai-vgc && sbatch scripts/slurm/nn_dataset.sh
#SBATCH --job-name=vgc-nn-dataset
#SBATCH --output=logs/%x-%j.out
#SBATCH --mem=96G
#SBATCH --time=8:00:00

cd ~/ai-vgc
# One BLAS thread per worker: 32 workers x 64 threads hits the 2000-process limit.
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
~/.local/bin/uv run --extra nn python -m ai_vgc.nn.dataset --workers 32 "$@"
