#!/bin/bash -l
# Self-play RL (PPO) from the imitation policy: 1 GPU for the learner, CPU workers
# play the games on local Showdown servers. Extra arguments go to ai_vgc.nn.rl:
#   cd ~/ai-vgc && sbatch scripts/slurm/archive/mb-bo3-ppo-v1.sh
#   cd ~/ai-vgc && sbatch scripts/slurm/archive/mb-bo3-ppo-v1.sh --resume      # continue after a timeout
# Double oracle, continuing from the first RL run (do_rl):
#   cd ~/ai-vgc && sbatch scripts/slurm/archive/mb-bo3-ppo-v1.sh --init data/models/archive/mb-bo3-ppo-v1.pt --ref data/models/archive/mb-bo3-bc-v1.pt \
#     --pool-init data/models/archive/mb-bo3-bc-v1.pt data/models/archive/mb-bo3-ppo-v1_snapshots/iter_0{050,100,200}.pt data/models/archive/mb-bo3-ppo-v1.pt \
#     --mix self:0.3,nash:0.7 --out data/models/archive/mb-bo3-do-v1.pt --iters 200
# Reg M-C, R-NaD from the retrained BC model:
#   cd ~/ai-vgc && sbatch scripts/slurm/archive/mb-bo3-ppo-v1.sh --init data/models/all-bo3-bc-v2.pt --out data/models/mc-bo1-rnad-v3.pt \
#     --reg-every 20 --format gen9championsvgc2026regmc --teams data/teams/reg_mc --showdown pokemon-showdown-mc
# Specialise on one team (the learner always plays MC196; opponents draw from the pool):
#   cd ~/ai-vgc && sbatch --exclude=ilab1 scripts/slurm/archive/mb-bo3-ppo-v1.sh --init data/models/mc-bo1-rnad-v3.pt --ref data/models/mc-bo1-rnad-v3.pt \
#     --out data/models/mc-bo1-rnad-v3-mc196-selfmix.pt --my-team data/teams/reg_mc/MC196.txt --reg-every 20 --iters 100 \
#     --format gen9championsvgc2026regmc --teams data/teams/reg_mc --showdown pokemon-showdown-mc
# Bo3 (games 2-3 see the series so far; each game is its own episode), from the series model:
#   cd ~/ai-vgc && sbatch --exclude=ilab1 scripts/slurm/archive/mb-bo3-ppo-v1.sh --init data/models/all-bo3-bc-v4-series.pt \
#     --out data/models/mc-bo3-rnad-v5.pt --reg-every 20 --format gen9championsvgc2026regmcbo3 \
#     --teams data/teams/reg_mc --showdown pokemon-showdown-mc
#SBATCH --job-name=vgc-nn-rl
#SBATCH --output=logs/%x-%j.out
#SBATCH -G 1
#SBATCH --mem=96G
#SBATCH --time=24:00:00

cd ~/ai-vgc
nvidia-smi -L
node --version
# One BLAS/torch thread per worker process: many threads each hit the 2000-process limit.
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
~/.local/bin/uv run --extra nn python -m ai_vgc.nn.rl "$@"
