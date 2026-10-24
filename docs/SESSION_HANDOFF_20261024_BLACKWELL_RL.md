# Multi-GPU Blackwell RL Scaling & League Training — October 24, 2026

## 1. Executive Summary & Cluster Infrastructure

- **Hardware Allocation:** 4× NVIDIA RTX PRO 5000 Blackwell GPUs on cluster node `ilab4` (partition: `unlimited`).
- **Active Job:** Slurm Job **`760236`** (`vgc-ema-blackwell`), scheduled for 12 hours (uninterrupted run through **10:00 AM EDT / 7:00 AM PDT**).
- **Run Directory:** `/common/home/nsn44/ai-vgc/runs/heuristic19-blackwell-760236`
- **Current Progress:** **`71,668,883` games** / **`567,803,904` environment steps** (Update 647).
- **Overnight Target:** **`125,000,000` games** / **`1,000,000,000` steps** (1.0 Billion steps).
- **Live Throughput:** **~2,400 – 2,550 games/sec** (`~600 – 636 g/gpu`, `~21,000 – 22,500 steps/sec`).

---

## 2. Key Operational & Engineering Findings

### A. Slurm Scheduler & CPU Allocation Policy
- **Rutgers CS Slurm Submission Rule:** Explicitly requesting `--cpus-per-task` causes immediate rejection:
  ```
  srun: error: Job rejected: Please do not specify the number of CPUs. There is actually no limit.
  ```
- **Scheduling Model:** The cluster scheduler (`TRESBillingWeights=gres/gpu=1.0`) allocates exclusively by GPU and does not enforce hard cgroup CPU quotas on tasks.
- **Rayon Thread Optimization:** With 4 DDP worker ranks per node, setting `VGC_RAYON_THREADS=16` created 64+ concurrent simulation threads. Under high host CPU contention (when other users run heavy workloads), this caused context-switching overhead. Reducing to **`VGC_RAYON_THREADS=8`** (32 total threads across 4 ranks) stabilized rollout latency and increased throughput to **>2,500 games/sec**.

### B. User Disk Quota Management on `/common/home`
- **Root Cause of Terminations:** Job `754188` terminated at Step 500M and Job `758157` exited with `0:53` because `/common/home` exceeded its 50 GB user quota (`51,418M / 51,200M`).
- **Pruning & Recovery:** 
  - Pruned dead intermediate uncompacted checkpoints from completed older runs (`runs/model19-beliefs-20261016/` and `runs/human19-from30m-lr-three/`), preserving final checkpoints and active artifacts.
  - Reduced used disk space from **`47,057 MB`** to **`39,511 MB`**, expanding free headroom to **`11,689 MB` (~11.7 GB free)**.
  - With automatic compaction (`--compact-checkpoints`), the 1.0B-step continuation requires only **~2.4 GB**, leaving almost 5× safety margin.

### C. Slurm Relative Path Trap
- Slurm evaluates relative `#SBATCH --output=logs/...` against the working directory from which `sbatch` is invoked. If submitted from `~`, it fails with `0:53`.
- **Fix:** Explicit `#SBATCH --chdir=/common/home/nsn44/ai-vgc` and absolute path `#SBATCH --output=/common/home/nsn44/ai-vgc/logs/blackwell-resume-%j.log` in all launch scripts.

### D. Multi-GPU NCCL Barrier Timeouts
- On Rank 0, 2,000-game holdout evaluations run periodically while other ranks wait at `dist.barrier()`.
- If GPU 3 shares compute with an active exploiter, the evaluation barrier can exceed the default 10-minute NCCL timeout.
- **Fix:** Configured `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800` and `NCCL_TIMEOUT=1800` (30 minutes).

---

## 3. Architecture & Mechanistic Dynamics

### A. Full Forward Batching
Inference in `rl18.py` does not operate on single battles:
- **Mirror Mode (Self-Play):** Both Player 1 and Player 2 perspectives across 8,192 parallel environments are concatenated along dimension 0 (`B = 16,384`), evaluated in a **single batched forward pass** through Transformer19 on each GPU.
- **League / Exploiter Mode:** Evaluated in two single batched forward passes (`B = 8,192` each).
- **Team Preview:** When battles terminate, all reset slots are evaluated in a single forward pass through the 180-plan Preview Head.
- **PPO Backprop:** Minibatches of 8,192 transitions in mixed precision (bfloat16 / TF32) with gradient accumulation of 2.

### B. Action Accounting: `Preview` vs. `Mega`
- **`Preview` (Episode-Level):** Counts only battles that terminated and received win/loss outcome credit (+1 / -1) backpropagated to the Preview Head during that specific 32-step rollout window.
- **`Mega` (Step-Level):** Counts every action where a Mega move is declared (`action >= 19`) across the 1,048,576 total decision opportunities per update ($4\text{ GPUs} \times 8,192\text{ envs} \times 32\text{ steps}$).
- **Mega Evolution Rates:**
  - **Per-Game Rate:** **`~88.2%`** of all battles feature a successful Mega Evolution by the learner.
  - **Per-Decision Step Rate:** **`~10.0%`** of all turn decisions across the environment pool are Mega move declarations.

---

## 4. Benchmark Performance & Evaluation History

### A. Evaluation Against Frozen Initial Network (2,000 Holdout Games Each)

| Step | Games | Raw Policy Win Rate | EMA Shadow Win Rate ($\beta=0.999$) | EMA vs Raw Delta | Note |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **510.1M** | ~64.5M | 1,550 / 2,000 (**77.50%**) | 1,562 / 2,000 (**78.10%**) | +0.60% | |
| **520.6M** | ~65.8M | 1,567 / 2,000 (**78.35%**) | 1,524 / 2,000 (**76.20%**) | -2.15% | |
| **530.1M** | ~67.0M | 1,587 / 2,000 (**79.35%**) | 1,574 / 2,000 (**78.70%**) | -0.65% | Peak raw policy win rate |
| **540.5M** | ~68.4M | 1,552 / 2,000 (**77.60%**) | 1,554 / 2,000 (**77.70%**) | +0.10% | |
| **551.0M** | ~69.7M | 1,548 / 2,000 (**77.40%**) | 1,571 / 2,000 (**78.55%**) | +1.15% | Baseline for Job 760236 |
| **560.5M** | ~70.8M | 1,558 / 2,000 (**77.90%**) | 1,591 / 2,000 (**79.55%**) | **+1.65%** | First run peak |
| **570.9M** | ~72.0M | 1,549 / 2,000 (**77.45%**) | 1,588 / 2,000 (**79.40%**) | **+1.95%** | Sustained high performance |
| **580.4M** | ~73.1M | 1,570 / 2,000 (**78.50%**) | 1,550 / 2,000 (**77.50%**) | -1.00% | Policy rebound after exploiter injection |
| **630.7M** | ~78.8M | 1,607 / 2,000 (**80.35%**) | 1,607 / 2,000 (**80.35%**) | 0.00% | First crossing of 80% threshold |
| **660.1M** | ~82.2M | 1,576 / 2,000 (**78.80%**) | 1,617 / 2,000 (**80.85%**) | **+2.05%** | Strong EMA smoothing gain |
| **760.7M** | ~92.8M | 1,611 / 2,000 (**80.55%**) | 1,591 / 2,000 (**79.55%**) | -1.00% | Policy lead |
| **860.4M** | ~103.4M | 1,596 / 2,000 (**79.80%**) | 1,623 / 2,000 (**81.15%**) | **+1.35%** | **All-time record peak win rate (81.15%)** |
| **980.9M** | ~116.8M | 1,619 / 2,000 (**80.95%**) | 1,620 / 2,000 (**81.00%**) | +0.05% | All-time policy peak |
| **1050.1M** | ~124.6M | 1,601 / 2,000 (**80.05%**) | 1,606 / 2,000 (**80.30%**) | +0.25% | Final 1.0B-step holdout evaluation |

### B. Showdown & Cross-Generation Benchmarks

| Model / Checkpoint | Eval Target | Format / Ruleset | Win Rate (%) | Wilson 95% CI | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Step 1054.3M EMA** | `v6` (`mc-cts-rnad-v6.pt`) | Showdown MC378 (50 games) | **80.00%** (40 / 50) | [67.03%, 88.81%] | **All-time record peak vs generalist v6** (+16.80% gain over 383M) |
| **Step 1054.3M EMA** | `v7f-MC378` (`mc-cts-rnad-v7f-MC378.pt`) | Showdown MC378 (50 games) | **28.00%** (14 / 50) | [17.51%, 41.67%] | Massive jump from early 45M checkpoint (6.00%) vs mirror specialist |
| **Step 860.4M EMA** | Frozen Initial Network | Pool 5,600 (Holdout 2,000 games) | **81.15%** (1,623 / 2,000) | [79.40%, 82.80%] | All-time holdout record |
| **Step 980.9M Policy** | Frozen Initial Network | Pool 5,600 (Holdout 2,000 games) | **80.95%** (1,619 / 2,000) | [79.18%, 82.60%] | All-time raw policy peak |
| **Step 1050.1M EMA** | Frozen Initial Network | Pool 5,600 (Holdout 2,000 games) | **80.30%** (1,606 / 2,000) | [78.50%, 81.98%] | Final milestone checkpoint |
| **Step 383.2M EMA** | `v6` (`mc-cts-rnad-v6.pt`) | Showdown MC378 (500 games) | **63.20%** (316 / 500) | [58.89%, 67.31%] | Outperforms human BC peak by +7.45% |
| **Step 320.3M Policy** | `v6` (`mc-cts-rnad-v6.pt`) | Showdown MC378 (500 games) | **60.40%** (302 / 500) | [56.05%, 64.59%] | Raw un-smoothed policy |
| **Step 383.2M + Search** | `v6` (`mc-cts-rnad-v6.pt`) | Showdown MC378 (50 games) | **68.00%** (34 / 50) | [54.19%, 79.24%] | Depth-2 Regret Matching Search (+4.80% gain) |
| **Human Imitation Peak** | `v6` baseline | Showdown MC378 (500 games) | **55.75% – 57.00%** | — | Pure behavioral cloning prior |

---

## 5. Job Progression Timeline

1. **Job `750394` (Initial Blackwell Scaling):**
   - Ran 0 to 50,032,333 games (383,254,528 steps) across 4× Blackwell GPUs in 1h 46m (`0:0` clean exit).
   - Saved final checkpoints: `model19_sp_383254k.pt` and `model19_sp_383254k_ema.pt`.
2. **Job `754188` (Continuation 1):**
   - Resumed from 383M to 500,695,040 steps (63,988,310 games). Exited at step 500M when disk quota was reached.
   - Checkpoints saved: `model19_sp_500695k.pt` and `model19_sp_500695k_ema.pt`.
3. **Job `758615` (Continuation 2):**
   - Resumed from 500.6M steps to 558,366,720 steps (70,617,081 games).
   - Produced `exploiter_500695040.pt` and saved `model19_sp_551026k.pt`.
4. **Job `760236` (Completed 1.0B-Step Milestone Run):**
   - Resumed from `model19_sp_551026k.pt` and ran to **1,054,343,168 steps / 125,105,327 games** in 7h 22m (`0:0` clean exit).
   - Reached all-time record peak holdout score of **`81.15%`** (Step 860M) and finished at **`80.30%`** (Step 1.05B).
   - Final checkpoints saved: `model19_sp_1054343k.pt` and `model19_sp_1054343k_ema.pt`.

---

## 6. Post-Run Search Upgrade: Chance Branching & Sequential Halving

Following the 1.054B run, the search architecture was upgraded from Monte Carlo seed averaging to exact analytical chance branching and sequential halving:
- **Full Architecture Specification Document:** [`docs/search_engine_architecture.md`](file:///home/nisanth/projects/ai-vgc/docs/search_engine_architecture.md)
- **Analytical Chance Branching:** Damage 3-bucket collapse ($\frac{5}{16}, \frac{6}{16}, \frac{5}{16}$), crits, accuracy, secondaries, and speed ties evaluated via `vgc_engine.multi_world_rollout_chance`. Zero simulator sampling variance.
- **Candidate Scaling & Halving:** $K=20$ our candidates vs $M=40$ opponent candidates (floor 12, preserving best slot 0 and slot 1 switch-out) progressively halved ($20 \to 10 \to 5 \to 3$ / $40 \to 20 \to 12$) with escalating world counts ($4 \to 8 \to 16 \to 32$).
- **Double Oracle Verification:** Dropped candidate audit restores any action beating the final equilibrium mix; full legal best-reply audit protects against unexpected exploits.
- **3% Win-Rate Cap:** Action selection samples from equilibrium mix but caps any candidate with $q(a) < q_{\max} - 0.06$.
- **Unit Test Suite:** [`tests/test_chance_branching_search.py`](file:///home/nisanth/projects/ai-vgc/tests/test_chance_branching_search.py) (5 passed in 16.10s).

