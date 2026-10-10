# Native `vgc-engine` PPO / PFSP training

This is the pure-native rollout path for `ai_vgc.nn.rl`. It is designed for
throughput-sensitive PPO: battle stepping, team preview, policy legality, and
slot resets happen in `vgc-engine`; Python exchanges fixed packed NumPy arrays
with the policy rather than constructing `poke_env` battle objects.

This is **not** a claim that a checkpoint is ladder-ready by itself. Before
promotion, evaluate it against held-out opponents on Pokémon Showdown. The
native engine is the training simulator; Showdown remains the compatibility
oracle and final evaluation backend.

## Rollout contract

Use `--backend vgc-engine --native-slots N --native-horizon H` together.

- An actor owns `N` live `BattleBatch` slots for one fixed-horizon collection.
- Every normal policy wave evaluates all slots at a stable shape.
- A terminal slot is immediately reset and, when enabled, runs its own
  policy-controlled bring-four / lead preview. Other live slots are untouched.
- Forced replacement choices are regular policy decisions and are retained in
  the PPO trajectory.
- Only fully terminal trajectories are emitted. Unfinished horizon tails are
  deliberately discarded so the existing terminal GAE calculation cannot leak
  values or rewards across a slot reset.
- Native VGC policy actions retain the historical 107-action head. Tera actions
  stay masked because they are not legal in this project configuration.

For symmetric `self` jobs, both physical player perspectives are PPO rows. For
`bc` and `pool` league jobs, only the learner perspective is retained. The
learner alternates physical P1/P2 ownership by slot and flips after every reset;
the collector reports `learner_p1_games` and `learner_p2_games` in its rollout
statistics for balance auditing.

## Opponent mix and PFSP

The native backend accepts only:

```text
self, bc, pool
```

`nash`, `heuristic`, `human`, and `spec` deliberately remain unavailable for
native rollouts because their current implementations require Showdown players
or the Showdown double-oracle evaluator.

- `self` uses the current learner on both sides.
- `bc` uses the immutable `--init` checkpoint as a frozen opponent.
- `pool` samples a frozen snapshot using empirical PFSP. The learner's terminal
  reward against each snapshot is persisted in:

  ```text
  <out stem>_native_pfsp.json
  ```

  A snapshot with learner win rate `w` is selected proportional to
  `max((1 - w)^2, 0.001)`. Fresh snapshots start at a neutral `w = 0.5` prior;
  difficult frozen policies are therefore sampled more often, without starving
  any live snapshot permanently. `--resume` reloads the records. Snapshot
  records absent from the active pool are retired after each successful PPO
  iteration.

Pool-only startup needs `--pool-init`; alternatively give `self` or `bc` a
positive mix weight until `--snapshot-every` creates snapshots.

## Benchmark before spending a training allocation

Use the persistent collector benchmark on the **actual target GPU node**. CPU
smokes prove correctness but do not establish GPU throughput.

```bash
VGC_DEX_DIR=$HOME/Dev/localdex/data \
uv run --extra nn python scripts/benchmark_native_persistent.py \
  --device cuda \
  --model data/models/mc-cts-rnad-v7-MC301.pt \
  --teams data/teams/reg_mb \
  --format gen9championsvgc2026regmb \
  --team-preview \
  --slots 1024 2048 4096 8192 \
  --horizon 64 \
  --warmup 2 \
  --repeats 5
```

Repeat once with `--compile`. Enable `--native-compile` only when that measured
CUDA run improves completed games/s without exceeding the memory headroom.
The useful outputs are:

- **completed games/s** — the target metric, not policy states/s;
- **mean turns** — confirms the team/format workload did not change;
- **export/sample/step/reset/preview percentages** — identifies whether the
  bottleneck is packed export, model inference, Rust stepping, or refills;
- **peak CUDA memory** — leave headroom for the learner optimizer and actor
  checkpoint refresh.

For specialist frozen-opponent measurements:

```bash
VGC_DEX_DIR=$HOME/Dev/localdex/data \
uv run --extra nn python scripts/benchmark_native_persistent.py \
  --league \
  --learner-team data/teams/reg_mb/MB100.txt \
  --opponent-model data/models/mc-cts-rnad-v7-MC301.pt \
  --device cuda --team-preview \
  --slots 1024 2048 4096 --horizon 64 --warmup 2 --repeats 5
```

Do not report the 500 completed-games/s goal as achieved until this benchmark
has run on the target accelerator with production preview settings.

## Production invocation

The launcher reserves the learner GPU when multiple CUDA devices are visible.
With `--native-actors 0`, native actors default to all non-learner GPUs; on a
one-GPU machine a single actor intentionally shares the learner GPU. Set an
explicit actor count only after confirming device availability.

Example specialist-league run (first benchmark-derived values shown as
placeholders):

```bash
sbatch scripts/slurm/nn_rl.sh \
  --backend vgc-engine \
  --init data/models/mc-cts-rnad-v7-MC301.pt \
  --out data/models/native-regmb-pfsp-v1.pt \
  --format gen9championsvgc2026regmb \
  --teams data/teams/reg_mb \
  --my-team data/teams/reg_mb/MB100.txt \
  --native-opponent-teams data/teams/reg_mb \
  --mix self:0.20,bc:0.20,pool:0.60 \
  --pool-init data/models/mc-cts-rnad-v7-MC301.pt \
  --native-slots 4096 \
  --native-horizon 64 \
  --native-actors 11 \
  --native-preview \
  --native-max-turns 128 \
  --snapshot-every 5 \
  --iters 300 \
  --device cuda:0 \
  --mem-limit 0.85
```

Replace `4096`, `64`, and `11` with values proven by the benchmark. Do **not**
turn on `--native-compile` blindly; add it only after its benchmark wins.

`--games` is a quota only for finite native collection. Persistent mode submits
exactly one horizon per actor per PPO iteration; its collection size is driven
by actors × slots × horizon and actual battle duration.

The Slurm wrapper checkpoints every iteration and treats exit code 42 as a
memory-headroom restart. It resubmits with fewer `--native-actors` (and its
legacy vectorized controls) while adding `--resume`.

## Required validation gates

Run these before a long allocation or after modifying native rollout behavior:

```bash
VGC_DEX_DIR=$HOME/Dev/localdex/data \
uv run --extra nn pytest -q \
  tests/test_native_pfsp.py \
  tests/test_native_persistent.py \
  vgc_engine_repo/crates/vgc-engine-py/tests/test_battle_batch.py

VGC_DEX_DIR=$HOME/Dev/localdex/data \
uv run --extra nn python -m ai_vgc.nn.rl \
  --backend vgc-engine \
  --init data/models/mc-cts-rnad-v7-MC301.pt \
  --out /tmp/native-pfsp-smoke.pt \
  --iters 1 --workers 1 --native-actors 1 \
  --teams data/teams/reg_mb \
  --my-team data/teams/reg_mb/MB100.txt \
  --format gen9championsvgc2026regmb \
  --mix pool:1 \
  --pool-init data/models/mc-cts-rnad-v7-MC301.pt \
  --native-slots 2 --native-horizon 2 --native-max-turns 1 \
  --native-preview --device cpu --value-warmup 0 \
  --snapshot-every 100 --minibatch 16 --ppo-epochs 1 --lr 1e-6
```

For deterministic native regression measurements between frozen checkpoints,
use the side-balanced evaluator:

```bash
VGC_DEX_DIR=$HOME/Dev/localdex/data \
uv run --extra nn python scripts/eval_native_policy.py \
  --learner data/models/candidate.pt \
  --opponent data/models/mc-cts-rnad-v7-MC301.pt \
  --learner-team data/teams/reg_mb/MB100.txt \
  --teams data/teams/reg_mb --format gen9championsvgc2026regmb \
  --games 512 --slots 64 --horizon 128 --team-preview --device cuda
```

It reports learner score separately for physical P1 and P2, so a position bias
cannot masquerade as policy improvement. It remains a native-engine diagnostic:
perform an external held-out Showdown evaluation of each candidate checkpoint
before promotion. Native self-play and PFSP metrics measure training progress
only; they do not validate engine fidelity or ladder performance.

## Current limits

1. The snapshot pool is empirical PFSP, not a native Nash meta-game. A frozen
   checkpoint evaluator and a payoff-matrix solver are still needed for true
   PSRO/Nash selection.
2. Actor processes and cached models live for the whole training job, but a
   `BattleBatch` is rebuilt at each PPO horizon. That keeps terminal-only GAE
   correct; preserving unfinished episode tails across PPO updates is a separate
   algorithmic change.
3. Collection waits for all actors before PPO and sends compact arrays through
   multiprocessing queues. It is correct and measurable, but not yet an
   overlapped actor/learner pipeline.
4. No GPU benchmark from this machine exists. Do not infer CUDA performance
   from CPU functional tests.
