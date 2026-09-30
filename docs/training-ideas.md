# Training ideas beyond vgc-bench

vgc-bench covers behavior cloning, then pure self-play, fictitious play, double oracle and
exploiter training. The ideas below go past that. Most are borrowed from game-AI research in
other games, so they are new for VGC bots rather than new to reinforcement learning.

Where we are (Reg M-B, greedy, 1000 games):

| Model | vs heuristic | vs nn_lw1 | vs vgc-bench BC |
|---|---|---|---|
| nn_lw1 (our BC) | 83.8% | – | 75.7% |
| nn_rl (PPO self-play) | 93.0% | 65.4% | |
| nn_do (double oracle) | 91.6% | 67.8% | 86.7% |

nn_do beats nn_rl 54.4%. Self-play gains plateaued around iteration 100.

The ideas are numbered in the recommended order to complete them.


## 1. Move interactions the network can't see (Armor Tail and friends)

The bot misses basic interactions, such as using priority into Armor Tail. The cause is in
`encode.py`:

- **Abilities are only an ID embedding.** Nothing says "Armor Tail blocks priority". The model
  has to infer it from human games where Fake Out went into Farigiraf, which is rare, so the
  embedding is barely trained.
- **The damage features assume every move lands.** `_dmg` gives damage, KO chance and speed
  order as if the move connects, so Fake Out into an Armor Tail side reads as free chip damage
  plus a flinch.
- **Priority is a static number per move.** Nothing tells the model what cancels it.

Some of these rules, such as Lightning Rod / Storm Drain redirection and Good as Gold, were
already written for an earlier player; the network never sees them.

The fix has three layers, in the order to do them:

1. ~~**Find the rest from real games instead of guessing.** Showdown logs every blocked move:
   `-activate|…|ability: Armor Tail`, `-fail`, `-immune`, and redirections name the cause.
   Run a few thousand self-play games, count our bot's failed and blocked moves by cause, and
   write rules for the most common ones first.~~ **Done 2026-09-27** (`ai_vgc.nn.failures`,
   nn_do vs itself, 2000 games each in M-B and M-C, about 24 moves per game).

   About 21% of moves don't do what they were meant to, but most of that is reads, not rules:

   | Cause | M-B | M-C | Rule can fix? |
   |---|---|---|---|
   | Opponent Protected | 15.3% | 15.6% | no, a read |
   | Protect failed (used twice in a row) | 1.4% | 1.4% | discourage, not ban (1/3 chance) |
   | Sucker Punch failed (target didn't attack) | 1.4% | 1.4% | no, a read |
   | Single-target move into a type immunity (Close Combat into Ghost, Expanding Force into Dark) | 0.81% | 0.83% | **yes**, unless the target switches |
   | Wide Guard | 0.17% | 0.53% | no, a read |
   | Priority into Psychic Terrain (mostly Fake Out into Indeedee's side) | – | 0.25% | **yes** |
   | Priority into Armor Tail / Queenly Majesty (Fake Out, Sucker Punch, Aqua Jet) | 0.20% | 0.10% | **yes** |
   | Setup that can't work: Tailwind or screens already up, Stockpile at 3, Life Dew / Heal Pulse at full HP, Encore, Simple Beam, sleep moves into sleep | ~0.5% | ~0.1% | **yes** |
   | Status move into an immunity: Good as Gold (including our own ally's Life Dew / Helping Hand), Toxic into Steel/Poison, powder into Grass, Prankster into Dark | 0.14% | 0.06% | **yes** |
   | Ability immunities (Flash Fire, Levitate, ...) | 0.02% | 0.01% | **yes** |

   Spread moves with one immune target (Earthquake into a Flying ally or foe) are counted
   separately and aren't mistakes. Rule-fixable causes total about 1.5–2% of moves, roughly
   one blunder every two or three games.
2. ~~**Block failing moves at play time (no retraining).** Add
   `move_fails(battle, attacker, move, target)` and remove those actions from the mask in
   `NNPlayer.choose_move` before the network picks. Cases to cover first:~~
   - ~~priority moves into an Armor Tail / Dazzling / Queenly Majesty side;~~
   - ~~priority into a grounded target under Psychic Terrain;~~
   - ~~Prankster status moves into Dark types;~~
   - ~~Good as Gold against status moves;~~
   - ~~type and ability immunities (Levitate, Flash Fire, Water Absorb, Volt Absorb, Sap Sipper
     and similar);~~
   - ~~single-target moves that Lightning Rod / Storm Drain will redirect.~~

   ~~"Priority" means *effective* priority, including Prankster, Gale Wings, Triage and Grassy
   Glide in Grassy Terrain. Always leave at least one legal action, such as a switch or another
   move.~~ **Done 2026-09-27** (`ai_vgc/nn/rules.py`, on by default in `NNPlayer`; turn it off
   with `--no-rules`). It also covers Fake Out after the first turn, Air Balloon, Bulletproof /
   Soundproof / Wind Rider, and setup that can't work: Tailwind, screens, weather or terrain
   already up, Aurora Veil without snow, Stockpile at 3, heals at full HP.

   The rules drop about 0.6 actions per decision, and the bot never picks a move they flag.
   Failures in 2000 self-play games (M-B), before → after:

   | Cause | No rules | Rules |
   |---|---|---|
   | Immune by type | 0.79% | 0.69% |
   | Blocked by Armor Tail | 0.19% | 0.17% |
   | Stockpile / Life Dew / screens / weather / Reflect | 0.30% | 0.09% |
   | Good as Gold, Levitate, Soundproof, Helping Hand | 0.07% | 0.03% |

   Every leftover case checked was something no rule can see when choosing: the target
   switched in that turn (Sinistcha into Close Combat, Farigiraf into Fake Out) or Rage Powder
   redirected the move. Those are reads, which is what step 3 and search (idea 7) are for.
   Head-to-head, rules vs the same model without them, 1000 games: **50.0%** [46.9, 53.1] in
   M-B and **50.6%** [47.5, 53.7] in M-C. The fixed blunders are too rare (about one every
   15 games) to move the win rate. The rules stay on because they cost nothing and remove
   visible mistakes when a person plays the bot.
3. ~~**Teach the network (next retrain).** Add a "will fail / is blocked" flag to each
   move × target damage feature (`N_DMG` 8 → 9), then re-encode the dataset and retrain BC and
   RL. Masking alone stops the bad click but never teaches the knock-on effects, such as
   "no Fake Out pressure here, so lead differently".~~ **Done 2026-09-28** for BC
   (`nn_v2_base`: 86.1% vs heuristic, 53.0% vs nn_lw1; see the retrain table under
   "Recommended order"). RL on top of it is idea 3.

   The flag is
   `rules.foe_blocked`, asked for both sides' moves (so the network also sees that *their*
   Fake Out can't touch our Farigiraf side). It is set on about 6% of move × target rows, mostly
   Fake Out into Armor Tail or Ghosts, Dragon moves into Fairies and Prankster moves into
   Dark types. Old checkpoints read only the first 8 features, so `nn_do` still loads and plays.

Step 1 takes about 10 minutes and decides which rules step 2 needs first. Step 3 goes
into the next ilab retrain alongside ideas 2 and 3 below.

## 2. ~~Symmetry augmentation (cheapest)~~

**Done 2026-09-28, no gain on its own:** `nn_v2_aug` 84.0% vs heuristic, 49.4% vs nn_lw1
(base without it: 86.1%, 53.0%). It helps only combined with ideas 4 and 5 (`nn_v2_all`).

`train.py --aug`: It swaps our two slots and, independently, the
opponent's two, in a random half of each batch. Actions, targets, masks, damage rows and
opponent labels are swapped to match. Tested: applying it twice gives back the original
batch, and every label stays legal.

Swapping slot A and slot B, with actions and targets mirrored, gives an equally valid training
example. So does reordering the back Pokémon. This doubles the BC data for free and teaches the
network that these symmetries hold. BC currently overfits after about 12 epochs, so more data
is exactly what it needs.

## 3. ~~Regularized Nash dynamics (from DeepNash, the Stratego bot)~~

PPO already pulls the policy toward a fixed reference (nn_lw1) through the KL penalty. R-NaD
periodically *replaces* that reference with the current policy. That small change gives
convergence guarantees toward a Nash equilibrium in two-player zero-sum games, and it replaces
the snapshot pool. This targets the plateau around iteration 100.

Cost: about 20 lines in `rl.py`.

*Code ready 2026-09-27: `rl.py --reg-every N`.* The anchor is saved to `<out>_reg.pt` so
`--resume` keeps it. It runs after the BC retrain.

**M-B run done 2026-09-28** (`nn_v3.pt`: `--init nn_v2_all.pt --reg-every 20`, 300 iterations).
1000 greedy games each:

| nn_v3 vs | result |
|---|---|
| heuristic | 91.9% [90.0, 93.4] |
| nn_v2_all (its init) | 63.9% [60.9, 66.8] |
| nn_do | 49.0% [45.9, 52.1] |

A large gain over its starting point, but only a tie with nn_do, which had more RL (the old
pool run).

~~**Remaining runs.**~~ **Done 2026-09-28**, 1000 greedy games each:

| model | vs heuristic | vs nn_do | other |
|---|---|---|---|
| nn_v3, resumed to 600 iterations | 92.6% | **53.5%** [50.4, 56.6] | 54.1% vs nn_v3 at 300 |
| nn_v3_mc (M-C, 300 iterations) | 91.3% | **56.9%** [53.8, 59.9] (on M-C) | |
| nn_v3_do (pool with nn_do) | 89.7% | 40.9% [37.9, 44.0] | 43.3% vs nn_v3 at 300 |

R-NaD keeps improving with more iterations: nn_v3 at 600 is now the best M-B model, just ahead
of nn_do. Mixing nn_do snapshots into the pool hurt badly, so plain R-NaD from the BC model is
the recipe. nn_v3_mc is the M-C model. Against the human-like nn_v2_all, though, nn_v3 at 600
gets 64.5% [61.5, 67.4], the same as nn_v3 at 300 (63.9%) and nn_do (64.8%). The extra
iterations beat other bots, not human-style play.

## 4. ~~Offline RL on the human games~~

**Done 2026-09-28, the best single addition vs the heuristic:** `nn_v2_awr` (`--awr-beta 1`)
87.5% vs heuristic, 51.4% vs nn_lw1.

Instead of copying every human decision equally, weight each one by how much the value head
says it improved the win chance (advantage-weighted regression or IQL). This keeps the good
human moves and down-weights the blunders. It needs no extra games, only a change to the BC
loss.

*Code ready 2026-09-27: `train.py --awr-beta B`.* The advantage is V(next decision) − V(this
decision), with the final result after the last decision. V comes from nn_lw1's value head,
and the weight is capped at `--awr-max` (default 5).

## 5. ~~Auxiliary prediction heads~~ (opponent-action head done)

**Opponent-action head done 2026-09-28, no gain on its own:** `nn_v2_aux` (`--aux-coef 0.2`)
82.0% vs heuristic, 51.7% vs nn_lw1. It is kept because idea 7's search needs it, and it
helps inside `nn_v2_all`. The spread and turn-order heads are still open.

Self-play gives ground truth for extra heads that predict:

- the opponent's next action,
- the opponent's EV spread and speed from turn evidence (speed order, damage rolls),
- which side moves first.

These give the shared encoder a much richer training signal than win or loss alone. The
opponent-action head also directly improves the search in idea 7.

*Opponent-action head, code ready 2026-09-27: `train.py --aux-coef C`.* For each opposing
active it predicts which of its four moves it uses (or a switch) and whether it targets our
slot a, slot b or something else. The labels come from the logs (`replay.opp_labels`,
`opp` in the dataset). About 65% of turns are labelled; the rest are flinches, faints and
moves not yet revealed. The spread and turn-order heads are left for later.

## 6. ~~Team preview as its own game~~

**Tried 2026-09-28, no gain.** `ai_vgc.nn.preview` (ilab: `scripts/slurm/nn_preview.sh`): for
each matchup, nn_do's own preview plus its 3 next most likely previews per side, a 4 x 4
matrix of 64 nn_do-vs-nn_do games per cell, solved for side A's maximin mix, then 64 fresh
games of the solved mix and of nn_do's own preview against B's own preview. Over 32 M-B
matchups (2048 games each): **solved 48.4%, policy 47.7%**. nn_do's preview is already about
as good as this gets for its own battling (RL trained both together), and the top candidates
are usually close to tied, so the solver picks among noisy near-ties. Not worth distilling;
the 44% top-1 against humans mostly reflects previews that are about equally good.


Preview top-1 accuracy is only 44%. Preview is a one-shot 6v6 matrix game, and choosing the
wrong four can decide the match.

- For each matchup, score the candidate lead/back choices by playing them out with the
  current policy.
- Solve the matrix game and distil the result into the preview head.

This is cheap to do offline, and the 378 M-C teams in `data/teams/reg_mc/` give plenty of
matchups.

## 7. Search at play time, then train on the search (biggest expected gain)

Open team sheets make VGC almost a perfect-information game. You see the opponent's species,
moves, items, abilities and Tera types; only EV spreads and RNG are hidden. That makes search
far more practical than in hidden-information games.

- Each turn, copy the battle into Showdown's simulator.
- Take the network's top ~8 joint actions for each side and play out every pairing for one or
  two turns.
- Score the results with the value head, then solve the small matrix game with regret
  matching. `nash()` in `rl.py` already does this.
- Guess opponent EV spreads with the most common spread for that set.
- Training loop (expert iteration, as in AlphaZero): play with search, then train the network
  to imitate what the search chose. Each round gets better targets than raw self-play.

Parts of this already exist from earlier search work.

Progress:

1. ~~**Simulator bridge.**~~ **Done 2026-09-28** (`ai_vgc/nn/sim_bridge.js`, `search.position`).
   It rebuilds the position from what poke-env sees: open-sheet sets with the pool's spread
   (the most common one for that species), HP, status, boosts, Megas (including an opposing
   Mega, which poke-env keeps under the base species), items used up, weather, terrain, Trick
   Room, screens and Tailwind with turns left, Protect counters and Fake Out turns. The
   opponent's unseen back Pokemon are filled in from team preview. Each pairing's turn log is
   parsed into a copy of the live battle and scored by the value head, as in real play.
   Rebuilt states checked against poke-env on captured battles; no rejected choices in 300
   simulated turns.
2. ~~**One-turn search.**~~ **Done 2026-09-28** (`search.search`, `player --search`). Our
   policy's top 6 joint actions vs the opponent's top 6 (from the opponent-action head via
   `--opp-model`, or a random 6 without one), 2 seeds per pairing. The pick is the best expected
   value against the opponent's distribution. About 0.35 s per decision on one core.
3. ~~**Search vs no search.**~~ **Done 2026-09-28: a small gain.** nn_do+search vs nn_do, with the
   opponent head from nn_v2_all: 2 seeds 343/650 = 52.8%, 4 seeds 470/900 = 52.2%; together
   813/1550 = **52.5% [50.0, 54.9]**, at about 0.8 s (2 seeds) to 1.7 s (4 seeds) per decision
   on a busy ilab node. Without an opponent head (random replies): 46.5% over 200 games. More
   seeds didn't help; `--search-prior 0.1` is untested (its job lost its server, see below).
   (Missing shards in these runs were two jobs on one node sharing Showdown ports; the Slurm
   scripts now take ports from the job id.)

   Against the human-like nn_v2_all, the setting that matters, nn_do+search won 603/900 =
   **67.0% [63.9, 70.0]** vs plain nn_do's 64.8% [61.8, 67.7] (0.6 s per decision). That is
   +2.2 ± 4.3 points, the same small gain as above. On top of nn_v3 (plain nn_v3: 64.5% vs
   nn_v2_all), search got 517/800 = 64.6% [61.2, 67.9], a gain of zero. With
   `--search-prior 0.1` it got 472/700 = 67.4% [63.9, 70.8], about +3 points, but not
   significant (each run lost shards). About 0.9 s per decision. The goal is play against humans, so the
   opponent head stays trained on human games. Next fixes: a better human opponent head
   (weight games by rating, measure top-k recall on held-out replays), and a bigger
   `--search-prior` test.

   Checking the head first: `ai_vgc.nn.opp_eval` measures how often the human's real reply
   is among the head's top 6 joint replies (what search plays against) on the held-out games,
   split by rating, next to a head that knows nothing.

   **Result 2026-09-28** (58,889 held-out decisions):

   | head | move @1 | move+target @1 | joint moves @6 | joint move+target @6 | @12 |
   |---|---|---|---|---|---|
   | knows nothing | 19.7% | 7.8% | 24.0% | 3.6% | 7.2% |
   | nn_v2_all | 57.4% | 40.3% | 79.4% | 46.2% | 60.5% |
   | nn_v2_aux | 57.8% | 41.0% | 80.0% | 47.1% | 61.4% |
   | nn_v3 (RL never trained it) | 54.7% | 38.4% | 75.8% | 42.1% | 55.7% |

   The moves are predicted well: the real pair is in the top 6 about 80% of the time. The
   targets are the weak part: move+target top-1 is 40% against 57% for moves alone. Search only
   splits targets for single-target moves, so the real reply is in its 6 about 50–75% of the
   time; up to half the turns are planned against replies the human didn't make. Rating barely
   matters (most games are under 1300), so weighting by rating won't help much. Next: more
   opponent replies (`--search-opp-k 12`), then a target head that sees which move it is
   predicting a target for. nn_v2_aux is the best head to search with.

   *Per-move target head, code ready 2026-09-28: `train.py --aux-mt`.* The target is
   predicted from the opponent query plus that move's features, so Protect, spread moves and
   single-target attacks each get their own target; the loss uses the target of the move
   actually used. `opp_logits` now returns targets for each move (old heads repeat theirs),
   and search and `opp_eval` use the chosen move's targets. Retrain as nn_v2_all with the flag
   (`nn_v2_mt.pt`), then compare it with `opp_eval` and in search.

   ~~Retrain and measure.~~ **Done 2026-09-28:** nn_v2_mt vs nn_v2_aux on held-out human
   games: move+target @1 47.0% (was 41.0%), @3 82.6% (70.2%), joint move+target @6 **62.0%**
   (47.1%), @12 80.0% (61.4%). Move accuracy is unchanged (57.5%). The real reply is now in
   search's 6 far more often. ~~Next: search with `--opp-model nn_v2_mt.pt`.~~

   **Search with nn_v2_mt, 2026-09-28** (nn_v3 + `--search-prior 0.1` vs nn_v2_all; plain
   nn_v3 gets 64.5% [61.5, 67.4]):

   | opponent replies | result | per decision |
   |---|---|---|
   | 6 | 280/400 = 70.0% [65.3, 74.3] | 0.9 s |
   | 12 | 563/800 = **70.4% [67.1, 73.4]** | 1.6 s |
   | both | 843/1200 = 70.2% [67.6, 72.8] | |

   About **+6 points** over no search, a clear gain for the first time (the intervals don't
   overlap), and twice the old head's +3. The better target predictions are what made search
   work. Doubling the replies adds nothing yet. In the 6-reply job, 3 shards played without
   search (their bridge process died at the start: "0 decisions, 734 by policy"; those games
   are left out above). The bridge now restarts if it dies. Search passes the step-4 bar of
   about +2.5 points, so expert iteration is next.
   A later failure (2026-09-28): every `pokemon-showdown start` rebuilds `dist/`, and a job
   starting its servers broke the running search bridges ("conditions ... must export an
   object"). Each of those shards then fell back to the policy on every turn. The fixes:
   `ensure_server` passes `--skip-build` when `dist/` exists, and `SearchPlayer` restarts
   its bridge after 3 failed searches in a row. Leave out shards with many "by policy"
   decisions from 391410/391411.
   A second failure in the same runs: 2–3 copies per job stopped mid-run and never finished
   (killed after 75 minutes, while the others took 19). The bridge read had no time limit and
   blocks the whole event loop, so one stuck simulation froze a copy. The read now times out
   after 60 s, kills the bridge and falls back to the policy for that turn. Final totals:
   391410 (6 replies) 280/400 = 70.0%, 391411 (12 replies) 563/800 = 70.4%.
4. ~~**Expert iteration:**~~ (tried, shelved; see below) unblocked 2026-09-28 (search +6 points with nn_v2_mt). Built 2026-09-28
   in `nn/exit.py` and tested locally with nn_do (4 games, 29 decisions, 2 epochs).
   - **gen:** plays search games (greedy, k 6, 6 replies, prior 0.1). For every decision it
     searched, it records the inputs and search's k candidates, with each one's value and
     log-prob, plus the game result.
   - **train:** fine-tunes from `--init`. The target is softmax((value + 0.1·log-prob)/τ) over
     the candidates (τ 0.02; 0 means search's pick only).
     - A KL penalty to `--init` (0.1) keeps the policy near the human-trained one.
     - Value BCE on the result.
     - 5% of the data is held out; `agree` is how often the policy's favourite candidate is
       search's.
   - **ilab:** `nn_exit_gen.sh` (8 copies, 4000 games vs nn_v2_all and nn_v3 → `data/exit/r1`),
     then `nn_exit_train.sh`, then evaluate the new policy **without search** against nn_v3 and
     nn_v2_all. If it beats nn_v3, run gen again with it (round 2).
   - **Round 1 gen (392136):** all 8 copies finished cleanly in about 37 min: 28,273 searched
     decisions from 4000 games, and only 4 decisions fell back to the policy. The win rates were
     lost because the grep missed "vs" lines (fixed).
   - **Round 1 train:** 56,857 decisions from two gen runs, both with nn_v3 (392099 and 392136,
     8000 games); 2842 held out.
     - Before training, nn_v3's top candidate is search's pick 89% of the time, so search
       overrides it on about 11% of decisions.
     - nn_x1 (τ 0.02): CE fell from 0.569 to 0.500; KL 0.093; agree fell from 0.892 to 0.846.
     - nn_x1h (τ 0): CE fell from 0.554 to 0.492; KL 0.103; agree fell from 0.892 to 0.834.
     - Both became less sure of their top pick. Raising the probability of search's rare
       overrides lowers the loss even when it costs argmax agreement. Eval 392302 decides.
   - **Round 1 eval (392302, 2000 games each, greedy, no search):**

     | model | vs nn_v2_all | vs nn_v3 | vs heuristic |
     |---|---|---|---|
     | nn_x1h (τ 0) | 64.5% | 49.9% | 92.6% |
     | nn_x1 (τ 0.02) | 63.4% | 48.8% | 94.0% |

     Neither beats nn_v3 (64.5% vs nn_v2_all). Search's 6 points did not carry over.
   - **Next: learn only from overrides.** `--agree-weight 0` drops the ~89% of decisions where
     search kept the policy's pick from the search loss. The KL still covers them.
     - The new `overrides` metric is the share of held-out overrides the policy now makes itself
       (0 before training).
   - **Round 1b (override-only), 2000 games each, greedy, no search:**

     | model | agree-weight | overrides learned | vs nn_v2_all | vs nn_v3 | vs heuristic |
     |---|---|---|---|---|---|
     | nn_x1o | 0 | 37% | 63.4% | 47.7% | 92.3% |
     | nn_x1p | 0.1 | 30% | 63.2% | 50.2% | 92.0% |

     - Both learn about a third of search's held-out overrides, but they win no more. Held-out CE
       rises after epoch 1, so ~6k overrides are overfit.
     - Likely cause: search's edge comes from simulating the exact position (damage, speed,
       revealed sets) with small, noisy value gaps. A policy can't pick that up from 6k examples.
   - **Verdict (2026-09-28): shelved** at this scale. Search at play time keeps its +6 points.
     Revisit with ~10x the data, more seeds per search, and only overrides with a clear value
     margin.

## 8. Game-by-game adaptation in a Bo3

The bot can play Bo3 series (`ChallengeNNPlayer` in `nn/player.py`), but it plays each game
as if fresh. Humans adapt between games, and the losing player often wins game 2 by changing
leads. That makes this a real weakness in tournament play.

### What carries over between games

Open team sheets already show the opponent's six sets. A series adds what they actually *do*
with them:

- **Team preview choices:** which four they brought and which two they led, per game, and
  whether they changed after a loss.
- **Speed and bulk evidence:** turn order and damage rolls narrow down EV spreads. Examples:
  "their Incineroar outsped our 100-Speed Pokémon", "took 38–42% from X". Those bounds can
  replace the guessed spreads in the damage calc for games 2 and 3.
- **Habits:** do they Protect turn 1, when do they Tera or Mega Evolve, do they double-target
  or spread damage, do they switch out of bad matchups?
- **Series state:** game number, series score, and our own previous leads (they will adapt
  to those too).

### How to use it, cheapest first

1. **Carry spreads forward (no retraining).** Keep a per-series record of speed and damage
   bounds for each opponent Pokémon, and feed those into `damage_pct` and `effective_speed`
   in later games. Every damage and speed feature the network already reads becomes more
   accurate, with no model change.
2. **Adaptive team preview (no retraining).** Build on idea 6: treat preview as a matrix game,
   but weight the opponent's options by what they did earlier in the series (a Bayesian prior
   over their leads). After a loss, don't repeat the losing lead unless it's still clearly
   best. Keep some randomness, because a good opponent will counter-adapt to a bot that
   always "counters the last lead".
3. **Series context as a network input (retrain).** *Code ready 2026-09-29 (`nn/series.py`):
   see "Series context input" at the end.* Add a small context vector:
   - per opponent Pokémon: times brought, times led, inferred speed and bulk bounds;
   - our last leads;
   - game number and series score.

   Give it zeros in game 1. The network can then learn things like "they always lead Fake Out
   plus Tailwind, so open with Protect plus a spread move".
4. **Train on series, not just games.** For imitation, use Bo3 replays (tournament and ladder
   Bo3) so the network sees how humans change between games. For RL, play whole self-play
   series with a series-win reward, so adaptation, and not being exploitable by adaptation,
   is rewarded directly.

### Measuring it

Play Bo3 series against:

- a fixed opponent (does it win more in games 2 and 3 than in game 1?);
- an opponent that always repeats its game-1 leads (does it learn to counter them?);
- an opponent that counters our previous lead (does it avoid being predictable?).

Compare series win rate against the current no-memory bot.

### Progress

- ~~**Series scorecard.**~~ Done 2026-09-28. In a Bo3 format (`--format ...bo3`, `--n` counts
  series), `NNPlayer` reads each game's "Game N of <series>" banner. It records the result,
  both sides' leads and the opponent's four. `player.py` then prints the series won, the win
  rate per game number, and how often the opponent repeated its leads or four. Local test
  (nn_do vs nn_do, greedy, 40 series): series 16/40; game 1 50%, game 2 35%, game 3 5/12.
  The greedy opponent repeated its leads 100% of the time and its four 88%. Sampling
  opponents repeat much less (about 50–60% in a 6-series smoke test).
- **Adaptive preview (step 2, first version).** `nn/bo3.py`, enabled with `--adapt`. For games
  2 and 3, it takes the policy's 8 most likely previews and starts a simulated battle with
  each against the opponent's previous leads and four. The bridge's new "start" mode returns
  the start-of-battle log (leads out, Intimidate, weather). It scores each start position with
  the value head, plus 0.1 × the policy log-probability.
- ~~**How humans change between games.**~~ `scripts/human_bo3.py` over the scraped logs
  (58k series, Reg M-A/B/C):
  - Humans lead the same pair again 49% of the time after a win and 24% after a loss. They
    bring the same four 38% / 23% of the time.
  - The previous game's loser wins the next one 49.2% of the time. This holds whether the
    loser changed leads (48–50%) or kept them (49–50%).

  So, among humans, changing leads between games is not worth much. The gain from
  preview-level adaptation will be small, and the value is more likely in within-game reads
  (spreads, habits: steps 1 and 3).
- **Adaptive preview, second version.** The adapter now weights its read by how likely the
  opponent is to repeat: score = P(repeat) × (value − the policy pick's value) + 0.1 × log-prob.
  P(repeat) is 0.49 after their win and 0.24 after their loss. The test opponent can now repeat
  like a human: `--opponent-repeat 0.24 0.49` replays its last preview with those
  probabilities. The sampled bot already repeats about 45% of the time on its own, so the test
  opponent repeats more than humans do (58–73%).
- **Local A/B** (nn_do vs repeating nn_do, 150 series each): flat and noisy.
  - With the adapter: games 2–3 won 94/202 = 46.5%. It changed the policy's preview in 22 of
    202 games.
  - Without: 104/221 = 47.1%.
  - Game 1 is identical logic but came out 59% vs 44%, so the noise is ±7 points.
  - Needs about 1000 series per arm on ilab (`scripts/slurm/nn_bo3.sh`, sum the shards with
    `scripts/bo3_sum.py`).
- ~~**ilab A/B**~~ (2026-09-28, nn_v3 vs nn_v2_all repeating at human rates, 1000 series per
  arm; jobs 392083 / 392084):

  | | series | game 1 | game 2 | game 3 | games 2–3 |
  |---|---|---|---|---|---|
  | `--adapt` | 80.3% | 77.6% | 72.8% | 65.7% | 941/1324 = 71.1% |
  | plain | 84.0% | 78.7% | 76.0% | 72.7% | 973/1293 = 75.3% |

  **No gain; if anything worse.** The adapter changed the policy's top preview in only 36 of
  1324 games, far too few to explain the 4-point drop in games 2–3. The other difference: with
  `--adapt`, games 2–3 always play the single most likely preview (`preview_candidates`' top
  one), while plain play samples its preview. So the drop is most likely from giving up that
  sampling, not from the adaptation itself.

  Together with the human data (changing leads doesn't help the loser), preview-level
  adaptation is shelved. `--adapt` stays in the code, off by default. What's left of idea 8
  (carry spreads forward, series context input) is lower priority than the ladder and expert
  iteration.
- **`--vary` (2026-09-28): costs too much, off.** It samples games 2–3 so a human can't replay
  game 1's line. The preview comes from `preview_candidates`' top 4, and moves from those within
  0.02 of search's best score. Setup: nn_v3 + search vs nn_v2_all, M-B Bo3, 496 series per arm
  (jobs 393967 / 393968).

  | | game 1 | game 2 | game 3 | series |
  |---|---|---|---|---|
  | `--vary` | 68.3% | 56.0% | 50.3% | 62.3% |
  | plain | 69.0% | 71.2% | 70.5% | 75.4% |

  - Games 2–3 lose 15–20 points, far more than the 4 points `--adapt` lost with the same
    `preview_candidates` preview.
  - ~~Next: split the cost into preview and moves.~~ `--vary-margin 0` (preview only) scored
    game 2 63.3% and game 3 51.9%; `--vary-k 1` (moves only) scored 59.3% and 59.5%.
  - Both lost points, and `--vary-k 1` still goes through `preview_candidates`. That pointed at
    a **bug**, which is now fixed:
    - `preview_candidates` clears every `_selected_in_teampreview` flag. The `--vary` and
      `--adapt` previews never set them again.
    - Those flags are what the action mask (`encode.py`), the "brought" input and search's
      rebuilt position use to know which four were brought.
    - So in games 2–3 the bot saw no bench: no switches, and a wrong position for search.
    - Fix: `NNPlayer.teampreview` now sets the flags from whatever order was chosen. Its
      subclasses (`PreviewPlayer`) override `_choose_preview`.
    - The earlier `--adapt` result (−4 points) and any `PreviewPlayer --mix` results had the
      same bug and need a rerun before they can be trusted.
  - **After the fix:** `--vary` scored game 1 70.8%, game 2 64.9%, game 3 65.2% and series 72.2%.
    Plain scored 71.2%, 70.5% and 75.4%.
    - Games 2–3 still cost about 6 points, about 2 standard errors.
    - That's the whole price: against nn_v2_all, which never adapts, varying can't gain anything.
    - ~~Next: rerun the split.~~ The split after the fix and a plain control, all 496 series:

      | | game 1 | game 2 | game 3 | series |
      |---|---|---|---|---|
      | plain (393968) | 69.0% | 71.2% | 70.5% | 75.4% |
      | plain control (later) | 69.2% | 68.1% | 54.1% | 69.6% |
      | `--vary` | 70.8% | 64.9% | 65.2% | 72.2% |
      | preview only (395005) | 66.3% | 65.7% | 53.3% | 66.9% |
      | moves only (395006) | 63.3% | 66.5% | 53.5% | 65.9% |

    - The code is fine: search fell back to the policy almost never, and only 0.1–0.4% of the
      simulated pairings failed.
    - Two runs of plain play differ by 6 points per series and 16 in game 3. Games in one series
      share teams, so 500 series is noisier than the binomial error suggests. Treat about ±3
      points per game as noise.
    - Game 2 minus game 1 in the same run cancels out run-to-run conditions: plain +2.2 and −1.1,
      full `--vary` −5.9, preview only −0.6, moves only +3.2. Either half alone looks about free.
    - ~~Next: 2000 series each of plain vs moves-only `--vary`.~~ Done (395067 plain, 395068
      `--vary --vary-k 1`):

      | | game 1 | game 2 | game 3 | series |
      |---|---|---|---|---|
      | plain | 68.8% | 68.4% | 57.9% | 70.6% |
      | moves only | 69.6% | 67.6% | 60.7% | 71.3% |

    - **Moves-only `--vary` is free:** every difference is within about 1 standard error, since
      each game-1/game-2 figure is ±1 point. It's on for the ladder / challenge bot
      (`--vary --vary-k 1`), so games 2–3 don't replay game 1's lines move for move.
- **`--adapt` rerun after the flags fix (2026-09-29, nn_v3 vs nn_v2_all, `--opponent-repeat 0.24 0.49`;
  396978 plain, 396981 `--adapt`, 7 of 8 shards read):**

  | | series | game 1 | game 2 | game 3 | games 2–3 |
  |---|---|---|---|---|---|
  | plain | 84.0% | 76.6% | 76.6% | 72.4% | 1005/1330 = 75.6% |
  | `--adapt` | 81.8% | 76.6% | 76.0% | 67.5% | 852/1152 = 74.0% |

  - Still no gain: −1.6 points in games 2–3, within 1 standard error (±1.7). Game 1 matches exactly.
  - The adapter changed the policy's preview in only 27 of 1152 games, so it can't move the result.
    Preview-level adaptation stays shelved. The ladder's G2/G3 drop needs within-game reads instead.

## Recommended order

1. **Now, no retraining:** idea 1's failure log and play-time rules. They are quick and fix
   visible blunders.
2. ~~**Next ilab retrain:** idea 1's "blocked" feature plus ideas 2–5. They are all changes to
   the data, loss or heads, so one re-encode and retrain covers them. Symmetry augmentation
   targets the overfitting, R-NaD the self-play plateau, offline RL the human blunders, and
   the auxiliary heads the encoder's training signal.~~ **BC part done 2026-09-28**; R-NaD
   (idea 3) is next.

   BC retrain, all with the blocked feature and `--loss-weight 1.0` (Reg M-B, greedy, 1000
   games, rules on for both sides):

   | Model | Added | vs heuristic | vs nn_lw1 |
   |---|---|---|---|
   | nn_lw1 (old) | – | 83.8% | – |
   | nn_v2_base | blocked flag | 86.1% [83.8, 88.1] | 53.0% [49.9, 56.1] |
   | nn_v2_aug | + symmetry aug | 84.0% [81.6, 86.1] | 49.4% [46.3, 52.5] |
   | nn_v2_awr | + AWR (β=1) | **87.5%** [85.3, 89.4] | 51.4% [48.3, 54.5] |
   | nn_v2_aux | + opponent head | 82.0% [79.5, 84.3] | 51.7% [48.6, 54.8] |
   | nn_v2_all | all three | 85.9% [83.6, 87.9] | **54.2%** [51.1, 57.3] |

   Every gain is a few points and mostly inside the noise. Only `nn_v2_all` beats nn_lw1 with a
   confidence interval clear of 50%, so it is the init for the R-NaD run. BC changes are
   small next to what RL added (nn_do: 91.6%, 67.8%).
3. ~~**Team preview (idea 6):**~~ Tried, no gain (see idea 6). offline matchup solving on the retrained policy. It is also the
   base for the adaptive preview in idea 8.
4. **Search plus training on its results (idea 7):** the one expected to move well past nn_do
   rather than add a few points. It uses the opponent-action head from idea 5.
5. **Bo3 adaptation (idea 8):** its steps 1 and 2 (carrying spreads forward, adaptive preview)
   need no retraining and can go in any time once idea 6 exists. Steps 3 and 4 fit a later
   retrain once Bo3 replays are collected.

## Team choice (2026-09-28)

Ladder losses on M-C looked like bad matchups, not bad play: the bot draws a random team from the
pool each game.

- `nn/team_test.py` locks one team and plays every pool team the same number of games. It prints
  the overall win rate and the worst matchups (`scripts/slurm/nn_team_test.sh` on ilab).
- `nn/team_rank.py` ranks the whole pool: each team plays random pool teams, and the same model
  plays both sides, so an average team scores 50% (`scripts/slurm/nn_team_rank.sh`).
- `scripts/team_sum.py` adds up the shards and copies the top teams into a folder for
  `player --teams`.
- ~~Next: rank M-C, run a longer second round on the top 40~~ (done, below), then ladder with the
  top teams only.
- **Round 1 (nn_v3_mc both sides, 100 games per team vs random pool teams):** overall 49.8%,
  which is correct for same-model play.
  - The top is mostly rain: Pelipper or Politoed, Archaludon and Golisopod (MC408 77%, MC142,
    MC100 and MC371 67–70%).
  - Only the top few stand clear of chance. With 378 teams, the best 50% team would score about
    65% by luck alone, so round 2 plays 400 games each on the top 40.
- **Trick Room team (Indeedee-F / Kingambit / Camerupt / Farigiraf / Hatterene / Incineroar),
  8 games vs each pool team:** 53.3% [51.6, 55.1], a little above average.
  - Worst matchups over-represent Milotic (53% of the worst 15 vs 16% of the pool; Competitive
    punishes Intimidate), Salamence (53% vs 38%) and sand (Tyranitar/Excadrill 27% vs 8–10%).
  - Games in one matchup are not independent: both sides greedy with fixed teams means the
    same leads every game, so a 0/8 can be one losing line played 8 times.
- **Round 2 (top 40 + Trick Room, 400 games each vs random pool teams):** overall 58.2%.
  - Most teams moved back toward the mean, as expected. MC408 fell from 77% to 64.8%, and MC142
    from 70% to 53%.
  - The best are MC196 73.8% [69.2, 77.8], MC147 70.2%, MC371 67.0%, MC378 66.0%, MC358 65.2%,
    MC408 64.8% and MC337 64.5%.
  - Rain (Pelipper or Politoed, Archaludon, Golisopod) holds 6 of the top 6 spots. MC337 (Raichu /
    Salamence / Rillaboom / Gholdengo) is the best non-rain team.
  - Trick Room scored 53.2%, the same as its team test: average, not a pick.
  - The top 10 were copied to `data/teams/reg_mc_top` for the ladder bot.

## Fixes (2026-09-29)

- ~~**`Invalid action [0 0] … /choose pass and /choose pass are incompatible`.**~~ Fixed.
  - When both actives faint and one Pokémon is left, poke-env's mask lets each slot switch it
    in or pass. The policy could pick pass twice, Showdown rejected it, and poke-env played a
    random move.
  - Fix: `player.action_mask` removes pass from slot a in that case. Slot b then can only pass,
    because the joint mask already stops a double switch-in.
  - `NNPlayer` and `SearchPlayer` both use `action_mask`, so the fix covers search, the ladder,
    challenges and RL self-play.
- **Ladder bot (2026-09-29).**
  - `player --accept NAME --ladder --server showdown` plays Bo3 ladder games; we only ladder Bo3.
    It prints a `watch:` link for each battle, and `--watch` opens it in the browser.
  - Each game also goes to `data/bot_logs/logs_<format>.json`, laid out like the scraped logs but
    kept out of `data/battle_logs`.
    - It includes the `|showteam|` team sheets. poke-env drops them, so `ChallengeMixin` adds
      them back.
    - Build training data from them separately:
      `nn.dataset --logs data/bot_logs --out data/nn_bot`.

## Team specialisation (2026-09-29)

- [x] `rl.py --my-team FILE`: the learner always plays that team, opponents (self copy, BC, pool, heuristic) still draw from `--teams`; self-play trains on the learner's side only.
- [x] Run overnight, 200 iters each (snapshots every 10 in `<out>_snapshots/`, so 100 and 150 can be evaluated too):
  - A: default mix (self 0.5, bc 0.2, pool 0.15, heuristic 0.15) → `data/models/nn_mc196.pt`
  - B: BC-heavy mix `self:0.3,bc:0.5,pool:0.2`, since nn_v3_mc is the closer stand-in for humans → `data/models/nn_mc196_bc.pt`
  - Evals are chained with `afterok`, and the nn_v3_mc baseline runs right away.
- [x] Eval: team_test MC196 vs the reg_mc pool, `--model nn_mc196 --opponent-model nn_v3_mc`, against the baseline (nn_v3_mc both sides: 73.8% in round 2; rerun it alongside).
- Result (MC196 vs 378 reg_mc teams × 8, opponents piloted by nn_v3_mc, 3024 games):

  | MC196 pilot | win % |
  |---|---|
  | nn_v3_mc (baseline) | 69.3 [67.7, 71.0] |
  | A nn_mc196 (default mix) | 94.4 [93.5, 95.1] |
  | B nn_mc196_bc (bc 0.5) | 94.9 [94.1, 95.7] |

  +25 points is too big to take at face value. Both runs trained directly against nn_v3_mc (the `bc` opponent), which is also the opponent in this eval, so they may just exploit it.
- [x] Fair check: the same eval with nn_v2_all (the human proxy, never trained against) piloting the pool.
- Fair check result (nn_v2_all pilots the pool; 3024 games each):

  | MC196 pilot | win % |
  |---|---|
  | nn_v3_mc (baseline) | 80.5 [79.1, 81.9] |
  | A nn_mc196 | 85.4 [84.1, 86.6] |
  | B nn_mc196_bc | 87.6 [86.4, 88.8] |

  Specialising is a real gain, +7 for B. Training mostly against nn_v3_mc (bc 0.5) beat the default mix.
- [x] If it gains a few points: use it for ladder (bot.sh with `--teams` = MC196 only).

## Ladder night 1 (2026-09-28 23:54 → 09-29 08:27, M-C Bo3, reg_mc_top, nn_v3_mc + search, --vary-k 1)

- 71 series: won 40, lost 29, 2 unfinished. 170 games: 98-72 (57.6%). 40 of the 98 wins were opponent forfeits.
- By game in the series: G1 51/71 (72%), G2 34/70 (49%), G3 13/29 (45%). Humans adapt after G1 and we don't: after a G1 win we took G2 only 27/50.
- Rating: 1041 → peak 1324 → ~1200. Vs 1000s 25/38, 1100s 44/67, 1200s 16/41, 1300s 13/20, 1400s 0/4.
- Best team: rain Charizard/Politoed/Archaludon (MC196-style), 19/26. Tyranitar/Sinistcha 13/26, Raichu/Primarina 12/23.
- Toughest opposing mons: Arcanine-Hisui (lost 67% of 18), Garchomp (55% of 62), Raichu (52% of 58).
- [ ] Main gap is G2/G3: rerun `--adapt` (earlier result was hit by the flags bug) and consider opponent-conditioned play in games 2 and 3.
- [x] Ladder with MC196 only (bot.sh default now: data/teams/mc196 + nn_mc196_bc; `--general` for the old setup), since rain did best here too.

## Opponent-conditioned play in games 2–3 (2026-09-29)

- **Move habits don't help much.** `scripts/opp_habits.py` replays games 2–3 of held-out human
  M-C Bo3 series (400 games, 7409 opposing slot actions). It blends nn_v2_mt's move head with
  what that Pokémon picked earlier in the series:

  | | n | head alone | best blend (boost 0.25) |
  |---|---|---|---|
  | all | 7409 | 56.3% top-1, NLL 1.066 | 56.6%, 1.063 |
  | turn 1, with history | 532 | 54.9%, 1.137 | 57.0%, 1.123 |
  | turn 2+, with history | 3462 | 55.2%, 1.089 | 55.5%, 1.085 |

  About +2 top-1 on turn 1 only, where earlier turn-1 moves are compared. Much smaller than the
  head gain that gave search +6 (joint top-6 47 → 62%). Not worth wiring into search.
- **The real problem on the ladder is that we never change.** Ladder night 1: our leads in
  game N+1 matched game N **every time** (greedy, one team, `--vary-k 1` keeps the preview at the top
  candidate). After a loss where the opponent kept their leads too, we won 10/27 (37%): the bot
  replays a game it just lost. (Excluding forfeits, G1 was 31/51 and G2 21/57, so the G1-vs-G2 gap
  is real, not just forfeits.)
- [x] `--change-after-loss`: in game N+1 after a loss, lead with the policy's likeliest preview
  whose lead pair differs (`preview_candidates`). The Bo3 summary now prints "we repeated
  leads after our loss". Local check: 0/5 repeats with the flag, 8/8 without.
- [x] Cost check on ilab vs nn_v2_all (it never punishes repeats, so this only measures the cost).
  Free:

  | | series | game 1 | game 2 | game 3 | games 2–3 |
  |---|---|---|---|---|---|
  | plain (396978) | 84.0% | 76.6% | 76.6% | 72.4% | 1005/1330 = 75.6% |
  | `--change-after-loss` (396999) | 83.2% | 75.3% | 77.4% | 71.5% | 1002/1319 = 76.0% |

  Repeated leads after our loss 0/396. Game 2 minus game 1: +2.1 vs 0.0 for plain.
- [ ] Ladder with `scripts/bot.sh --ladder --change-after-loss` and compare G2/G3 after losses.
- **Why games 2–3 still lose: humans answer our turn 1 (ladder nights, 2026-09-29).** Games 2–3
  with the same leads on both sides as the game before:
  - We played the same turn 1 in 60 of 76, because search + greedy is deterministic (`--vary`'s 0.02
    margin rarely moves it).
  - When we kept turn 1 and they changed theirs: we won 5/19 (26%). When we changed ours: 11/22 (50%).
  - After a G1 win that wasn't a forfeit, we won G2 only 14/38 (37%); they changed their four in 35 of 43.
  - Small samples (about 75 series), but it fits: a deterministic bot is easy to read between games.
- [x] `--counter-t1 W` (search only): in games 2+ on turn 1 with the same leads on both sides, the
  opponent-reply weights q become (1 − W) q plus W on their best reply to our last turn-1 choice
  (the column that minimises our value in the search matrix). So search assumes they may counter what
  we did last time and picks what holds up against that too. Local smoke test: 12/12 such turn 1s countered.
  Also fixed: `SearchPlayer` never recorded leads, so "we repeated leads after our loss" was blank
  for search players.
- [ ] Ladder with `--change-after-loss --counter-t1 0.5`. nn_v2_all never counters, so ilab can only
  measure cost, not gain.

## Series context input (idea 8, step 3; 2026-09-29)

- [x] `nn/series.py` summarises each earlier game of a Bo3 from its log, per player and nickname
  (nicknames stay the same across a series; species change with megas): brought, led, moves used
  on turn 1 and at all, and the winner.
- [x] `encode` reads it from `battle._series` and adds `ser_tok` [12, 4] (share of games brought /
  led, brought / led last game), `ser_mv` [12, 4, 2] (share of games each move was used on turn 1 /
  at all) and `ser_glob` [5] (game 2+, game 3, series score, won last game). All zeros in game 1.
- [x] `Policy(series=True)` adds them to the move vectors, tokens and global token through
  zero-initialised layers. `train.py --init CKPT --series` starts from a checkpoint and is identical
  to it at step 0. Older checkpoints and datasets still load (their series inputs are zeros).
- [x] The dataset builder passes each game the logs of its series' earlier games. Series missing an
  earlier game (22% on M-C) get no context, so the game number stays right. All scraped formats
  are Bo3, so every game 2–3 gets context (about 60% of games).
- [x] Players keep each finished game's log (`battle._build_replay_log()`) per series and attach
  the context in games 2–3 (preview, policy and search). Local check: 108 of 109 game 2–3
  decisions had it.
- [x] `opp_eval.py` now also splits "game 2+ (context)" from "no context".
- [x] ilab: rebuilt into `data/nn_ser` (397106), fine-tuned `nn_v2_all` → `nn_v4_ser` (397107) and
  `nn_v2_mt` → `nn_v4_ser_mt` (397108). `opp_eval --data data/nn_ser_mc` (M-C held out, 7477 joint):

  | jact@6 (move@1) | all | game 2+ (context), 3678 | no context, 3799 |
  |---|---|---|---|
  | nn_v2_mt | 61.3% (56.3%) | 60.6% (56.0%) | 61.9% (56.6%) |
  | nn_v4_ser_mt | 62.3% (56.3%) | 62.0% (56.3%) | 62.6% (56.4%) |

  Game 2+ gains +1.4 points of jact@6, but the no-context rows also gain +0.7 (from the extra
  fine-tuning), so the context itself is worth about +0.7. That is within noise (±0.8 at n = 3678).
  Per-slot numbers don't move. This matches `opp_habits.py`: what a human did in earlier games
  hardly predicts their next move beyond what the board already says. `nn_v4_ser_mt` is no worse,
  so it can be the opponent model, but this isn't where games 2–3 are lost.
- Known gaps: no speed or bulk bounds yet, and a back Pokémon that never switched in stays unseen.
