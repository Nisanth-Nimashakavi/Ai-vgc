# Handoff (2026-10-02)

Where things stand, for picking up in a new chat. Details and all results: `docs/training-ideas.md` (the last few sections), team table: `docs/team-ranking.md`, the 10 hand-picked teams: `docs/specialist-teams.md`.

## Working setup

- Ladder (laptop): CTS Bo1 Reg M-C, team MC378 (rain), model `mc-cts-rnad-v7-MC378.pt`, one-turn search:
  `scripts/bot.sh --ladder --cts --use MC378 --model data/models/mc-cts-rnad-v7-MC378.pt --opp-model data/models/mc-cts-opp-v6.pt`
  Ladder results (CTS Bo1): MC301 v7 72–59, peak 1646 (Oct 2 08:43), latest 1614: the best so far. MC378 v7 30–24, peak 1568, latest 1484. v7h 2–7. v6 general model peaked 1526.
  MC301 (Sun + Trick Room: Rillaboom / Sylveon / Charizard / Incineroar / Farigiraf / Garchomp) is now the strongest ladder team; laddered with `--use MC301 --model data/models/mc-cts-rnad-v7-MC301.pt`.
- One rain team only (user's call). Non-rain candidates still to fair-test: MC301, U6, MC41, MC4 (command in docs/team-ranking.md).
- Results notebook: `uv run --extra viz jupyter lab notebooks/results.ipynb` (copy team_rr/team_tests/logs from ilab first; command in the notebook). Ladder games record `model` since Oct 1.
- The user runs every ilab command; give one-line bash commands (ilab is bash, laptop zsh). Laptop → ilab sync: `cd ~/projects/ai-vgc && rsync -a src scripts docs gpu:ai-vgc/`.

## Running / pending on ilab

1. `TEAM=MC378 bash scripts/slurm/value_pipeline.sh` (may not be submitted yet): STAGE=f RL (v7 + speed inference, damage inference, set guesses → `mc-cts-rnad-v7f-MC378.pt`), then 1,600 search games, then value-head training on search values → `mc-cts-rnad-v7fv-MC378.pt`. Job ids in `logs/value_jobs.txt`.
2. Possibly `STAGE=s` (speed inference only → v7s). Superseded by v7f.

When done, copy to laptop and test locally (MC378, CTS, 2000 games, `--only MC378 --opponent-teams data/teams/reg_mc --closed-sheets`):
- v7f with `--speed-inference --damage-inference --set-guess`, no search, vs v7 without them; opponents `nn:data/models/mc-cts-opp-v6.pt` and `nn:data/models/all-bo3-bc-v2.pt` (v7 baseline: 73.0% and 96.0%).
- v7fv + search (same flags) vs v7f without search.
New flags must only be used with models trained with them (v7f+), never with v7.

## What's been learned (don't redo)

- Specialists (stage 1, v7) are a real gain: +2 to +6 over v6 on the fair test.
- No gain: stage 2 (v8), human-imitation opponent (v7h, 2–7 on ladder), --rating setting (≤1 point), bigger or 2-turn search, team-level set guesses for search. Search adds ~nothing in CTS (74.9 plain vs 73.8–77.0 searched).
- Big bug found Oct 2: in CTS the damage features were always zero (opponent stats unknown → calc fails). Every CTS model so far played without them. `--damage-inference` fixes it; v7f is the first model trained with it.
- Other bugs fixed: sim bridge race (search lock), poke-env "[from] move" KeyError, round robin POOL/TEAMS mix-up, team_rr ranking.csv, process limits (team RL uses 16 workers; team tests SHARDS=4).

## Ideas not yet done

- Balanced (equilibrium) move choice per turn instead of best response (PokaiTrainer; small).
- Enumerate damage rolls instead of 2 random seeds in search.
- Fine-tune on our own ladder games (data/bot_logs) once there are a few hundred.
- Bo3 games 2–3 slump (69% G1 → ~45%) if going back to Bo3.
- PokaiTrainer (arXiv 2608.29197) is the closest bot: Reg M-B OTS Bo3, 1350–1400 level; no code/checkpoints released yet.
