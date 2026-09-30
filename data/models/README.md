# Models

Checkpoints are named `<reg>-<format>-<method>-v<N>[-<variant>].pt`:

| Part | Values |
|---|---|
| reg | `mb`, `mc` (Reg M-B / M-C), `all` (trained on every regulation's logs) |
| format | `bo3`, `bo1`, `cts` (Bo1 with closed team sheets) |
| method | `bc` imitation of human logs · `opp` imitation with the opponent-action head (search's opponent model) · `ppo` self-play PPO · `do` double oracle · `rnad` R-NaD self-play · `exit` expert iteration |
| vN | generation; a model starts from the previous generation |
| variant | what sets it apart: a team (`mc196`, `MC147`), `series` (Bo3 context), an ablation |

Files next to a checkpoint share its name: `_snapshots/`, `_meta.json` (double-oracle table),
`_reg.pt`, `.log.jsonl`. `archive/` holds retired models, kept for comparison and history only.
`nn_vocab.json` is the vocabulary every model's embeddings index into.

## In use

| Model | What | Old name |
|---|---|---|
| `mc-bo3-rnad-v5` | **Current best.** Bo3 R-NaD from `all-bo3-bc-v4-series`; 72% vs its start | `nn_v5_bo3` |
| `mc-bo1-rnad-v3` | Reg M-C R-NaD from `all-bo3-bc-v2`; `bot.sh --general` default | `nn_v3_mc` |
| `mc-bo1-rnad-v3-mc196` | `mc-bo1-rnad-v3` specialised on team MC196; `bot.sh` default | `nn_mc196_bc` |
| `all-bo3-opp-v2` | Human-trained opponent model for `--search --opp-model` | `nn_v2_mt` |
| `all-bo3-bc-v2` | Imitation on all logs; the "human proxy" opponent | `nn_v2_all` |
| `all-bo3-bc-v4-series` | `all-bo3-bc-v2` fine-tuned with Bo3 series context | `nn_v4_ser` |

Planned: `mc-bo3-rnad-v6-<team>` (per-team specialists), `mc-cts-rnad-v6`, `mc-cts-opp-v6`.

## Archive

| Model | What | Old name |
|---|---|---|
| `mb-bo3-bc-v0` | First imitation model | `nn_default` |
| `mb-bo3-bc-v1` | Imitation, loss-weighted | `nn_lw1` |
| `mb-bo3-ppo-v1` | First self-play PPO | `nn_rl` |
| `mb-bo3-do-v1` | Double oracle from `mb-bo3-ppo-v1` | `nn_do` |
| `mb-bo3-rnad-v3` | Reg M-B R-NaD from `all-bo3-bc-v2` | `nn_v3` |
| `all-bo3-bc-v2-{base,aug,awr,aux}` | v2 imitation ablations | `nn_v2_*` |
| `mb-bo3-exit-v1*` | Expert-iteration attempts, no gain | `nn_x1*` |

`docs/training-ideas.md` uses the old names. `scripts/rename_models.py` holds the full mapping and
applies it on any machine that holds checkpoints under the old names.
