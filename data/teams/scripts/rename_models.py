"""Move checkpoints from the old ad-hoc names (nn_v3_mc.pt, ...) to the naming scheme in
data/models/README.md, retired ones into data/models/archive/. Safe to rerun, and run it on every
machine that holds models (laptop and ilab); models it doesn't find are skipped.

    uv run python scripts/rename_models.py --dry-run
    uv run python scripts/rename_models.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

MODELS = Path("data/models")

# old stem -> new path under data/models (without .pt)
RENAMES = {
    # in use
    "nn_v5_bo3": "mc-bo3-rnad-v5",
    "nn_v3_mc": "mc-bo1-rnad-v3",
    "nn_mc196_bc": "mc-bo1-rnad-v3-mc196",
    "nn_mc196": "mc-bo1-rnad-v3-mc196-selfmix",
    "nn_v4_ser": "all-bo3-bc-v4-series",
    "nn_v4_ser_mt": "all-bo3-opp-v4-series",
    "nn_v2_all": "all-bo3-bc-v2",
    "nn_v2_mt": "all-bo3-opp-v2",
    # retired
    "nn_default": "archive/mb-bo3-bc-v0",
    "nn_lw1": "archive/mb-bo3-bc-v1",
    "nn_rl": "archive/mb-bo3-ppo-v1",
    "nn_do": "archive/mb-bo3-do-v1",
    "nn_v3": "archive/mb-bo3-rnad-v3",
    "nn_v3_300": "archive/mb-bo3-rnad-v3-iter300",
    "nn_v2_base": "archive/all-bo3-bc-v2-base",
    "nn_v2_aug": "archive/all-bo3-bc-v2-aug",
    "nn_v2_awr": "archive/all-bo3-bc-v2-awr",
    "nn_v2_aux": "archive/all-bo3-bc-v2-aux",
    "nn_x1": "archive/mb-bo3-exit-v1",
    "nn_x1h": "archive/mb-bo3-exit-v1h",
    "nn_x1o": "archive/mb-bo3-exit-v1o",
    "nn_x1p": "archive/mb-bo3-exit-v1p",
    # planned (names the scripts now write)
    "nn_cts_rl": "mc-cts-rnad-v6",
    "nn_cts_mt": "mc-cts-opp-v6",
}
# Files rl.py and train.py keep next to a checkpoint, named after its stem.
SIDECARS = [".pt", "_snapshots", "_meta.json", "_reg.pt", "_current.pt", ".log.jsonl"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    moved = 0
    for old, new in RENAMES.items():
        for suffix in SIDECARS:
            src, dst = MODELS / f"{old}{suffix}", MODELS / f"{new}{suffix}"
            if not src.exists():
                continue
            if dst.exists():
                print(f"skip {src}: {dst} exists")
                continue
            print(f"{src} -> {dst}")
            moved += 1
            if not args.dry_run:
                dst.parent.mkdir(parents=True, exist_ok=True)
                src.rename(dst)
    print(f"{moved} moved" + (" (dry run)" if args.dry_run else ""))


if __name__ == "__main__":
    main()
