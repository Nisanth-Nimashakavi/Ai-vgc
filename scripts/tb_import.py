"""Load past RL runs' .log.jsonl files into TensorBoard (runs/rl/<model>), for runs from before
nn/tb.py existed.

    uv run --extra nn python scripts/tb_import.py data/models/*.log.jsonl
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ai_vgc.nn.tb import scalars, writer  # noqa: E402

for path in map(Path, sys.argv[1:]):
    name = Path(path.name.removesuffix(".log.jsonl"))
    if (Path("runs/rl") / name.stem).exists():
        print(f"skip {name}: runs/rl/{name.stem} exists")
        continue
    tb = writer("rl", name)
    n = 0
    for line in path.read_text().splitlines():
        rec = json.loads(line)
        it = rec["iter"]
        scalars(tb, "loss", {k: v for k, v in rec.items() if k not in ("iter", "steps", "play_s", "total_s", "win")}, it)
        scalars(tb, "win", {k: v[0] for k, v in rec.get("win", {}).items()}, it)
        scalars(tb, "time", {"play_s": rec.get("play_s", 0), "iter_s": rec.get("total_s", 0)}, it)
        tb.add_scalar("steps", rec.get("steps", 0), it)
        n += 1
    tb.close()
    print(f"{path} -> runs/rl/{name.stem} ({n} iterations)")
