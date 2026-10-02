"""TensorBoard logging for the training scripts: one run folder per model, under runs/.

    uv run --extra nn tensorboard --logdir runs --port 6006

A missing tensorboard package only turns logging off, so training never fails because of it.
"""

from __future__ import annotations

from pathlib import Path


class _Off:
    def add_scalar(self, *args, **kwargs) -> None:
        pass

    def add_text(self, *args, **kwargs) -> None:
        pass

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass


def writer(kind: str, out: Path, args=None):
    """A SummaryWriter at runs/<kind>/<model name>; `args` goes in as text so runs can be told apart."""
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        print("tensorboard not installed: no runs/ logging", flush=True)
        return _Off()
    w = SummaryWriter(str(Path("runs") / kind / Path(out).stem), flush_secs=30)
    if args is not None:
        w.add_text("args", "  \n".join(f"{k}: {v}" for k, v in sorted(vars(args).items())))
    return w


def scalars(w, prefix: str, values: dict, step: int) -> None:
    for k, v in values.items():
        if isinstance(v, (int, float)):
            w.add_scalar(f"{prefix}/{k}", v, step)
