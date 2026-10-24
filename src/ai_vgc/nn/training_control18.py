"""Restart-safe entropy/LR damping and time-window skill checks."""
from dataclasses import dataclass, asdict
import math


@dataclass
class DampingSchedule:
    start_steps: int
    duration_steps: int
    initial_lr: float
    final_lr: float
    initial_entropy: float
    final_entropy: float

    def values(self, steps, multiplier=1.0):
        progress = min(1.0, max(0.0, (steps - self.start_steps) / max(1, self.duration_steps)))
        lr = self.initial_lr + progress * (self.final_lr - self.initial_lr)
        entropy = self.initial_entropy + progress * (self.final_entropy - self.initial_entropy)
        return max(self.final_lr, lr * multiplier), entropy

    def state_dict(self):
        return asdict(self)


@dataclass
class SkillCheckState:
    reference_steps: int
    elapsed_seconds: float = 0.0
    lr_multiplier: float = 1.0
    checks: int = 0

    def due(self, interval, extra_seconds=0.0):
        return interval > 0 and self.elapsed_seconds + extra_seconds >= interval

    def finish(self, result, *, expected_games, threshold, steps):
        games = result["games"]
        counts = [result[k] for k in ("wins", "losses", "draws_or_capped")]
        if games <= 0 or games != expected_games or min(counts) < 0 or sum(counts) != games:
            raise ValueError("Incomplete skill check; learning rate was not changed")
        score = result["wins"] / games
        halve = score < threshold
        if halve:
            self.lr_multiplier *= 0.5
        self.reference_steps = steps
        self.elapsed_seconds = 0.0
        self.checks += 1
        return score, halve

    def state_dict(self, extra_seconds=0.0):
        state = asdict(self)
        state["elapsed_seconds"] += max(0.0, extra_seconds)
        return state


def validate_skill_check(seconds, games, batch_size, threshold):
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("skill-check seconds must be finite and nonnegative")
    if games < 2 or games % 2 or batch_size < 2 or batch_size % 2:
        raise ValueError("skill-check games and batch size must be positive even numbers")
    if not 0 < threshold < 1:
        raise ValueError("skill-check threshold must be between zero and one")
