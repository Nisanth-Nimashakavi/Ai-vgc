"""Shared helpers for local matches: start a Showdown server, make throwaway accounts, and put a
confidence interval on a win rate."""

from __future__ import annotations

import math
import socket
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

from poke_env import AccountConfiguration
from poke_env.battle.abstract_battle import AbstractBattle

ROOT = Path(__file__).resolve().parents[2]


def _patch_from_move() -> None:
    """poke-env looks up a "[from] move: X" message's move X in the user's known moves and raises
    KeyError when X isn't there (Round's follow-up, or an unrevealed move with closed sheets), which
    kills the whole player. Record X as known before the message is parsed."""
    parse = AbstractBattle.parse_message
    if getattr(parse, "_from_move_patch", False):
        return

    def parse_message(self, split_message):
        if len(split_message) > 3 and split_message[1] == "move" and split_message[-1].startswith("[from] move: "):
            try:
                self.get_pokemon(split_message[2])._add_move(split_message[-1].split(": ")[-1])
            except Exception:
                pass
        return parse(self, split_message)

    parse_message._from_move_patch = True
    AbstractBattle.parse_message = parse_message


_patch_from_move()


def wilson(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = wins / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half


def _server_ready(port: int) -> bool:
    # The port opens before Showdown's workers can finish a websocket handshake,
    # so wait for a real HTTP response rather than just a TCP connect.
    try:
        urllib.request.urlopen(f"http://localhost:{port}/", timeout=2)
    except urllib.error.HTTPError:
        pass
    except OSError:
        return False
    return True


def ensure_server(port: int, root: Path | None = None) -> subprocess.Popen | None:
    with socket.socket() as s:
        if s.connect_ex(("localhost", port)) == 0:
            return None
    # Many servers booting at once on a busy node can take a while, or one can die at startup;
    # wait up to 3 minutes and restart a server whose process exited.
    cwd = Path(root or ROOT / "pokemon-showdown")
    # `start` rebuilds dist/ every time, which breaks other jobs' servers and sim bridges that
    # are loading it at that moment; only build when there is nothing built yet.
    skip = ["--skip-build"] if (cwd / "dist" / "sim").is_dir() else []
    for _attempt in range(3):
        proc = subprocess.Popen(
            ["node", "pokemon-showdown", "start", *skip, "--no-security", str(port)],
            cwd=cwd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        for _ in range(360):
            time.sleep(0.5)
            if _server_ready(port):
                time.sleep(1)
                return proc
            if proc.poll() is not None:
                break
        proc.kill()
        time.sleep(2)
    raise RuntimeError(f"Showdown server did not start on port {port}")


def account(prefix: str) -> AccountConfiguration:
    # Showdown usernames max out at 18 characters.
    return AccountConfiguration(f"{prefix}{uuid.uuid4().hex[:6]}", None)
