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

ROOT = Path(__file__).resolve().parents[2]


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
