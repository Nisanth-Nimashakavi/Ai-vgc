"""Play live battles with the policy network.

    uv run python -m ai_vgc.nn.player --model data/models/archive/mb-bo3-bc-v1.pt --n 100
    uv run python -m ai_vgc.nn.player --model data/models/archive/mb-bo3-ppo-v1.pt --opponent nn:data/models/archive/mb-bo3-bc-v1.pt
    uv run python -m ai_vgc.nn.player --model data/models/archive/mb-bo3-do-v1.pt --accept nnbot --n 10  # play it yourself
    uv run --extra nn python -m ai_vgc.nn.player --model data/models/archive/mb-bo3-rnad-v3.pt --search \
        --opp-model data/models/all-bo3-opp-v2.pt --search-prior 0.1 --greedy \
        --accept MyBotName --server showdown --from MyMainName --n 5  # on play.pokemonshowdown.com

`--server showdown` logs in to the public server, not a local one. The password comes from
$PS_PASSWORD, or you type it at a prompt (blank for an unregistered name). `--from` accepts
challenges only from those accounts; without it, anyone on the server can start a game.

Each decision is the one the network was trained on: encode the battle, pick
slot a's action, then slot b's given slot a. Live play masks with poke-env's
request-based mask (`DoublesEnv.get_action_mask`), which knows about
disabled moves, Choice locks, trapping etc. that the log-based mask can't see.
With `rules=True` (the default) it also drops moves that are sure to fail, such
as Fake Out into Armor Tail or Close Combat into a Ghost (`rules.py`).

Team preview is two decisions, as in the dataset: the two leads, then (with
the leads marked as selected) the two in the back.

Decisions are batched: every NNPlayer on the same model in one process queues its
decision, and the model runs once for everything waiting (`_Batcher`). An RL worker
or eval with 8 battles at once then pays for about one forward pass, not eight.

In a Bo3 format (e.g. `--format gen9championsvgc2026regmbbo3`, --n counts series) the
player also records, per game, its series and game number, the result, and both sides'
leads and Pokemon brought, and main() prints a series summary (`series_summary`). With
`--adapt`, games 2 and 3 previews answer the opponent's previous leads (`bo3.py`).

With `record=True` the player keeps every decision it makes (inputs, action,
log-probability, value) per battle, for reinforcement learning (`rl.py`).
"""

from __future__ import annotations

import argparse
import asyncio
import getpass
import json
import os
import random
import re
import time
import weakref
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from poke_env import (
    AccountConfiguration,
    ServerConfiguration,
    ShowdownServerConfiguration,
)
from poke_env.battle import AbstractBattle, DoubleBattle
from poke_env.data import to_id_str
from poke_env.environment import DoublesEnv
from poke_env.player import BattleOrder, Player, SimpleHeuristicsPlayer

from ai_vgc.nn.encode import N_MV, encode
from ai_vgc.nn.model import Policy, masked, slot_b_mask
from ai_vgc.nn.rules import apply_rules
from ai_vgc.nn.series import Context, summarize

_get_pokemon = AbstractBattle.get_pokemon


def _get_pokemon_forme(self, identifier: str, *args, **kwargs):
    """poke-env keys un-nicknamed regional formes by species ("p2: Zoroark-Hisui") from team
    preview, but Showdown names them by base species ("p2a: Zoroark"). A message that only
    carries the name (a move's target) then can't match it and raises "team already has N
    pokemons", which kills the battle. Fall back to the one team member of that base species."""
    try:
        return _get_pokemon(self, identifier, *args, **kwargs)
    except ValueError:
        role, name = identifier[:2], to_id_str(identifier[3:].split(":", 1)[-1])
        team = self._team if role == self.player_role else self._opponent_team
        hits = [m for m in team.values() if to_id_str(m.species).startswith(name)]
        if len(hits) != 1:
            raise
        return hits[0]


AbstractBattle.get_pokemon = _get_pokemon_forme


def load_policy(path: str | Path, device: str = "cpu") -> Policy:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    sizes = ckpt["sizes"]
    # The move table is a buffer in the state dict, so zeros are enough here.
    model = Policy(sizes, np.zeros((sizes["moves"], N_MV), np.float32), **ckpt["config"])
    model.load_state_dict(ckpt["state"])
    return model.to(device).eval()


@torch.no_grad()
def sample_batch(model: Policy, obs: list[dict[str, np.ndarray]],
                 greedy: list[bool]) -> list[tuple[np.ndarray, float, float]]:
    """(action [2], joint log-prob, value logit) for each state; obs carry their "mask"."""
    b = {k: torch.from_numpy(np.stack([o[k] for o in obs])) for k in obs[0]}
    m = b["mask"]
    g = torch.tensor(greedy)
    enc = model.encode(b)

    def pick(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logp = F.log_softmax(logits, -1)
        a = torch.where(g, logp.argmax(-1), torch.multinomial(logp.exp(), 1)[:, 0])
        return a, logp.gather(1, a[:, None])[:, 0]

    a0, lp0 = pick(masked(model.slot_logits(enc, b, 0), m[:, 0]))
    a1, lp1 = pick(masked(model.slot_logits(enc, b, 1, a0), slot_b_mask(m[:, 1], a0)))
    v = model.value(enc[0])[:, 0]
    acts = torch.stack([a0, a1], 1).numpy()
    return [(acts[i], (lp0[i] + lp1[i]).item(), v[i].item()) for i in range(len(obs))]


def sample(model: Policy, obs: dict[str, np.ndarray], mask: np.ndarray,
           greedy: bool = False) -> tuple[np.ndarray, float, float]:
    """`sample_batch` for one state."""
    return sample_batch(model, [obs | {"mask": mask}], [greedy])[0]


class _Batcher:
    """Collects the decisions waiting on one model and runs them as one batch, once the
    event loop has handled the messages that are already in."""

    def __init__(self, model: Policy):
        self.model = model
        self.pending: list[tuple[dict, bool, asyncio.Future]] = []
        self.loop: asyncio.AbstractEventLoop | None = None
        self.calls = self.decisions = 0

    def decide(self, obs: dict[str, np.ndarray], greedy: bool) -> asyncio.Future:
        loop = asyncio.get_running_loop()
        if loop is not self.loop:  # a new asyncio.run: anything left from the old loop is dead
            self.loop, self.pending = loop, []
        fut = loop.create_future()
        if not self.pending:
            loop.call_soon(self._flush)
        self.pending.append((obs, greedy, fut))
        return fut

    def _flush(self) -> None:
        batch, self.pending = self.pending, []
        self.calls += 1
        self.decisions += len(batch)
        try:
            out = sample_batch(self.model, [o for o, _, _ in batch], [g for _, g, _ in batch])
        except Exception as e:
            for _, _, fut in batch:
                if not fut.done():
                    fut.set_exception(e)
            return
        for (_, _, fut), r in zip(batch, out):
            if not fut.done():
                fut.set_result(r)


_BATCHERS: weakref.WeakKeyDictionary[Policy, _Batcher] = weakref.WeakKeyDictionary()


def batcher(model: Policy) -> _Batcher:
    if model not in _BATCHERS:
        _BATCHERS[model] = _Batcher(model)
    return _BATCHERS[model]


class NNPlayer(Player):
    def __init__(self, model: Policy | str | Path, rating: float = 1700, greedy: bool = False,
                 record: bool = False, rules: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.model = load_policy(model) if isinstance(model, (str, Path)) else model
        self.rating = rating
        self.greedy = greedy
        self.record = record
        self.rules = rules
        self.steps: dict[str, list[dict]] = {}
        self.episodes: list[tuple[list[dict], float]] = []  # (steps, reward) per finished battle
        self.series_of: dict[str, tuple[str, int]] = {}  # battle tag -> (Bo3 room, game number)
        self.leads: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {}  # battle tag -> (ours, theirs)
        self.games: list[dict] = []  # one entry per finished Bo3 game
        self.adapter = None  # bo3.Adapter: games 2+ previews as a response to the last game
        # Games 2+ of a series: sample the preview from the policy's top `vary_k` and (with search)
        # pick among candidates within `vary_margin` of the best, so the opponent can't just
        # replay game 1's answer to it. Game 1 stays greedy.
        self.vary = False
        self.vary_k = 4
        self.vary_margin = 0.02
        # Games 2+ after a loss: lead with the policy's likeliest other pair. The bot is deterministic,
        # so keeping the leads replays the game it just lost (ladder night 1: 10/27 when both kept).
        self.change_after_loss = False
        # Bo3 test opponent: replay the last game's preview with these probabilities after
        # losing / winning it (humans: 0.24 / 0.49 for the leads, scripts/human_bo3.py).
        self.repeat: tuple[float, float] | None = None
        self.last_preview: dict[str, str] = {}  # Bo3 room -> our last "/team" order
        self.series_logs: dict[str, list[str]] = {}  # Bo3 room -> logs of its finished games

    async def _handle_battle_message(self, split_messages: list[list[str]]):
        # Each game of a Bo3 announces "Game N of <a href="/game-bestof3-...">"; poke-env ignores it.
        for m in split_messages[1:]:
            if len(m) > 3 and m[1] == "uhtml" and m[2] == "bestof":
                g = BESTOF.search("|".join(m[3:]))
                if g:
                    self.series_of[split_messages[0][0][1:]] = (g.group(2), int(g.group(1)))
        await super()._handle_battle_message(split_messages)

    async def _decide(self, battle: DoubleBattle, mask: np.ndarray, preview: bool) -> np.ndarray:
        obs = encode(battle, self.rating)
        obs["mask"] = mask
        action, logp, value = await batcher(self.model).decide(obs, self.greedy)
        if self.record:
            self.steps.setdefault(battle.battle_tag, []).append(
                obs | {"action": action.astype(np.int16), "logp": logp, "value": value, "preview": preview})
        return action

    def attach_series(self, battle: DoubleBattle) -> None:
        """Give `encode` the earlier games of this battle's Bo3 series (none in game 1)."""
        series = self.series_of.get(battle.battle_tag)
        logs = self.series_logs.get(series[0], []) if series else []
        battle._series = Context.of(logs, battle.player_username, battle.opponent_username or "")

    async def teampreview(self, battle: AbstractBattle) -> str:
        assert isinstance(battle, DoubleBattle)
        self.attach_series(battle)
        order = await self._choose_preview(battle)
        if not order.startswith("/team "):  # e.g. PreviewPlayer's probe forfeits
            return order
        # The action mask and search read these flags all game to know which four were brought,
        # so every way of picking the preview has to leave them set (preview_candidates clears them).
        chosen = {int(i) for i in order.removeprefix("/team ")}
        for i, mon in enumerate(battle.team.values(), 1):
            mon._selected_in_teampreview = i in chosen
        return order

    async def _choose_preview(self, battle: DoubleBattle) -> str:
        series = self.series_of.get(battle.battle_tag)
        prev = [g for g in self.games if series and g["series"] == series[0]]
        if self.adapter and prev:
            try:
                order = self.adapter.choose(self, battle, prev[-1])
            except Exception as e:  # anything the bridge can't handle: the policy's preview
                self.logger.warning("Bo3 preview failed: %r", e)
                order = None
            if order:
                self.last_preview[series[0]] = order
                return order
        if self.repeat and prev and series[0] in self.last_preview \
                and random.random() < self.repeat[bool(prev[-1]["won"])]:
            return self.last_preview[series[0]]
        if self.change_after_loss and prev and not prev[-1]["won"] and series[0] in self.last_preview:
            from ai_vgc.nn.preview import preview_candidates

            last = set(self.last_preview[series[0]].removeprefix("/team ")[:2])
            order = "/team " + next(c for c, _ in preview_candidates(self.model, battle, self.rating)
                                    if set(c[:2]) != last)
        else:
            order = await self._policy_preview(battle)
        if series:
            self.last_preview[series[0]] = order
        return order

    def later_game(self, battle: AbstractBattle) -> bool:
        return self.series_of.get(battle.battle_tag, ("", 1))[1] > 1

    async def _policy_preview(self, battle: DoubleBattle) -> str:
        if self.vary and self.later_game(battle):
            from ai_vgc.nn.preview import preview_candidates

            cands = preview_candidates(self.model, battle, self.rating)[:self.vary_k]
            p = np.exp(np.array([lp for _, lp in cands]) - cands[0][1])
            return "/team " + cands[np.random.choice(len(cands), p=p / p.sum())][0]
        team = list(battle.team.values())
        for mon in team:
            mon._selected_in_teampreview = False
        leads = await self._decide(battle, encode(battle)["mask"], True)
        for i in leads:
            team[i - 1]._selected_in_teampreview = True
        back = await self._decide(battle, encode(battle)["mask"], True)
        for i in back:
            team[i - 1]._selected_in_teampreview = True
        return "/team " + "".join(str(i) for i in [*leads, *back])

    async def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        assert isinstance(battle, DoubleBattle)
        self.attach_series(battle)
        if battle.battle_tag not in self.leads:
            self.leads[battle.battle_tag] = (
                tuple(m.species for m in battle.active_pokemon if m),
                tuple(m.species for m in battle.opponent_active_pokemon if m))
        mask = action_mask(battle)
        if self.rules:
            mask = apply_rules(battle, mask)
        action = await self._decide(battle, mask, False)
        return DoublesEnv.action_to_order(action, battle, strict=False)

    def _battle_finished_callback(self, battle: AbstractBattle) -> None:
        tag = battle.battle_tag
        if tag in self.series_of:
            ours, theirs = self.leads.pop(tag, ((), ()))
            series, game = self.series_of.pop(tag)
            logs = self.series_logs.setdefault(series, [])
            logs.append(battle._build_replay_log())
            wins = Counter(summarize(x).winner for x in logs)
            if max(wins.values()) >= 2 or len(logs) >= 3:  # series over
                del self.series_logs[series]
            self.games.append({
                "series": series, "game": game, "won": battle.won,
                "our_leads": ours, "their_leads": theirs,
                "their_brought": tuple(m.species for m in battle.opponent_team.values() if m.revealed),
            })
        steps = self.steps.pop(battle.battle_tag, [])
        if self.record and steps:
            reward = 1.0 if battle.won else 0.0 if battle.lost else 0.5
            self.episodes.append((steps, reward))


BESTOF = re.compile(r'Game (\d)</strong> of <a href="/(game-bestof\d-[^"]+)"')


def series_summary(games: list[dict]) -> str:
    """Series won, win rate by game number, and how often the opponent repeated its previous
    game's leads / four (and whether that depended on who won the previous game)."""
    by_series: dict[str, list[dict]] = defaultdict(list)
    for g in games:
        by_series[g["series"]].append(g)
    won = sum(sum(g["won"] for g in gs) >= 2 for gs in by_series.values())
    n = len(by_series)
    by_game = defaultdict(list)
    same_lead = defaultdict(list)  # "after their win" / "after their loss" -> repeated?
    same_four = []
    we_kept = []  # after our loss: did we lead the same pair again?
    for gs in by_series.values():
        gs.sort(key=lambda g: g["game"])
        for prev, g in zip(gs, gs[1:]):
            same_lead["after their win" if not prev["won"] else "after their loss"].append(
                set(g["their_leads"]) == set(prev["their_leads"]))
            same_four.append(set(g["their_brought"]) == set(prev["their_brought"]))
            if not prev["won"]:
                we_kept.append(set(g["our_leads"]) == set(prev["our_leads"]))
        for g in gs:
            by_game[g["game"]].append(g["won"])
    pct = lambda xs: f"{sum(xs)}/{len(xs)} = {sum(xs) / max(len(xs), 1):.0%}"  # noqa: E731
    games_part = ", ".join(f"game {k} {pct(v)}" for k, v in sorted(by_game.items()))
    lead_part = ", ".join(f"{k} {pct(v)}" for k, v in sorted(same_lead.items()))
    return (f"series {won}/{n} = {won / max(n, 1):.1%}; {games_part}; opponent repeated leads: {lead_part}; "
            f"same four {pct(same_four)}; we repeated leads after our loss {pct(we_kept)}")


def action_mask(battle: DoubleBattle) -> np.ndarray:
    """poke-env's request-based mask, [2, N_ACT].

    When both actives faint and one Pokemon is left, poke-env lets each slot switch it in or
    pass, so the policy can pass twice, which Showdown rejects (poke-env then plays a random
    move). Slot a has to bring it in then; slot b's only option left is to pass.
    """
    mask = np.array(DoublesEnv.get_action_mask(battle), bool).reshape(2, -1)
    if all(battle.force_switch) and mask[0, 1:].any():
        mask[0, 0] = False
    return mask


class ChallengeMixin:
    """Accepts challenges in several formats, e.g. Bo1 and Bo3 of one regulation. Set
    `formats` (and optionally `allowed`, the user ids it accepts from) after construction.

    poke-env keys its bookkeeping (Bo3 series rooms, open team sheets, game counting) on the
    player's single format, so it switches to each challenge's format before accepting it
    and plays one game or series at a time.
    """

    formats: list[str]
    allowed: set[str] | None = None
    save_dir: Path | None = None  # games.jsonl plus an HTML replay per game
    logs_dir: Path | None = None  # logs_<format>.json, the scraped logs' layout, for the bot's games only
    watch_url: str | None = None  # "https://play.pokemonshowdown.com/": print each battle's link there
    watch_open = False  # also show it in the browser, in one tab that follows each new battle
    _watching: list[str] | None = None  # [current battle url], read by the watch page's server

    def _watch(self, url: str) -> None:
        """Point the watch tab at `url`. The first call starts a small local server and opens its
        page once. Showdown won't run in an iframe, so after one click the page opens a battle tab
        and then moves that same tab to each new battle (an opener may navigate its popup)."""
        if self._watching is not None:
            self._watching[0] = url
            return
        import http.server
        import threading
        import webbrowser

        self._watching = watching = [url]
        page = b"""<!doctype html><title>nimnimbot</title>
<style>body{font:16px sans-serif;background:#222;color:#ddd;padding:2em}
button{font-size:1.2em;padding:.5em 1em}</style>
<button id=b>Watch battles</button><p id=s>Click once: the battles open in one tab that follows each new game.</p>
<script>
let win = null, cur = "";
const b = document.getElementById("b"), s = document.getElementById("s");
async function latest() { return (await fetch("/current", {cache: "no-store"})).text(); }
b.onclick = async () => {
  cur = await latest();
  win = window.open(cur, "nimnimbot-battle");
  s.textContent = win ? "Watching: " + cur : "Popup blocked: allow popups for this page and click again.";
};
setInterval(async () => {
  if (!win) return;
  if (win.closed) { win = null; s.textContent = "Battle tab closed: click to reopen."; return; }
  try {
    const u = await latest();
    if (u && u !== cur) { cur = u; win.location.href = u; s.textContent = "Watching: " + u; }
  } catch (e) {}
}, 2000);
</script>"""

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                body = watching[0].encode() if self.path == "/current" else page
                self.send_response(200)
                self.send_header("Content-Type", "text/plain" if self.path == "/current" else "text/html")
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        page_url = f"http://127.0.0.1:{server.server_address[1]}/"
        print(f"watch tab: {page_url}", flush=True)
        webbrowser.open(page_url)

    async def _handle_battle_message(self, split_messages: list[list[str]]):
        await super()._handle_battle_message(split_messages)
        # poke-env reads the open team sheets but leaves them out of the battle's log; the scraped
        # human logs have them (right after |teampreview|), and replay.py needs them for full sets.
        sheets = [m for m in split_messages[1:] if len(m) > 2 and m[1] == "showteam"]
        battle = self._battles.get(split_messages[0][0].removeprefix(">")) if sheets else None
        if battle is not None:
            battle._replay_data.extend(m[:] for m in sheets)

    async def _create_battle(self, split_message: list[str]) -> AbstractBattle:
        battle = await super()._create_battle(split_message)
        if self.watch_url:
            url = self.watch_url + battle.battle_tag
            print(f"watch: {url}", flush=True)
            if self.watch_open:
                self._watch(url)
        return battle

    def _battle_finished_callback(self, battle: AbstractBattle) -> None:
        super()._battle_finished_callback(battle)
        if self.save_dir is None:
            return
        try:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            replay = battle.save_replay(self.save_dir / f"{battle.battle_tag}.html")
            result = "win" if battle.won else "loss" if battle.lost else "tie"
            row = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "tag": battle.battle_tag, "user": self.username,
                   "format": self._format, "opponent": battle.opponent_username, "result": result,
                   "turns": battle.turn, "replay": replay.name,
                   "ours": [m.species for m in battle.team.values()],
                   "theirs": [m.species for m in battle.opponent_team.values()],
                   # Team file the pool last handed out (ladder plays one match at a time).
                   "team": getattr(self._team, "last_name", None),
                   # The bot's view of the battle, as Showdown sent it (our side's exact HP included).
                   "log": battle._build_replay_log()}
            with open(self.save_dir / "games.jsonl", "a") as f:
                f.write(json.dumps(row) + "\n")
            if self.logs_dir:
                self._add_log(battle.battle_tag.removeprefix("battle-"), row["log"])
            print(f"{result} vs {battle.opponent_username} in {battle.turn} turns ({replay})", flush=True)
        except Exception as e:  # never lose the session over a log line
            print(f"could not save {battle.battle_tag}: {e!r}", flush=True)

    def _add_log(self, battle_id: str, log: str) -> None:
        """Add one game to logs_dir/logs_<format>.json ({id: [unix time, log]}, like
        data/battle_logs), kept apart so bot games never mix into the human training data."""
        import fcntl

        self.logs_dir.mkdir(parents=True, exist_ok=True)
        path = self.logs_dir / f"logs_{self._format}.json"
        # A ladder bot and a challenge bot can finish games at once: lock the read-modify-write.
        with open(self.logs_dir / ".lock", "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            logs = json.loads(path.read_text()) if path.exists() else {}
            # The scraped logs start at the first "|j|"; drop the room header poke-env keeps.
            lines = log.split("\n")
            start = next((i for i, x in enumerate(lines) if x.startswith("|j|")), 0)
            logs[battle_id] = [int(time.time()), "\n".join(lines[start:])]
            tmp = path.with_suffix(f".{os.getpid()}.tmp")
            tmp.write_text(json.dumps(logs))
            tmp.replace(path)

    def _wanted(self, user: str, fmt: str) -> bool:
        user = user.strip()
        return (user != self.username and fmt in self.formats
                and (self.allowed is None or to_id_str(user) in self.allowed))

    async def _handle_challenge_request(self, split_message: list[str]):
        if len(split_message) >= 6 and self._wanted(split_message[2], split_message[5]):
            await self._challenge_queue.put((split_message[2].strip(), split_message[5]))

    async def _update_challenges(self, split_message: list[str]):
        for user, fmt in json.loads(split_message[2]).get("challengesFrom", {}).items():
            if self._wanted(user, fmt):
                await self._challenge_queue.put((user, fmt))

    async def _accept_challenges(self, opponent, n_challenges: int, packed_team: str | None):
        await self.ps_client.logged_in.wait()
        for _ in range(n_challenges):
            user, fmt = await self._challenge_queue.get()
            await self._battle_count_queue.join()  # the previous game or series is over
            self._format = fmt
            print(f"accepting {fmt} from {user}", flush=True)
            await self.ps_client.accept_challenge(to_id_str(user), packed_team or self.next_team)
            await self._battle_semaphore.acquire()
        await self._battle_count_queue.join()


class ChallengeNNPlayer(ChallengeMixin, NNPlayer):
    """NNPlayer that accepts challenges (`ChallengeMixin`)."""

    def __init__(self, *args, formats: list[str], **kwargs):
        super().__init__(*args, battle_format=formats[0], **kwargs)
        self.formats = formats


def _public_login(player: Player, password: str | None) -> None:
    """Log in to the public server, replacing poke-env's login: without a password poke-env
    sends an empty assertion, which only a local --no-security server accepts, and with a wrong
    one it dies on a KeyError that hides the login server's reason."""
    import requests

    client = player.ps_client
    name = client.account_configuration.username

    def call(method: str, **kw):
        # The login server is sometimes slow: wait longer than poke-env's 10s, and retry.
        for attempt in range(1, 4):
            try:
                return requests.request(method, client.server_configuration.authentication_url, timeout=30.0, **kw)
            except requests.RequestException as e:
                print(f"login server: {type(e).__name__} (try {attempt}/3)", flush=True)
                time.sleep(5 * attempt)
        print("login server unreachable: try again later", flush=True)
        os._exit(1)

    async def log_in(split_message: list[str]) -> None:
        challstr = f"{split_message[2]}|{split_message[3]}"
        if password:
            r = call("POST", data={"act": "login", "name": name, "pass": password, "challstr": challstr})
            try:
                reply = json.loads(r.text[1:])  # the reply starts with "]"
            except ValueError:
                reply = {}
            assertion = reply.get("assertion", "")
            if not assertion or assertion.startswith(";"):
                err = (reply.get("error") or assertion.lstrip(";") or r.text[:200]).strip()
                print(f"login failed for '{name}': {err or 'wrong password?'}\n"
                      f"  (unregistered name: run again and leave the password blank)", flush=True)
                os._exit(1)
        else:
            r = call("GET", params={"act": "getassertion", "userid": to_id_str(name), "challstr": challstr})
            assertion = r.text.strip()
            if assertion.startswith(";"):
                print(f"'{name}' is a registered name: run again and type its password", flush=True)
                os._exit(1)
            if not assertion or assertion.startswith("{"):
                print(f"login server refused '{name}': {assertion[:200]}", flush=True)
                os._exit(1)
        await client.send_message(f"/trn {name},0,{assertion}")
        await client.change_avatar(client._avatar)

    client.log_in = log_in


def main() -> None:
    from ai_vgc.showdown import account, ensure_server, wilson
    from ai_vgc.teams import TEAMS_DIR, RandomPoolTeambuilder

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="data/models/archive/mb-bo3-bc-v1.pt")
    ap.add_argument("--opponent", default="heuristic", help="heuristic, or nn:<checkpoint>")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--format", default="gen9championsvgc2026regmb")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--rating", type=float, default=1700)
    ap.add_argument("--greedy", action="store_true", help="always play the most likely action")
    ap.add_argument("--no-rules", action="store_true", help="don't mask moves that are sure to fail (rules.py)")
    ap.add_argument("--opponent-no-rules", action="store_true", help="--no-rules for an nn: opponent")
    ap.add_argument("--teams", type=Path, default=None, help="folder of team .txt files (default: Reg M-B pool)")
    ap.add_argument("--showdown", type=Path, default=None,
                    help="Showdown checkout to start if nothing is on --port (e.g. pokemon-showdown-mc for Reg M-C)")
    ap.add_argument("--accept", metavar="NAME",
                    help="log in as NAME and accept --n challenges from anyone, Bo1 or Bo3 of --format "
                         "(play it yourself at http://localhost:<port>, or on --server showdown)")
    ap.add_argument("--ladder", action="store_true",
                    help="with --accept NAME: play --n ladder games of --format as NAME instead of waiting for "
                         "challenges (one at a time; games are saved as with --accept)")
    ap.add_argument("--server", choices=["local", "showdown"], default="local",
                    help="showdown: play.pokemonshowdown.com (--accept only; password from $PS_PASSWORD or a prompt)")
    ap.add_argument("--save-dir", type=Path, default=Path("data/live_games"),
                    help="--accept: where each game goes (games.jsonl and one HTML replay per game)")
    ap.add_argument("--logs-dir", type=Path, default=Path("data/bot_logs"),
                    help="--accept: also add each game to <dir>/logs_<format>.json, the scraped logs' layout "
                         "(kept out of data/battle_logs, so training only uses them when asked)")
    ap.add_argument("--watch", action="store_true", help="--accept: show each battle in one browser tab that follows the next game")
    ap.add_argument("--from", dest="from_", nargs="+", metavar="NAME",
                    help="--accept: only accept challenges from these accounts")
    ap.add_argument("--challenge", nargs="+", metavar="NAME",
                    help="challenge these accounts (e.g. scripts/vgcbench_bot.py) --n times in total, "
                         "one battle per account at a time")
    ap.add_argument("--search", action="store_true", help="pick move-turn actions by one-turn search (search.py)")
    ap.add_argument("--opp-model", default=None,
                    help="checkpoint with the opponent-action head (--aux) for --search; without one, "
                         "the opponent's options are a random sample, equally likely")
    ap.add_argument("--search-k", type=int, default=6, help="our candidate joint actions")
    ap.add_argument("--search-opp-k", type=int, default=6, help="the opponent's candidate joint actions")
    ap.add_argument("--search-seeds", type=int, default=2, help="simulations per pairing")
    ap.add_argument("--search-prior", type=float, default=0.0, help="weight on the policy's log-probability")
    ap.add_argument("--adapt", action="store_true",
                    help="Bo3: pick games 2-3 previews against the opponent's last leads (bo3.py)")
    ap.add_argument("--adapt-k", type=int, default=8, help="our candidate previews for --adapt")
    ap.add_argument("--adapt-prior", type=float, default=0.1, help="weight on the policy's log-probability")
    ap.add_argument("--vary", action="store_true",
                    help="Bo3 games 2+: sample the preview from the policy's top --vary-k, and with --search "
                         "pick among moves within --vary-margin of the best, so game 1 can't be replayed against it")
    ap.add_argument("--vary-k", type=int, default=4)
    ap.add_argument("--counter-t1", type=float, default=0.0,
                    help="with --search, Bo3 games 2+ with the same leads: weight on the opponent's best "
                         "reply to our last turn 1 (0 = off)")
    ap.add_argument("--change-after-loss", action="store_true",
                    help="Bo3 games 2+ after a loss: lead with the policy's likeliest different pair")
    ap.add_argument("--vary-margin", type=float, default=0.02, help="in expected win probability")
    ap.add_argument("--opponent-repeat", type=float, nargs=2, metavar=("AFTER_LOSS", "AFTER_WIN"),
                    help="Bo3: an nn: opponent replays its last preview with these probabilities "
                         "(human-like: 0.24 0.49)")
    ap.add_argument("--opponent-adapt", action="store_true",
                    help="Bo3: an nn: opponent counter-picks our previous game's leads (the Adapter, assuming "
                         "we repeat them), as a human who has seen game 1 would")
    ap.add_argument("--opponent-teams", type=Path, default=None, help="team folder for the opponent (default: --teams)")
    ap.add_argument("--only", nargs="+", default=None, metavar="TEAM",
                    help="draw each match's team at random from just these --teams files (stems)")
    ap.add_argument("--closed-sheets", action="store_true",
                    help="decline open team sheets (Bo1 formats where they're optional): play closed (CTS)")
    ap.add_argument("--gauntlet", type=int, nargs=2, default=None, metavar=("PER_ROUND", "KEEP"),
                    help="with --only: each team plays PER_ROUND series, the worse half is dropped, repeat "
                         "until KEEP are left (results read from --save-dir, see --gauntlet-since)")
    ap.add_argument("--gauntlet-since", default="0000", help="count saved games from this time on (YYYY-MM-DD HH:MM:SS)")
    ap.add_argument("--cycle", nargs="+", default=None, metavar="TEAM",
                    help="use these --teams files (stems, e.g. MC196 MC147) in turn, one per match, not at random")
    ap.add_argument("--sets", type=Path, default=None,
                    help="team folder search and --adapt read EV spreads from (default: --teams). Use the full "
                         "pool when --teams is small, or opponents' spreads fall back to 0 EVs")
    args = ap.parse_args()
    torch.set_num_threads(1)

    def teams() -> RandomPoolTeambuilder:
        if args.gauntlet:
            from ai_vgc.teams import GauntletTeambuilder

            return GauntletTeambuilder(args.teams or TEAMS_DIR, args.only, args.save_dir, args.format,
                                       args.gauntlet_since, *args.gauntlet, user=args.accept)
        if args.only or args.cycle:
            return RandomPoolTeambuilder(args.teams or TEAMS_DIR, names=args.only or args.cycle,
                                         cycle=bool(args.cycle))
        return RandomPoolTeambuilder(args.teams) if args.teams else RandomPoolTeambuilder()

    public = args.server == "showdown"
    if public and not args.accept:
        ap.error("--server showdown only works with --accept")
    # The sim bridge (search) still needs the local Showdown checkout, but not a local server.
    proc = None if public else ensure_server(args.port, args.showdown)
    common = dict(
        battle_format=args.format, max_concurrent_battles=args.concurrency,
        accept_open_team_sheet=not args.closed_sheets,
        server_configuration=ShowdownServerConfiguration if public else ServerConfiguration(
            f"ws://localhost:{args.port}/showdown/websocket", "https://play.pokemonshowdown.com/action.php?"),
    )
    try:
        if args.accept:
            formats = [args.format.removesuffix("bo3"), args.format.removesuffix("bo3") + "bo3"]
            common.update(max_concurrent_battles=1)
            del common["battle_format"]
            password = None
            if public:
                password = os.environ.get("PS_PASSWORD") or \
                    getpass.getpass(f"Showdown password for {args.accept} (blank if unregistered): ") or None
            acct = AccountConfiguration(args.accept, password)
            kw = dict(rules=not args.no_rules, team=teams(), account_configuration=acct, **common)
            if args.search:
                from ai_vgc.nn.search import SearchPlayer
                from ai_vgc.teams import TEAMS_DIR

                class ChallengeSearchPlayer(ChallengeMixin, SearchPlayer):
                    pass

                bot = ChallengeSearchPlayer(args.model, args.rating, args.greedy, opp_model=args.opp_model,
                                            fmt=formats[0], teams_dir=args.sets or args.teams or TEAMS_DIR,
                                            showdown=args.showdown or "pokemon-showdown", k=args.search_k,
                                            opp_k=args.search_opp_k, seeds=args.search_seeds,
                                            prior=args.search_prior, **kw)
                bot.formats = formats
                bot.counter_t1 = args.counter_t1
            else:
                bot = ChallengeNNPlayer(args.model, args.rating, args.greedy, formats=formats, **kw)
            if public:
                _public_login(bot, password)
            bot.vary, bot.vary_k, bot.vary_margin = args.vary, args.vary_k, args.vary_margin
            bot.change_after_loss = args.change_after_loss
            bot.save_dir = args.save_dir / time.strftime("%Y-%m-%d")
            bot.logs_dir = args.logs_dir
            bot.watch_url = "https://play.pokemonshowdown.com/" if public else f"http://localhost:{args.port}/"
            bot.watch_open = args.watch
            if args.from_:
                bot.allowed = {to_id_str(u) for u in args.from_}
            where = "play.pokemonshowdown.com" if public else f"http://localhost:{args.port}"
            who = f" (from {', '.join(args.from_)} only)" if args.from_ else ""
            if args.challenge:
                bot._format = args.format
                them = args.challenge[0]
                print(f"'{args.accept}' challenging {them} to {args.format} at {where}: {args.n} games "
                      f"(they accept in their Showdown client)", flush=True)
                asyncio.run(bot.send_challenges(to_id_str(them), args.n))
            elif args.ladder:
                bot._format = args.format
                print(f"'{args.accept}' laddering {args.format} at {where}: {args.n} games", flush=True)
                asyncio.run(bot.ladder(args.n))
            else:
                print(f"Challenge '{args.accept}' to {' or '.join(formats)} at {where}{who}", flush=True)
                asyncio.run(bot.accept_challenges(None, args.n))
            print(f"{bot.n_won_battles}/{bot.n_finished_battles} games won by {args.accept}")
            if args.search:
                print(f"  search: {bot.searched} decisions, {bot.search_time / max(bot.searched, 1):.2f}s each, "
                      f"{bot.fallbacks} by policy, {bot.sim_errors}/{bot.sims} sims failed, "
                      f"{bot.countered} turn 1s countered")
            return
        if args.search:
            from ai_vgc.nn.search import SearchPlayer
            from ai_vgc.teams import TEAMS_DIR

            kw = {k: v for k, v in common.items() if k != "battle_format"}
            me = SearchPlayer(args.model, args.rating, args.greedy, rules=not args.no_rules,
                              account_configuration=account("nns"), team=teams(), opp_model=args.opp_model,
                              fmt=args.format, teams_dir=args.sets or args.teams or TEAMS_DIR,
                              showdown=args.showdown or "pokemon-showdown", k=args.search_k,
                              opp_k=args.search_opp_k, seeds=args.search_seeds, prior=args.search_prior, **kw)
            me.counter_t1 = args.counter_t1
        else:
            me = NNPlayer(args.model, args.rating, args.greedy, rules=not args.no_rules,
                          account_configuration=account("nn"), team=teams(), **common)
        me.vary, me.vary_k, me.vary_margin = args.vary, args.vary_k, args.vary_margin
        me.change_after_loss = args.change_after_loss
        if args.adapt:
            from ai_vgc.nn.bo3 import Adapter
            from ai_vgc.teams import TEAMS_DIR

            me.adapter = Adapter(args.format, args.sets or args.teams or TEAMS_DIR, args.showdown or "pokemon-showdown",
                                 args.adapt_k, args.adapt_prior, getattr(me, "bridge", None))
        if args.challenge:
            model = me.model
            mine = [me] + [NNPlayer(model, args.rating, args.greedy, rules=not args.no_rules, account_configuration=account("nn"),
                                    team=teams(), **common) for _ in args.challenge[1:]]
            per = [args.n // len(mine) + (i < args.n % len(mine)) for i in range(len(mine))]
            t = time.time()

            async def run() -> None:
                await asyncio.gather(*(p.send_challenges(o, k) for p, o, k in zip(mine, args.challenge, per)))

            asyncio.run(run())
            wins, n = sum(p.n_won_battles for p in mine), sum(p.n_finished_battles for p in mine)
            lo, hi = wilson(wins, n)
            print(f"{Path(args.model).name} vs {args.challenge[0]}: {wins}/{n} = {wins / max(n, 1):.1%}  "
                  f"95% CI [{lo:.1%}, {hi:.1%}]  {time.time() - t:.0f}s")
            return
        if args.opponent.startswith("nn:"):
            opp_team = RandomPoolTeambuilder(args.opponent_teams) if args.opponent_teams else teams()
            opp = NNPlayer(args.opponent[3:], args.rating, args.greedy, rules=not args.opponent_no_rules,
                           account_configuration=account("nnopp"),
                           team=opp_team, **common)
            opp.repeat = tuple(args.opponent_repeat) if args.opponent_repeat else None
            if args.opponent_adapt:
                from ai_vgc.nn.bo3 import Adapter
                from ai_vgc.teams import TEAMS_DIR

                opp.adapter = Adapter(args.format, args.sets or args.opponent_teams or args.teams or TEAMS_DIR,
                                      args.showdown or "pokemon-showdown", args.adapt_k, args.adapt_prior,
                                      repeat=(1.0, 1.0))
        else:
            opp = SimpleHeuristicsPlayer(account_configuration=account("heur"), team=teams(),
                                         **common)
        t = time.time()
        asyncio.run(me.battle_against(opp, n_battles=args.n))
        wins, n = me.n_won_battles, me.n_finished_battles
        lo, hi = wilson(wins, n)
        extra = ""
        if args.search:
            extra = (f"  search: {me.searched} decisions, {me.search_time / max(me.searched, 1):.2f}s each, "
                     f"{me.fallbacks} by policy, {me.sim_errors}/{me.sims} sims failed, "
                     f"{me.countered} turn 1s countered")
        print(f"{Path(args.model).name}{'+search' if args.search else ''} vs {args.opponent}: "
              f"{wins}/{n} = {wins / max(n, 1):.1%}  95% CI [{lo:.1%}, {hi:.1%}]  {time.time() - t:.0f}s{extra}")
        if me.games:
            print(f"  Bo3: {series_summary(me.games)}")
        if me.adapter:
            log = me.adapter.log
            changed = sum(a["pick"] > 0 for a in log)
            print(f"  adapted previews: {len(log)}, not the policy's top one: {changed}")
        if getattr(opp, "adapter", None):
            log = opp.adapter.log
            changed = sum(a["pick"] > 0 for a in log)
            print(f"  opponent adapted previews: {len(log)}, not the policy's top one: {changed}")
    finally:
        if proc:
            proc.terminate()


if __name__ == "__main__":
    main()
