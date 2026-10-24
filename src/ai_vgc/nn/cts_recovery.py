"""Recorded public-state policy comparisons and student-trajectory distillation."""
from __future__ import annotations

import argparse
import asyncio
from contextlib import contextmanager
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ai_vgc.nn.cts_distill import Recorder, public_pair, tensor_batch
from ai_vgc.nn.encode import encode
from ai_vgc.nn.model import masked, slot_b_mask
from ai_vgc.nn.player import NNPlayer, configure_inference, load_policy, _set_native_cts_flags
from ai_vgc.nn.preview import preview_candidates, group_preview_candidates


@contextmanager
def policy_view(battle, model, sets, fmt):
    """Use that policy's live contract; restore all flags on the public battle."""
    from types import SimpleNamespace
    fields = ("_native_cts", "_native_cts_v2", "_native_cts_v3", "_spd_on", "_dmg_on", "_guess")
    old = {key: battle.__dict__.get(key) for key in fields}
    config = SimpleNamespace(model=model)
    configure_inference(config, sets, fmt)
    try:
        _set_native_cts_flags(battle, model)
        # Original checkpoints retain their existing preview behavior. V3 now
        # matches its training-time inference projection at both boundaries.
        active = not battle.teampreview or getattr(model, "observation_contract", "") == "native-cts-v3"
        battle._spd_on = config.speed_inference if active else False
        battle._dmg_on = config.damage_inference if active else False
        battle._guess = config.set_guess if active else None
        yield
    finally:
        for key, value in old.items():
            if value is None:
                battle.__dict__.pop(key, None)
            else:
                setattr(battle, key, value)


@torch.no_grad()
def distribution(model, obs, previous=None):
    device = next(model.parameters()).device
    batch = tensor_batch({key: value[None] for key, value in obs.items()}, device)
    enc = model.encode(batch)
    la = F.log_softmax(masked(model.slot_logits(enc, batch, 0).float(), batch["mask"][:, 0]), -1)
    a = la.argmax(-1) if previous is None else torch.tensor([previous], device=device)
    lb = F.log_softmax(masked(model.slot_logits(enc, batch, 1, a).float(),
                             slot_b_mask(batch["mask"][:, 1], a)), -1)
    return la[0].cpu().numpy(), lb[0].cpu().numpy(), float(model.value(enc[0])[:, 0].sigmoid().item())


class RecoveryPlayer(Recorder):
    def __init__(self, teacher, behavior, *, audit, collect, case, **kwargs):
        super().__init__(teacher, **kwargs)
        self.teacher = self.model
        self.model = load_policy(behavior, str(next(self.teacher.parameters()).device))
        configure_inference(self, self.sets, self.fmt)
        self.record, self.greedy = False, True
        self.audit, self.collect, self.case = Path(audit), collect, case
        self.audit.mkdir(parents=True, exist_ok=False)
        self.preview_orders = {}
        self.agreements = [0, 0]

    def journal(self, filename, row):
        with (self.audit / filename).open("a") as handle:
            handle.write(json.dumps(row) + "\n")

    def pair(self, battle, mask):
        with policy_view(battle, self.teacher, self.sets, self.fmt):
            original, student = public_pair(battle, self.teacher, self.rating, mask, self.sets, self.fmt)
        with policy_view(battle, self.model, self.sets, self.fmt):
            behavior = encode(battle, self.rating)
        behavior["mask"] = mask.copy()
        return original, student, behavior

    def append_target(self, battle, obs, la, lb, value, a, preview, disagreement):
        if not self.collect:
            return
        bid = int.from_bytes(hashlib.blake2b((self.label + battle.battle_tag).encode(), digest_size=8).digest(), "little") % (2**63)
        self.rows.setdefault(battle.battle_tag, []).append(obs | {
            "previous_a": np.int16(a),
            "teacher_logp": np.clip(np.stack([la, lb]), -1e4, 0).astype(np.float16),
            "teacher_value": np.float32(value), "preview": np.bool_(preview), "battle_id": np.int64(bid),
            "weight": np.float32((2 if preview else 1) * (3 if disagreement else 1))})

    async def _decide(self, battle, mask, preview):
        original, student, behavior = self.pair(battle, mask)
        ba, bb, _ = distribution(self.model, behavior)
        action = np.array([ba.argmax(), bb.argmax()], np.int16)
        ta, tb, value = distribution(self.teacher, original, int(action[0]))
        ga, gb, _ = distribution(self.teacher, original)
        teacher_action = [int(ga.argmax()), int(gb.argmax())]
        differs = teacher_action != action.tolist()
        self.agreements[0] += int(not differs); self.agreements[1] += 1
        self.append_target(battle, student, ta, tb, value, int(action[0]), preview, differs)
        # Also train the conditional B head at the teacher's A when they differ.
        if action[0] != teacher_action[0]:
            self.append_target(battle, student, ga, gb, value, teacher_action[0], preview, True)
        self.journal("decisions.jsonl", dict(tag=battle.battle_tag, turn=battle.turn, preview=preview,
                     teacher=teacher_action, behavior=action.tolist(), disagreement=differs,
                     our_active=[m.species if m else None for m in battle.active_pokemon],
                     their_active=[m.species if m else None for m in battle.opponent_active_pokemon]))
        return action

    async def _policy_preview(self, battle):
        team = list(battle.team.values())
        selected = [m.selected_in_teampreview for m in team]
        try:
            orders = []
            for model in (self.teacher, self.model):
                with policy_view(battle, model, self.sets, self.fmt):
                    candidates = preview_candidates(model, battle, self.rating)
                groups = group_preview_candidates(candidates, len(team))
                orders.append(groups[0][2] if groups else candidates[0][0])
            self.preview_orders[battle.battle_tag] = orders
            self.journal("previews.jsonl", dict(tag=battle.battle_tag, teacher=orders[0], behavior=orders[1],
                         composition_agrees=sorted(orders[0]) == sorted(orders[1]),
                         leads_agree=orders[0][:2] == orders[1][:2], team=[m.species for m in team]))
            # Enumerate every ordered lead pair: preserve the distribution used
            # by greedy composition selection, not just sampled preview states.
            if self.collect:
                for leads in [(), *itertools.permutations(range(len(team)), 2)]:
                    for i, mon in enumerate(team):
                        mon._selected_in_teampreview = i in leads
                    original, student, behavior = self.pair(battle, encode(battle)["mask"])
                    ba, bb, _ = distribution(self.model, behavior)
                    a = int(ba.argmax())
                    ta, tb, value = distribution(self.teacher, original, a)
                    ga, gb, _ = distribution(self.teacher, original)
                    differs = a != int(ga.argmax()) or int(bb.argmax()) != int(gb.argmax())
                    self.append_target(battle, student, ta, tb, value, a, True, differs)
                    if a != int(ga.argmax()):
                        self.append_target(battle, student, ga, gb, value, int(ga.argmax()), True, True)
            return "/team " + orders[1]
        finally:
            for mon, flag in zip(team, selected):
                mon._selected_in_teampreview = flag

    def _battle_finished_callback(self, battle):
        self.journal("games.jsonl", dict(tag=battle.battle_tag, case=self.case, won=battle.won,
                     lost=battle.lost, turns=battle.turn, previews=self.preview_orders.pop(battle.battle_tag, None),
                     log=battle._build_replay_log()))
        super()._battle_finished_callback(battle)


def generate(args):
    from poke_env import ServerConfiguration
    from poke_env.teambuilder import ConstantTeambuilder
    from ai_vgc.showdown import ensure_server, account
    from ai_vgc.teams import RandomPoolTeambuilder
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    cases = json.loads(args.cases.read_text())
    summary = []
    for index, case in enumerate(cases):
        port = args.port + index
        proc = ensure_server(port, str(args.showdown))
        common = dict(battle_format=args.format, max_concurrent_battles=4, accept_open_team_sheet=False,
                      server_configuration=ServerConfiguration(f"ws://localhost:{port}/showdown/websocket", "https://play.pokemonshowdown.com/action.php?"), log_level=40)
        me = RecoveryPlayer(args.teacher, args.behavior, audit=args.audit / case["name"], collect=args.collect,
                            case=case["name"], output=args.out, label=f"recovery{args.seed}-{index}",
                            sets=args.sets, fmt=args.format, device=args.device,
                            team=ConstantTeambuilder(args.my_team.read_text()), account_configuration=account("repair"), **common)
        opponent_kwargs = dict(team=RandomPoolTeambuilder(Path(case["teams"]), cycle=case.get("cycle", False)),
                               account_configuration=account("repopp"), **common)
        if args.collect and case.get("distill", False):
            opponent = Recorder(case["model"], output=args.out.parent / "specialists" / case["name"],
                                label=f"opponent{args.seed}", sets=args.sets, fmt=args.format,
                                device=args.device, **opponent_kwargs)
        else:
            opponent = NNPlayer(case["model"], 1700., True, **opponent_kwargs)
        opponent.model.to(args.device)
        configure_inference(opponent, args.sets, args.format)
        try:
            asyncio.run(me.battle_against(opponent, n_battles=case["games"]))
            if me.n_finished_battles != case["games"]:
                raise RuntimeError("Incomplete recovery battles")
            me.flush()
            if isinstance(opponent, Recorder):
                opponent.flush()
            row = dict(name=case["name"], wins=me.n_won_battles, games=me.n_finished_battles,
                       joint_action_agreement=me.agreements[0] / max(me.agreements[1], 1))
            summary.append(row)
            print(json.dumps(row), flush=True)
        finally:
            if proc:
                proc.terminate()
    args.audit.joinpath("summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("teacher", "behavior", "cases", "sets", "my-team", "showdown", "out", "audit"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--seed", type=int, default=201)
    parser.add_argument("--port", type=int, default=25200)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--format", default="gen9championsvgc2026regmc")
    generate(parser.parse_args())


if __name__ == "__main__":
    import os, sys, traceback
    try:
        main()
    except BaseException:
        traceback.print_exc(); sys.stdout.flush(); sys.stderr.flush(); os._exit(1)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)
