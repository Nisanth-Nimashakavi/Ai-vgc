"""Distill Showdown policies into native-v3 using paired public observations.

The teacher sees its original input contract. The student sees v3 on the SAME
public battle object, never the simulator's opposing team sets or requests.
Soft targets include the autoregressive second head conditioned on a sampled
teacher first action. Validation is split by whole battle, not decision rows.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from ai_vgc.nn.encode import encode
from ai_vgc.nn.model import masked, slot_b_mask
from ai_vgc.nn.player import NNPlayer, _set_native_cts_flags, configure_inference, load_policy

INPUTS = ("tok_cat", "tok_num", "mv_cat", "mv_dyn", "dmg", "glob", "act_tok", "mask",
          "ser_tok", "ser_mv", "ser_glob")


def tensor_batch(arrays, device):
    result = {}
    for key, array in arrays.items():
        t = torch.as_tensor(array)
        if t.dtype in (torch.uint16, torch.uint32, torch.uint64):
            t = t.long()
        result[key] = t.to(device)
    return result


def public_pair(battle, model, rating, mask, sets, fmt):
    """Re-encode only public information and restore teacher flags afterward."""
    fields = ("_native_cts", "_native_cts_v2", "_native_cts_v3", "_spd_on", "_dmg_on", "_guess")
    old = {key: battle.__dict__.get(key) for key in fields}
    try:
        _set_native_cts_flags(battle, model)
        teacher = encode(battle, rating)
        _set_native_cts_flags(battle, SimpleNamespace(observation_contract="native-cts-v3"))
        battle._spd_on = battle._dmg_on = True
        battle._guess = (str(sets), fmt)
        student = encode(battle, rating)
    finally:
        for key, value in old.items():
            if value is None:
                battle.__dict__.pop(key, None)
            else:
                setattr(battle, key, value)
    teacher["mask"] = mask.copy()
    student["mask"] = mask.copy()
    return teacher, student


def soft_losses(model, b):
    a = b["previous_a"].long()
    ma, mb = b["mask"][:, 0], slot_b_mask(b["mask"][:, 1], a)
    la, lb, value = model(b, a)
    sa = F.log_softmax(masked(la.float(), ma), -1)
    sb = F.log_softmax(masked(lb.float(), mb), -1)
    ta, tb = b["teacher_logp"][:, 0].float(), b["teacher_logp"][:, 1].float()
    pa, pb = ta.exp(), tb.exp()
    # Quantized stored probabilities are re-normalized before use.
    pa, pb = pa / pa.sum(-1, keepdim=True), pb / pb.sum(-1, keepdim=True)
    per_row = ((pa * (pa.clamp_min(1e-30).log() - sa)).sum(-1)
               + (pb * (pb.clamp_min(1e-30).log() - sb)).sum(-1))
    weight = b.get("weight", torch.ones_like(per_row)).float()
    kl = (per_row * weight).sum() / weight.sum().clamp_min(1)
    vf = F.binary_cross_entropy_with_logits(value.float(), b["teacher_value"].float())
    agree = ((sa.argmax(-1) == ta.argmax(-1)).float()
             + (sb.argmax(-1) == tb.argmax(-1)).float()).mean() / 2
    return kl, vf, agree


def load_replay(folder, training_only=False):
    paths = sorted(Path(folder).glob("*.npz"))
    if not paths:
        raise ValueError(f"No distillation shards in {folder}")
    chunks = []
    for path in paths:
        with np.load(path) as data:
            chunk = {key: data[key] for key in data.files}
            chunk.setdefault("weight", np.ones(len(chunk["battle_id"]), np.float32))
            chunks.append(chunk)
    arrays = {key: np.concatenate([part[key] for part in chunks]) for key in chunks[0]}
    if training_only:
        keep = arrays["battle_id"] % 10 != 0
        arrays = {key: value[keep] for key, value in arrays.items()}
    return tensor_batch(arrays, "cpu")


class Recorder(NNPlayer):
    def __init__(self, path, *, output, label, sets, fmt, device="cpu", **kwargs):
        super().__init__(path, 1700., record=True, greedy=False, **kwargs)
        self.model.to(device)
        configure_inference(self, sets, fmt)
        self.output, self.label, self.sets, self.fmt = Path(output), label, sets, fmt
        self.rows = {}
        self.completed = []
        self.chunk = 0

    async def _decide(self, battle, mask, preview):
        original, student = public_pair(battle, self.model, self.rating, mask, self.sets, self.fmt)
        device = next(self.model.parameters()).device
        b = tensor_batch({key: value[None] for key, value in original.items()}, device)
        with torch.no_grad():
            enc = self.model.encode(b)
            la = F.log_softmax(masked(self.model.slot_logits(enc, b, 0).float(), b["mask"][:, 0]), -1)
            a = torch.distributions.Categorical(logits=la).sample()
            mb = slot_b_mask(b["mask"][:, 1], a)
            lb = F.log_softmax(masked(self.model.slot_logits(enc, b, 1, a).float(), mb), -1)
            other = torch.distributions.Categorical(logits=lb).sample()
            value = self.model.value(enc[0])[:, 0].sigmoid()
        action = np.asarray([a.item(), other.item()], np.int16)
        bid = int.from_bytes(hashlib.blake2b(battle.battle_tag.encode(), digest_size=8).digest(), "little") % (2**63)
        self.rows.setdefault(battle.battle_tag, []).append(student | {
            "previous_a": np.int16(action[0]), "teacher_logp": torch.stack([la[0], lb[0]]).cpu().numpy().astype(np.float16),
            "teacher_value": np.float32(value.item()), "preview": np.bool_(preview), "battle_id": np.int64(bid)})
        return action

    def _battle_finished_callback(self, battle):
        self.completed.extend(self.rows.pop(battle.battle_tag, []))
        if len(self.completed) >= 2048:
            self.flush()
        super()._battle_finished_callback(battle)

    def flush(self):
        if not self.completed:
            return
        self.output.mkdir(parents=True, exist_ok=True)
        path = self.output / f"{self.label}-{self.chunk:04d}.npz"
        if path.exists():
            raise FileExistsError(path)
        np.savez_compressed(path, **{key: np.stack([r[key] for r in self.completed]) for key in self.completed[0]})
        print(f"distill data {self.label}: {len(self.completed)} states -> {path.name}", flush=True)
        self.completed.clear()
        self.chunk += 1


def generate(args):
    from poke_env import ServerConfiguration
    from poke_env.teambuilder import ConstantTeambuilder
    from ai_vgc.showdown import ensure_server, account
    from ai_vgc.teams import RandomPoolTeambuilder
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    proc = ensure_server(args.port, str(args.showdown))
    kw = dict(battle_format=args.format, max_concurrent_battles=4, accept_open_team_sheet=False,
              server_configuration=ServerConfiguration(f"ws://localhost:{args.port}/showdown/websocket", "https://play.pokemonshowdown.com/action.php?"),
              log_level=40)
    me = Recorder(args.teacher, output=args.out / "mc301", label=f"shard{args.seed}",
                  sets=args.sets, fmt=args.format, device=args.device, team=ConstantTeambuilder(args.my_team.read_text()),
                  account_configuration=account("dst301"), **kw)
    opponent = Recorder(args.generalist, output=args.out / "v6", label=f"shard{args.seed}",
                        sets=args.sets, fmt=args.format, device=args.device,
                        team=RandomPoolTeambuilder(args.teams, cycle=True), account_configuration=account("dstv6"), **kw)
    try:
        asyncio.run(me.battle_against(opponent, n_battles=args.games))
        if me.n_finished_battles != args.games or opponent.n_finished_battles != args.games:
            raise RuntimeError("Incomplete distillation collection")
        me.flush(); opponent.flush()
        print(f"Collected {args.games} CTS games, both policies' public states", flush=True)
    finally:
        if proc:
            proc.terminate()


def train(args):
    from ai_vgc.nn.rl import save
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    data = load_replay(args.data)
    val = data["battle_id"] % 10 == 0
    tr_idx, va_idx = torch.where(~val)[0], torch.where(val)[0]
    if not len(tr_idx) or not len(va_idx):
        raise ValueError("Need separate training and validation battles")
    model = load_policy(args.init, args.device).eval()
    # This marks a new input contract, retaining the same network architecture.
    model.observation_contract = args.observation_contract = "native-cts-v3"
    args.speed_inference = args.damage_inference = args.set_guess = True
    args.closed_sheets = args.native_closed_sheets = True
    args.native_cts_version = 3
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        raise FileExistsError(args.out)
    best = float("inf")
    history = []

    def batch(indices):
        return {key: value[indices].to(args.device) for key, value in data.items()}

    for epoch in range(args.epochs + 1):
        if epoch:
            shuffled = tr_idx[torch.randperm(len(tr_idx))]
            for indices in shuffled.split(args.batch):
                kl, vf, _ = soft_losses(model, batch(indices))
                loss = kl + args.value_coef * vf
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite distillation loss")
                opt.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                opt.step()
        sums = np.zeros(3)
        with torch.no_grad():
            for indices in va_idx.split(args.batch):
                values = soft_losses(model, batch(indices))
                sums += np.asarray([v.item() for v in values]) * len(indices)
        kl, vf, agreement = sums / len(va_idx)
        rec = dict(epoch=epoch, validation_kl=float(kl), validation_value_bce=float(vf),
                   agreement=float(agreement), train_rows=len(tr_idx), validation_rows=len(va_idx))
        history.append(rec)
        print(json.dumps(rec), flush=True)
        if kl < best:
            best = kl
            save(model, args.out, args, {"distillation": rec, "teacher": str(args.init)})
    args.out.with_suffix(".distillation.json").write_text(json.dumps(history, indent=2) + "\n")
    if best > args.max_validation_kl:
        raise RuntimeError(f"Distillation gate failed: validation KL {best:.4f} > {args.max_validation_kl}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="mode", required=True)
    gen = sub.add_parser("gen")
    for name in ("teacher", "generalist", "my-team", "teams", "sets", "showdown", "out"):
        gen.add_argument(f"--{name}", required=True, type=Path)
    gen.add_argument("--games", type=int, default=1042)
    gen.add_argument("--port", type=int, default=19810)
    gen.add_argument("--format", default="gen9championsvgc2026regmc")
    gen.add_argument("--seed", type=int, default=0)
    gen.add_argument("--device", default="cpu")
    fit = sub.add_parser("train")
    for name in ("init", "data", "out"):
        fit.add_argument(f"--{name}", required=True, type=Path)
    fit.add_argument("--epochs", type=int, default=8)
    fit.add_argument("--batch", type=int, default=512)
    fit.add_argument("--lr", type=float, default=1e-4)
    fit.add_argument("--value-coef", type=float, default=.1)
    fit.add_argument("--max-validation-kl", type=float, default=.2)
    fit.add_argument("--my-team", type=Path, default=None)
    fit.add_argument("--seed", type=int, default=0)
    fit.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    generate(args) if args.mode == "gen" else train(args)


if __name__ == "__main__":
    # poke-env's global background event loop can hang interpreter shutdown.
    import os, sys, traceback
    try:
        main()
    except BaseException:
        traceback.print_exc(); sys.stdout.flush(); sys.stderr.flush(); os._exit(1)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)
