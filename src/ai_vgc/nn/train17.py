"""Training script for Wolfe_Alt6767 17-Token Transformer (~7.5M parameters).

Trains the model to imitate the high-throughput heuristic bot self-play dataset
via zero-copy memory-mapped binary streaming.
"""

from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ai_vgc.nn.model17 import Transformer17

STEP_DTYPE = np.dtype([
    ('tok_cat', np.uint32, (17, 4)),
    ('tok_num', np.float32, (17, 16)),
    ('action_a', np.uint8),
    ('action_b', np.uint8),
    ('opp_action_a', np.uint8),
    ('opp_action_b', np.uint8),
    ('preview_plan', np.uint8),
    ('opp_preview_plan', np.uint8),
    ('outcome', np.int8),
    ('is_preview', np.bool_),
    ('turn', np.uint16),
    ('_pad', np.uint16),
])


class Binary17Dataset(Dataset):
    """Zero-copy memory-mapped dataset for EncodedStep17 binary files."""

    def __init__(self, bin_path: str | Path):
        self.path = Path(bin_path)
        if not self.path.exists():
            raise FileNotFoundError(f"Binary file not found: {self.path}")

        file_size = self.path.stat().st_size
        item_size = STEP_DTYPE.itemsize
        if file_size % item_size != 0:
            raise ValueError(
                f"File size {file_size} is not a multiple of record size {item_size}"
            )
        self.length = file_size // item_size
        self.data = np.memmap(self.path, dtype=STEP_DTYPE, mode="r", shape=(self.length,))

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.data[idx]
        return {
            "tok_cat": torch.from_numpy(row["tok_cat"].copy()).long(),
            "tok_num": torch.from_numpy(row["tok_num"].copy()).float(),
            "action_a": torch.tensor(row["action_a"], dtype=torch.long),
            "action_b": torch.tensor(row["action_b"], dtype=torch.long),
            "opp_action_a": torch.tensor(row["opp_action_a"], dtype=torch.long),
            "opp_action_b": torch.tensor(row["opp_action_b"], dtype=torch.long),
            "preview_plan": torch.tensor(row["preview_plan"], dtype=torch.long),
            "outcome": torch.tensor(row["outcome"], dtype=torch.float32),
            "is_preview": torch.tensor(row["is_preview"], dtype=torch.bool),
        }


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    epoch: int,
    grad_clip: float = 1.0,
    is_main: bool = True,
) -> dict[str, float]:
    model.train()
    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()

    total_loss = 0.0
    act_a_correct = 0
    act_b_correct = 0
    preview_correct = 0
    total_samples = 0

    t0 = time.time()
    for step, batch in enumerate(loader):
        tok_cat = batch["tok_cat"].to(device, non_blocking=True)
        tok_num = batch["tok_num"].to(device, non_blocking=True)
        action_a = batch["action_a"].to(device, non_blocking=True)
        action_b = batch["action_b"].to(device, non_blocking=True)
        opp_a = batch["opp_action_a"].to(device, non_blocking=True)
        opp_b = batch["opp_action_b"].to(device, non_blocking=True)
        preview_plan = batch["preview_plan"].to(device, non_blocking=True)
        outcome = batch["outcome"].to(device, non_blocking=True)

        optimizer.zero_grad()
        with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            # Teacher-force Slot B with action_a during training
            out = model(tok_cat, tok_num, action_a=action_a)

            loss_act_a = ce_loss(out["policy_a"], action_a)
            loss_act_b = ce_loss(out["policy_b"], action_b)
            loss_prev = ce_loss(out["preview_plan"], preview_plan)
            loss_val = mse_loss(out["value"].squeeze(-1), outcome)
            loss_opp_a = ce_loss(out["opp_a"], opp_a)
            loss_opp_b = ce_loss(out["opp_b"], opp_b)

            loss = (
                loss_act_a
                + loss_act_b
                + 0.5 * loss_prev
                + 0.5 * loss_val
                + 0.25 * (loss_opp_a + loss_opp_b)
            )

        scaler.scale(loss).backward()
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        bs = tok_cat.shape[0]
        total_loss += loss.item() * bs
        act_a_correct += (out["policy_a"].argmax(dim=-1) == action_a).sum().item()
        act_b_correct += (out["policy_b"].argmax(dim=-1) == action_b).sum().item()
        preview_correct += (out["preview_plan"].argmax(dim=-1) == preview_plan).sum().item()
        total_samples += bs

        if is_main and ((step + 1) % 100 == 0 or (step + 1) == len(loader)):
            elapsed = time.time() - t0
            rate = total_samples / max(elapsed, 1e-4)
            print(
                f"Epoch {epoch:2d} [{step+1:4d}/{len(loader)}] "
                f"loss: {total_loss / total_samples:.4f} | "
                f"act_a acc: {act_a_correct / total_samples:.3f} | "
                f"act_b acc: {act_b_correct / total_samples:.3f} | "
                f"prev acc: {preview_correct / total_samples:.3f} | "
                f"{rate:.1f} samples/s",
                flush=True,
            )

    return {
        "loss": total_loss / max(total_samples, 1),
        "act_a_acc": act_a_correct / max(total_samples, 1),
        "act_b_acc": act_b_correct / max(total_samples, 1),
        "prev_acc": preview_correct / max(total_samples, 1),
    }


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()

    total_loss = 0.0
    act_a_correct = 0
    act_b_correct = 0
    preview_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in loader:
            tok_cat = batch["tok_cat"].to(device)
            tok_num = batch["tok_num"].to(device)
            action_a = batch["action_a"].to(device)
            action_b = batch["action_b"].to(device)
            opp_a = batch["opp_action_a"].to(device)
            opp_b = batch["opp_action_b"].to(device)
            preview_plan = batch["preview_plan"].to(device)
            outcome = batch["outcome"].to(device)

            with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                # In evaluation, Slot B is conditioned on Slot A's prediction
                out = model(tok_cat, tok_num, action_a=None)

                loss_act_a = ce_loss(out["policy_a"], action_a)
                loss_act_b = ce_loss(out["policy_b"], action_b)
                loss_prev = ce_loss(out["preview_plan"], preview_plan)
                loss_val = mse_loss(out["value"].squeeze(-1), outcome)

                loss = loss_act_a + loss_act_b + 0.5 * loss_prev + 0.5 * loss_val

            bs = tok_cat.shape[0]
            total_loss += loss.item() * bs
            act_a_correct += (out["policy_a"].argmax(dim=-1) == action_a).sum().item()
            act_b_correct += (out["policy_b"].argmax(dim=-1) == action_b).sum().item()
            preview_correct += (out["preview_plan"].argmax(dim=-1) == preview_plan).sum().item()
            total_samples += bs

    return {
        "val_loss": total_loss / max(total_samples, 1),
        "val_act_a_acc": act_a_correct / max(total_samples, 1),
        "val_act_b_acc": act_b_correct / max(total_samples, 1),
        "val_prev_acc": preview_correct / max(total_samples, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Train 17-token transformer model")
    parser.add_argument("--data", type=str, required=True, help="Path to .bin dataset")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--val-split", type=float, default=0.05, help="Fraction for validation")
    parser.add_argument("--out", type=str, default="data/models/model17_imitation.pt", help="Checkpoint save path")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers per GPU")
    args = parser.parse_args()

    # DDP setup
    is_ddp = "LOCAL_RANK" in os.environ
    if is_ddp:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group("nccl")
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        device = torch.device(f"cuda:{local_rank}")
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if rank == 0:
        print(f"Training on device: {device} (world_size={world_size}, is_ddp={is_ddp})", flush=True)

    # Load dataset
    full_ds = Binary17Dataset(args.data)
    total_len = len(full_ds)
    val_len = int(total_len * args.val_split)
    train_len = total_len - val_len

    if rank == 0:
        print(f"Loaded dataset from {args.data}: {total_len:,} decisions ({train_len:,} train, {val_len:,} val)", flush=True)

    train_ds, val_ds = torch.utils.data.random_split(
        full_ds,
        [train_len, val_len],
        generator=torch.Generator().manual_seed(42),
    )

    train_sampler = (
        torch.utils.data.distributed.DistributedSampler(
            train_ds, num_replicas=world_size, rank=rank, shuffle=True
        )
        if is_ddp
        else None
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = Transformer17().to(device)
    if is_ddp:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if rank == 0:
        print(f"Initialized Transformer17 with {n_params:,} parameters ({n_params / 1e6:.2f}M)", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        if is_ddp:
            train_sampler.set_epoch(epoch)

        train_metrics = train_epoch(
            model, train_loader, optimizer, scaler, device, epoch, is_main=(rank == 0)
        )

        if rank == 0:
            if val_len > 0:
                eval_model = model.module if is_ddp else model
                val_metrics = evaluate(eval_model, val_loader, device)
                print(
                    f"--> Epoch {epoch:2d} Val Loss: {val_metrics['val_loss']:.4f} | "
                    f"Val Act A: {val_metrics['val_act_a_acc']:.3f} | "
                    f"Val Act B: {val_metrics['val_act_b_acc']:.3f} | "
                    f"Val Prev: {val_metrics['val_prev_acc']:.3f}",
                    flush=True,
                )
                if val_metrics["val_loss"] < best_val_loss:
                    best_val_loss = val_metrics["val_loss"]
                    torch.save(
                        {
                            "model": eval_model.state_dict(),
                            "epoch": epoch,
                            "val_loss": best_val_loss,
                            "args": vars(args),
                        },
                        out_path,
                    )
                    print(f"  [*] Saved new best checkpoint to {out_path}", flush=True)
            else:
                eval_model = model.module if is_ddp else model
                torch.save(
                    {
                        "model": eval_model.state_dict(),
                        "epoch": epoch,
                        "args": vars(args),
                    },
                    out_path,
                )
                print(f"  [*] Saved checkpoint to {out_path}", flush=True)

    if is_ddp:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
