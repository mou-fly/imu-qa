import argparse
import csv
import os
import random
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

from data_provider.data_factory import data_provider
from pretrain_model import IMURQVAEAlignModel


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class LossWeights:
    recon: float
    vq: float
    align: float


def compute_total_loss(outputs: Dict[str, torch.Tensor], weights: LossWeights, stage: str) -> torch.Tensor:
    if stage == "rqvae":
        return weights.recon * outputs["recon_loss"] + weights.vq * outputs["vq_loss"]
    if stage == "joint":
        return (
            weights.recon * outputs["recon_loss"]
            + weights.vq * outputs["vq_loss"]
            + weights.align * outputs["align_loss"]
        )
    raise ValueError(f"Unknown stage: {stage}")


def run_one_epoch(
    model: nn.Module,
    loader,
    optimizer,
    device: torch.device,
    stage: str,
    weights: LossWeights,
    log_interval: int,
    phase: str,
    train: bool = True,
) -> Tuple[float, float, float, float]:
    model.train(train)
    total_loss = 0.0
    total_recon = 0.0
    total_vq = 0.0
    total_align = 0.0
    n_steps = 0

    for step, (batch_x, batch_y, mask) in enumerate(loader, start=1):
        imu_x = torch.as_tensor(batch_x, dtype=torch.float32, device=device)
        text = torch.as_tensor(batch_y, dtype=torch.long, device=device)
        text_mask = torch.as_tensor(mask, dtype=torch.long, device=device)

        if train:
            optimizer.zero_grad()

        outputs = model(imu_x, text, text_mask, compute_align=(stage == "joint"))
        loss = compute_total_loss(outputs, weights, stage)

        if train:
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item())
        total_recon += float(outputs["recon_loss"].item())
        total_vq += float(outputs["vq_loss"].item())
        total_align += float(outputs["align_loss"].item())
        n_steps += 1
        if step % max(log_interval, 1) == 0:
            if stage == "joint":
                print(
                    f"[{phase} step {step:04d}] "
                    f"loss={float(loss.item()):.6f}, "
                    f"recon={float(outputs['recon_loss'].item()):.6f}, "
                    f"vq={float(outputs['vq_loss'].item()):.6f}, "
                    f"align={float(outputs['align_loss'].item()):.6f}"
                )
            else:
                print(
                    f"[{phase} step {step:04d}] "
                    f"loss={float(loss.item()):.6f}, "
                    f"recon={float(outputs['recon_loss'].item()):.6f}, "
                    f"vq={float(outputs['vq_loss'].item()):.6f}"
                )

    if n_steps == 0:
        return 0.0, 0.0, 0.0, 0.0
    return (
        total_loss / n_steps,
        total_recon / n_steps,
        total_vq / n_steps,
        total_align / n_steps,
    )


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val: float,
    args: argparse.Namespace,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "epoch": epoch,
        "best_val_loss": best_val,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": vars(args),
    }
    torch.save(payload, path)


def train_pretrain(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() or "cpu" in args.device else "cpu")
    print(f"[Info] Using device: {device}")

    train_set, train_loader = data_provider(args, "train")
    val_set, val_loader = data_provider(args, "val")
    print(f"[Info] Train samples: {len(train_set)}, Val samples: {len(val_set)}")

    model = IMURQVAEAlignModel(args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    loss_weights = LossWeights(recon=args.lambda_recon, vq=args.lambda_vq, align=args.lambda_align)
    save_dir = os.path.join(args.root_path, args.ckpt_dir, args.exp_name)
    best_ckpt = os.path.join(save_dir, "best.pth")
    last_ckpt = os.path.join(save_dir, "last.pth")
    os.makedirs(save_dir, exist_ok=True)

    stage_tag = {"rqvae": "1", "joint": "2"}.get(args.stage, str(args.stage))
    datastamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_prefix = f"pretrain-stage_{stage_tag}-{datastamp}"
    log_txt = os.path.join(save_dir, f"{log_prefix}.log")
    metrics_csv = os.path.join(save_dir, f"{log_prefix}.csv")
    with open(log_txt, "a", encoding="utf-8") as f:
        f.write(f"\n[{datetime.now().isoformat()}] Start training stage={args.stage}\n")
    with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "epoch",
                "train_loss",
                "train_recon",
                "train_vq",
                "train_align",
                "val_loss",
                "val_recon",
                "val_vq",
                "val_align",
            ]
        )
    print(f"[Info] Log file: {log_txt}")
    print(f"[Info] Metrics file: {metrics_csv}")

    best_val = float("inf")
    for epoch in range(1, args.epoch + 1):
        train_metrics = run_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            stage=args.stage,
            weights=loss_weights,
            log_interval=args.log_interval,
            phase="train",
            train=True,
        )
        val_metrics = run_one_epoch(
            model=model,
            loader=val_loader,
            optimizer=optimizer,
            device=device,
            stage=args.stage,
            weights=loss_weights,
            log_interval=args.log_interval,
            phase="val",
            train=False,
        )

        train_loss, train_recon, train_vq, train_align = train_metrics
        val_loss, val_recon, val_vq, val_align = val_metrics

        if args.stage == "joint":
            msg = (
                f"[Epoch {epoch:03d}] "
                f"train_loss={train_loss:.6f} (recon={train_recon:.6f}, vq={train_vq:.6f}, align={train_align:.6f}) | "
                f"val_loss={val_loss:.6f} (recon={val_recon:.6f}, vq={val_vq:.6f}, align={val_align:.6f})"
            )
        else:
            msg = (
                f"[Epoch {epoch:03d}] "
                f"train_loss={train_loss:.6f} (recon={train_recon:.6f}, vq={train_vq:.6f}) | "
                f"val_loss={val_loss:.6f} (recon={val_recon:.6f}, vq={val_vq:.6f})"
            )
        print(msg)
        with open(log_txt, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
        with open(metrics_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    epoch,
                    f"{train_loss:.6f}",
                    f"{train_recon:.6f}",
                    f"{train_vq:.6f}",
                    f"{train_align:.6f}",
                    f"{val_loss:.6f}",
                    f"{val_recon:.6f}",
                    f"{val_vq:.6f}",
                    f"{val_align:.6f}",
                ]
            )

        save_checkpoint(last_ckpt, model, optimizer, epoch, best_val, args)
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(best_ckpt, model, optimizer, epoch, best_val, args)
            best_msg = f"[Info] New best checkpoint saved: {best_ckpt}"
            print(best_msg)
            with open(log_txt, "a", encoding="utf-8") as f:
                f.write(best_msg + "\n")
