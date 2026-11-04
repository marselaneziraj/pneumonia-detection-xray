"""Two-phase fine-tuning with checkpoint selection at the shipped operating point.

Phase 1: backbone frozen (BatchNorm in eval mode), only the new head trains.
Phase 2: everything unfrozen, AdamW + cosine decay, early stopping on the selection metric.
Selection metric: val specificity at the recall target (tie-break: val AUROC).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .config import run_dir
from .data import make_loader, read_manifests, resolve_root
from .metrics import choose_threshold, threshold_free
from .model import build_model, count_params, init_head_bias, set_backbone_trainable
from .utils import get_device, library_versions, set_seed, utc_now


@torch.no_grad()
def predict(model, loader, device, limit_batches: int | None = None, criterion=None):
    model.eval()
    scores, labels, idxs, losses = [], [], [], []
    for b, (x, y, i) in enumerate(loader):
        if limit_batches is not None and b >= limit_batches:
            break
        x, y = x.to(device, non_blocking=True), y.to(device)
        logits = model(x).float()
        if criterion is not None:
            losses.append(criterion(logits, y).item() * len(y))
        scores.append(torch.sigmoid(logits).cpu().numpy())
        labels.append(y.cpu().numpy())
        idxs.append(i.numpy())
    s, y, i = np.concatenate(scores), np.concatenate(labels).astype(int), np.concatenate(idxs)
    loss = float(np.sum(losses) / len(y)) if losses else float("nan")
    return s, y, i, loss


def _train_epoch(model, loader, optimizer, criterion, device, scaler, freeze_bn, limit):
    model.train()
    if freeze_bn:
        model.features.eval()  # frozen backbone: don't update BN running stats either
    total, n = 0.0, 0
    for b, (x, y, _) in enumerate(loader):
        if limit is not None and b >= limit:
            break
        x, y = x.to(device, non_blocking=True), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=scaler is not None):
            logits = model(x)
        loss = criterion(logits.float(), y)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total += loss.item() * len(y)
        n += len(y)
    return total / max(n, 1)


def train(
    cfg: dict,
    limit_train_batches: int | None = None,
    limit_val_batches: int | None = None,
    device: str | None = None,
) -> Path:
    seed = int(cfg["experiment"]["seed"])
    set_seed(seed, deterministic=bool(cfg["train"].get("deterministic", True)))
    dev = get_device(device or cfg["train"].get("device", "auto"))
    out = run_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)

    limit_train_batches = limit_train_batches or cfg["train"].get("limit_train_batches")
    limit_val_batches = limit_val_batches or cfg["train"].get("limit_val_batches")

    root = resolve_root(cfg)
    splits = read_manifests(cfg["data"]["splits_dir"])
    train_loader = make_loader(cfg, splits["train"], root, train=True)
    val_loader = make_loader(cfg, splits["val"], root, train=False)

    n_pos = int(splits["train"]["label"].sum())
    n_neg = int(len(splits["train"]) - n_pos)
    pos_weight = n_neg / n_pos  # < 1 here: pushes the model away from the lazy majority answer
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=dev))
    target = float(cfg["selection"]["target_recall"])

    m = cfg["model"]
    model = build_model(m["arch"], pretrained=bool(m["pretrained"]), dropout=float(m["dropout"]))
    init_head_bias(model, n_pos / (n_pos + n_neg))
    model.to(dev)
    use_amp = bool(cfg["train"].get("amp", True)) and dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    print(
        f"[train] {cfg['experiment']['name']}: arch={m['arch']} pretrained={m['pretrained']} "
        f"device={dev} amp={use_amp} train={len(splits['train'])} (pos={n_pos}, neg={n_neg}) "
        f"val={len(splits['val'])} pos_weight={pos_weight:.3f} target_recall={target}"
    )

    history, best = [], {"score": (-1.0, -1.0), "epoch": None}
    p1, p2 = cfg["train"]["phase1"], cfg["train"]["phase2"]
    phases = []
    if int(p1["epochs"]) > 0:
        phases.append(("phase1_head", p1, False))
    if int(p2["epochs"]) > 0:
        phases.append(("phase2_full", p2, True))

    global_epoch = 0
    for phase_name, pc, full in phases:
        set_backbone_trainable(model, full)
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=float(pc["lr"]), weight_decay=float(pc["weight_decay"]))
        epochs = int(pc["epochs"])
        sched = (
            torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=epochs, eta_min=float(pc.get("min_lr", 0))
            )
            if pc.get("scheduler", "none") == "cosine"
            else None
        )
        patience = pc.get("early_stopping_patience")
        stale = 0
        print(
            f"[train] {phase_name}: {count_params(model, True):,} trainable params, "
            f"lr={pc['lr']}, epochs<={epochs}"
        )
        for _ in range(epochs):
            global_epoch += 1
            t0 = time.time()
            lr = opt.param_groups[0]["lr"]
            tr_loss = _train_epoch(
                model,
                train_loader,
                opt,
                criterion,
                dev,
                scaler,
                freeze_bn=not full,
                limit=limit_train_batches,
            )
            s, y, _, va_loss = predict(model, val_loader, dev, limit_val_batches, criterion)
            tf = threshold_free(y, s)
            if y.sum() > 0:
                op = choose_threshold(y, s, target)
            else:  # only possible in tiny smoke runs
                op = {"threshold": 0.5, "recall": float("nan"), "specificity": float("nan")}
            if sched is not None:
                sched.step()
            row = {
                "epoch": global_epoch,
                "phase": phase_name,
                "lr": lr,
                "train_loss": tr_loss,
                "val_loss": va_loss,
                "val_auroc": tf["auroc"],
                "val_auprc": tf["auprc"],
                "val_threshold": op["threshold"],
                "val_recall": op["recall"],
                "val_specificity": op["specificity"],
                "seconds": round(time.time() - t0, 1),
            }
            history.append(row)
            pd.DataFrame(history).to_csv(out / "history.csv", index=False)
            score = (
                np.nan_to_num(op["specificity"], nan=-1.0),
                np.nan_to_num(tf["auroc"], nan=-1.0),
            )
            improved = score > best["score"]
            print(
                f"  epoch {global_epoch:2d} [{phase_name}] train_loss={tr_loss:.4f} "
                f"val_loss={va_loss:.4f} AUROC={tf['auroc']:.4f} @recall>={target}: "
                f"thr={op['threshold']:.4f} spec={op['specificity']:.4f}"
                f"{'  *best*' if improved else ''} ({row['seconds']}s)"
            )
            if improved:
                best = {"score": score, "epoch": global_epoch}
                stale = 0
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "config": cfg,
                        "epoch": global_epoch,
                        "val_threshold": op["threshold"],
                        "val_operating_point": op,
                        "pos_weight": pos_weight,
                    },
                    out / "best.pt",
                )
            else:
                stale += 1
                if patience is not None and full and stale >= int(patience):
                    print(f"[train] early stop: no improvement for {patience} epochs")
                    break

    meta = {
        "experiment": cfg["experiment"]["name"],
        "config_path": cfg.get("_config_path"),
        "best_epoch": best["epoch"],
        "best_val_specificity_at_target": best["score"][0],
        "best_val_auroc": best["score"][1],
        "target_recall": target,
        "pos_weight": pos_weight,
        "n_train": len(splits["train"]),
        "n_val": len(splits["val"]),
        "seed": seed,
        "device": str(dev),
        "amp": use_amp,
        "limit_train_batches": limit_train_batches,
        "finished_at": utc_now(),
        "versions": library_versions(),
    }
    with open(out / "train_meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"[train] best epoch {best['epoch']} -> {out / 'best.pt'}")
    return out
