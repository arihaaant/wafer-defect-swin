import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from . import data
from .model import FocalLoss, WaferSwin, mixup


@dataclass
class Config:
    data: str
    out: str
    epochs: int = 20
    batch_size: int = 64
    mixup_alpha: float = 0.2
    focal_gamma: float = 2.0
    recon_weight: float = 0.3
    lr_backbone: float = 1e-5
    lr_head: float = 1e-4
    seed: int = 42
    num_workers: int = 2
    limit: int | None = None
    pretrained: bool = True
    amp: bool = True
    balanced_sampler: bool = True
    class_weights: bool = True


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_amp_dtype(cfg, device):
    if not cfg.amp or device.type != "cuda":
        return None
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def run_epoch(model, loader, criterion, cfg, device, optimizer=None, dtype=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    l1 = nn.L1Loss()
    totals = {"loss": 0.0, "cls_loss": 0.0, "recon_loss": 0.0}
    preds, labels = [], []

    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if training and cfg.mixup_alpha > 0:
                x, y_a, y_b, lam = mixup(x, y, cfg.mixup_alpha)
            else:
                y_a, y_b, lam = y, y, 1.0
            target = x[:, :1] * 0.5 + 0.5

            with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
                logits, recon = model(x)
                cls_loss = lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
                recon_loss = l1(recon.float(), target)
                loss = cls_loss + cfg.recon_weight * recon_loss

            if training:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

            totals["loss"] += loss.item()
            totals["cls_loss"] += cls_loss.item()
            totals["recon_loss"] += recon_loss.item()
            preds += logits.argmax(1).tolist()
            labels += y_a.tolist()

    n = max(len(loader), 1)
    res = {k: v / n for k, v in totals.items()}
    res["macro_f1"] = f1_score(labels, preds, average="macro", zero_division=0)
    return res, preds, labels


def train(cfg, df=None):
    seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    dtype = get_amp_dtype(cfg, device)
    scaler = torch.amp.GradScaler(device.type, enabled=dtype == torch.float16)

    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2))

    if df is None:
        df = data.load_lswmd(cfg.data)
    if cfg.limit:
        df = df.sample(min(cfg.limit, len(df)), random_state=cfg.seed)
    train_df, val_df, test_df, class_names = data.split(df, cfg.seed)
    train_dl, val_dl, test_dl = data.make_loaders(train_df, val_df, test_df, cfg.batch_size,
                                                  cfg.num_workers, cfg.balanced_sampler)
    n_cls = len(class_names)
    print(f"device={device} amp={dtype} train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    weight = None
    if cfg.class_weights:
        w = 1.0 / np.maximum(data.class_counts(train_df, n_cls), 1)
        weight = torch.tensor(w / w.sum() * n_cls, dtype=torch.float32)
    criterion = FocalLoss(cfg.focal_gamma, weight).to(device)

    model = WaferSwin(n_cls, pretrained=cfg.pretrained).to(device)
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": cfg.lr_backbone},
        {"params": model.classifier.parameters(), "lr": cfg.lr_head},
        {"params": model.decoder.parameters(), "lr": cfg.lr_head},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=1e-7)

    history, best_f1 = [], -1.0
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        tr, _, _ = run_epoch(model, train_dl, criterion, cfg, device, optimizer, dtype, scaler)
        va, _, _ = run_epoch(model, val_dl, criterion, cfg, device, dtype=dtype)
        scheduler.step()
        history.append({"epoch": epoch, "train": tr, "val": va})
        print(f"epoch {epoch:02d}  train loss {tr['loss']:.4f} f1 {tr['macro_f1']:.4f}  "
              f"val loss {va['loss']:.4f} f1 {va['macro_f1']:.4f}  {time.time() - t0:.0f}s")
        if va["macro_f1"] > best_f1:
            best_f1 = va["macro_f1"]
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "class_names": class_names, "val_macro_f1": best_f1}, out / "best.pt")
    (out / "history.json").write_text(json.dumps(history, indent=2))

    ckpt = torch.load(out / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    te, preds, labels = run_epoch(model, test_dl, criterion, cfg, device, dtype=dtype)
    idx = list(range(n_cls))
    metrics = {
        "best_epoch": ckpt["epoch"],
        "val_macro_f1": best_f1,
        "test_macro_f1": te["macro_f1"],
        "test_accuracy": float(np.mean(np.array(preds) == np.array(labels))),
        "per_class": classification_report(labels, preds, labels=idx, target_names=class_names,
                                           zero_division=0, output_dict=True),
        "confusion_matrix": confusion_matrix(labels, preds, labels=idx).tolist(),
        "class_names": class_names,
        "n_test": len(labels),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"test macro f1 {metrics['test_macro_f1']:.4f} (epoch {metrics['best_epoch']})")
    return metrics


def main(argv=None):
    p = argparse.ArgumentParser()
    for name, field in Config.__dataclass_fields__.items():
        if name in ("data", "out"):
            p.add_argument("--" + name, required=True)
        elif isinstance(field.default, bool):
            p.add_argument("--no-" + name.replace("_", "-"), dest=name, action="store_false")
        else:
            t = int if name == "limit" else type(field.default)
            p.add_argument("--" + name.replace("_", "-"), type=t, default=field.default)
    train(Config(**vars(p.parse_args(argv))))


if __name__ == "__main__":
    main()
