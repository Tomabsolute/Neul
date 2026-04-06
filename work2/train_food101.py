#!/usr/bin/env python3
import argparse
import csv
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms


@dataclass
class EvalMetrics:
    loss: float
    acc: float
    macro_f1: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def macro_f1_score(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    conf = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(conf, (y_true, y_pred), 1)

    f1s: List[float] = []
    for c in range(num_classes):
        tp = conf[c, c]
        fp = conf[:, c].sum() - tp
        fn = conf[c, :].sum() - tp
        denom = 2 * tp + fp + fn
        f1s.append(0.0 if denom == 0 else (2.0 * tp) / denom)
    return float(np.mean(f1s))


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def build_food101_loaders(
    data_root: Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
    val_ratio: float,
    seed: int,
) -> Tuple[Dict[str, DataLoader], Dict[str, int], int]:
    tf_train = transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    tf_eval = transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    base_train = datasets.Food101(root=str(data_root), split="train", download=True)
    n_total = len(base_train)
    n_val = max(1, int(n_total * val_ratio))
    n_train = n_total - n_val

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_total, generator=g).tolist()
    train_idx, val_idx = perm[:n_train], perm[n_train:]

    train_src = datasets.Food101(root=str(data_root), split="train", transform=tf_train, download=True)
    val_src = datasets.Food101(root=str(data_root), split="train", transform=tf_eval, download=True)
    test_src = datasets.Food101(root=str(data_root), split="test", transform=tf_eval, download=True)

    datasets_map = {
        "train": Subset(train_src, train_idx),
        "val": Subset(val_src, val_idx),
        "test": test_src,
    }

    loaders = {
        split: DataLoader(
            datasets_map[split],
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        for split in ["train", "val", "test"]
    }

    sizes = {k: len(v) for k, v in datasets_map.items()}
    num_classes = len(base_train.classes)
    return loaders, sizes, num_classes


def build_model(model_name: str, mode: str, num_classes: int) -> nn.Module:
    if model_name == "resnext50_32x4d":
        weights = models.ResNeXt50_32X4D_Weights.DEFAULT if mode == "finetune" else None
        model = models.resnext50_32x4d(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if model_name == "densenet121":
        weights = models.DenseNet121_Weights.DEFAULT if mode == "finetune" else None
        model = models.densenet121(weights=weights)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        return model

    raise ValueError(f"Unsupported model: {model_name}")


def set_trainable(model: nn.Module, model_name: str, train_backbone: bool) -> None:
    m = unwrap_model(model)
    for p in m.parameters():
        p.requires_grad = train_backbone

    if model_name == "resnext50_32x4d":
        for p in m.fc.parameters():
            p.requires_grad = True
    elif model_name == "densenet121":
        for p in m.classifier.parameters():
            p.requires_grad = True


def split_head_backbone_params(model: nn.Module, model_name: str):
    m = unwrap_model(model)
    if model_name == "resnext50_32x4d":
        head = list(m.fc.parameters())
    elif model_name == "densenet121":
        head = list(m.classifier.parameters())
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    head_ids = {id(p) for p in head}
    backbone = [p for p in m.parameters() if id(p) not in head_ids]
    return backbone, head


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    optimizer: optim.Optimizer = None,
    scaler: torch.cuda.amp.GradScaler = None,
    use_amp: bool = False,
) -> EvalMetrics:
    is_train = optimizer is not None
    model.train(is_train)

    running_loss = 0.0
    preds_all: List[np.ndarray] = []
    labels_all: List[np.ndarray] = []

    for inputs, labels in dataloader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            if is_train:
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        preds_all.append(outputs.argmax(dim=1).detach().cpu().numpy())
        labels_all.append(labels.detach().cpu().numpy())

    y_pred = np.concatenate(preds_all)
    y_true = np.concatenate(labels_all)

    return EvalMetrics(
        loss=running_loss / len(dataloader.dataset),
        acc=float((y_pred == y_true).mean()),
        macro_f1=macro_f1_score(y_true, y_pred, num_classes),
    )


def save_history_csv(history: List[dict], save_path: Path) -> None:
    if not history:
        return
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def save_curves(history: List[dict], save_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    epochs = [r["epoch"] for r in history]
    train_loss = [r["train_loss"] for r in history]
    val_loss = [r["val_loss"] for r in history]
    train_acc = [r["train_acc"] for r in history]
    val_acc = [r["val_acc"] for r in history]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, train_loss, label="train")
    axes[0].plot(epochs, val_loss, label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, train_acc, label="train")
    axes[1].plot(epochs, val_acc, label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def append_summary_csv(row: dict, summary_csv: Path) -> None:
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    exists = summary_csv.exists()
    with summary_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Work2: Food101 scratch vs finetune")
    p.add_argument("--data-root", type=str, default="./work2/data")
    p.add_argument("--out-dir", type=str, default="./work2/results")
    p.add_argument("--model", choices=["resnext50_32x4d", "densenet121"], required=True)
    p.add_argument("--mode", choices=["scratch", "finetune"], required=True)

    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--multi-gpu", action="store_true")

    p.add_argument("--scratch-epochs", type=int, default=60)
    p.add_argument("--scratch-lr", type=float, default=0.01)

    p.add_argument("--freeze-epochs", type=int, default=10)
    p.add_argument("--finetune-epochs", type=int, default=40)
    p.add_argument("--head-lr", type=float, default=0.001)
    p.add_argument("--backbone-lr", type=float, default=0.0001)

    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count() if device.type == "cuda" else 0
    use_amp = bool(args.amp and device.type == "cuda")

    out_root = Path(args.out_dir)
    exp_name = f"food101_{args.model}_{args.mode}"
    exp_dir = out_root / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    loaders, sizes, num_classes = build_food101_loaders(
        data_root=Path(args.data_root),
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    model = build_model(args.model, args.mode, num_classes).to(device)
    if args.multi_gpu and num_gpus > 1:
        model = nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    history: List[dict] = []
    best_val_acc = -1.0
    best_path = exp_dir / "best_model.pt"

    t0 = time.time()

    if args.mode == "scratch":
        opt = optim.SGD(model.parameters(), lr=args.scratch_lr, momentum=args.momentum, weight_decay=args.weight_decay)
        sch = lr_scheduler.CosineAnnealingLR(opt, T_max=args.scratch_epochs)

        for epoch in range(1, args.scratch_epochs + 1):
            tr = run_epoch(model, loaders["train"], criterion, device, num_classes, opt, scaler, use_amp)
            va = run_epoch(model, loaders["val"], criterion, device, num_classes)
            sch.step()

            history.append({
                "epoch": epoch,
                "phase": "scratch",
                "train_loss": tr.loss,
                "train_acc": tr.acc,
                "train_macro_f1": tr.macro_f1,
                "val_loss": va.loss,
                "val_acc": va.acc,
                "val_macro_f1": va.macro_f1,
                "lr": opt.param_groups[0]["lr"],
            })
            print(f"[Epoch {epoch:03d}/{args.scratch_epochs}] train_acc={tr.acc:.4f} val_acc={va.acc:.4f}")

            if va.acc > best_val_acc:
                best_val_acc = va.acc
                torch.save(model.state_dict(), best_path)

    else:
        # stage 1: head only
        set_trainable(model, args.model, train_backbone=False)
        head_params = [p for p in model.parameters() if p.requires_grad]
        opt1 = optim.SGD(head_params, lr=args.head_lr, momentum=args.momentum, weight_decay=args.weight_decay)
        sch1 = lr_scheduler.StepLR(opt1, step_size=max(1, args.freeze_epochs // 2), gamma=0.1)

        for epoch in range(1, args.freeze_epochs + 1):
            tr = run_epoch(model, loaders["train"], criterion, device, num_classes, opt1, scaler, use_amp)
            va = run_epoch(model, loaders["val"], criterion, device, num_classes)
            sch1.step()

            history.append({
                "epoch": epoch,
                "phase": "finetune_head",
                "train_loss": tr.loss,
                "train_acc": tr.acc,
                "train_macro_f1": tr.macro_f1,
                "val_loss": va.loss,
                "val_acc": va.acc,
                "val_macro_f1": va.macro_f1,
                "lr": opt1.param_groups[0]["lr"],
            })
            print(f"[Head {epoch:03d}/{args.freeze_epochs}] train_acc={tr.acc:.4f} val_acc={va.acc:.4f}")

            if va.acc > best_val_acc:
                best_val_acc = va.acc
                torch.save(model.state_dict(), best_path)

        # stage 2: all layers
        set_trainable(model, args.model, train_backbone=True)
        backbone, head = split_head_backbone_params(model, args.model)
        opt2 = optim.SGD(
            [{"params": backbone, "lr": args.backbone_lr}, {"params": head, "lr": args.head_lr}],
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
        sch2 = lr_scheduler.CosineAnnealingLR(opt2, T_max=max(1, args.finetune_epochs))

        for i in range(1, args.finetune_epochs + 1):
            epoch = args.freeze_epochs + i
            tr = run_epoch(model, loaders["train"], criterion, device, num_classes, opt2, scaler, use_amp)
            va = run_epoch(model, loaders["val"], criterion, device, num_classes)
            sch2.step()

            history.append({
                "epoch": epoch,
                "phase": "finetune_all",
                "train_loss": tr.loss,
                "train_acc": tr.acc,
                "train_macro_f1": tr.macro_f1,
                "val_loss": va.loss,
                "val_acc": va.acc,
                "val_macro_f1": va.macro_f1,
                "lr": opt2.param_groups[0]["lr"],
            })
            print(f"[Full {i:03d}/{args.finetune_epochs}] train_acc={tr.acc:.4f} val_acc={va.acc:.4f}")

            if va.acc > best_val_acc:
                best_val_acc = va.acc
                torch.save(model.state_dict(), best_path)

    train_seconds = time.time() - t0

    model.load_state_dict(torch.load(best_path, map_location=device))
    te = run_epoch(model, loaders["test"], criterion, device, num_classes)

    save_history_csv(history, exp_dir / "history.csv")
    save_curves(history, exp_dir / "curves.png")

    payload = {
        "experiment": exp_name,
        "dataset": "food101",
        "model": args.model,
        "mode": args.mode,
        "device": str(device),
        "num_gpus": num_gpus,
        "multi_gpu": bool(args.multi_gpu and num_gpus > 1),
        "num_classes": num_classes,
        "dataset_sizes": sizes,
        "best_val_acc": best_val_acc,
        "test": asdict(te),
        "train_seconds": train_seconds,
        "train_hours": train_seconds / 3600.0,
        "args": vars(args),
    }
    (exp_dir / "metrics.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    append_summary_csv(
        {
            "experiment": exp_name,
            "model": args.model,
            "mode": args.mode,
            "dataset": "food101",
            "best_val_acc": f"{best_val_acc:.6f}",
            "test_acc": f"{te.acc:.6f}",
            "test_macro_f1": f"{te.macro_f1:.6f}",
            "train_seconds": f"{train_seconds:.1f}",
            "train_hours": f"{train_seconds / 3600.0:.4f}",
            "device": str(device),
        },
        Path(args.out_dir) / "summary.csv",
    )

    print("\n=== Done ===")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
