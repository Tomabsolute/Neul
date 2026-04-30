import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, utils

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.models import ConditionalDiscriminator, ConditionalGenerator, init_dcgan_weights


CLASS_NAMES = {
    "fashion-mnist": [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ],
    "mnist": [str(i) for i in range(10)],
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train a small conditional DCGAN.")
    parser.add_argument("--dataset", choices=["fashion-mnist", "mnist"], default="fashion-mnist")
    parser.add_argument("--data-root", default="work3/data")
    parser.add_argument("--output-dir", default="work3/results/cgan_fashion_mnist")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-samples", type=int, default=6000)
    parser.add_argument("--z-dim", type=int, default=100)
    parser.add_argument("--embed-dim", type=int, default=50)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--sample-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto, cuda, mps or cpu")
    parser.add_argument("--download", action="store_true", help="Download dataset if missing.")
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_dataset(args):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    dataset_cls = datasets.FashionMNIST if args.dataset == "fashion-mnist" else datasets.MNIST
    dataset = dataset_cls(args.data_root, train=True, transform=transform, download=args.download)
    if args.num_samples and args.num_samples < len(dataset):
        generator = torch.Generator().manual_seed(args.seed)
        indices = torch.randperm(len(dataset), generator=generator)[: args.num_samples].tolist()
        dataset = Subset(dataset, indices)
    return dataset


@torch.no_grad()
def save_fixed_grid(generator, fixed_noise, fixed_labels, out_path: Path) -> None:
    generator.eval()
    fake = generator(fixed_noise, fixed_labels)
    utils.save_image(fake, out_path, nrow=10, normalize=True, value_range=(-1, 1))
    generator.train()


def save_checkpoint(path: Path, generator, discriminator, args, history, epoch: int) -> None:
    ckpt = {
        "epoch": epoch,
        "dataset": args.dataset,
        "z_dim": args.z_dim,
        "embed_dim": args.embed_dim,
        "num_classes": 10,
        "class_names": CLASS_NAMES[args.dataset],
        "generator_state_dict": generator.state_dict(),
        "discriminator_state_dict": discriminator.state_dict(),
        "args": vars(args),
        "history": history,
    }
    torch.save(ckpt, path)


def write_history(path: Path, history) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "d_loss", "g_loss", "real_score", "fake_score", "seconds"])
        writer.writeheader()
        writer.writerows(history)


def plot_history(path: Path, history) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    epochs = [row["epoch"] for row in history]
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(epochs, [row["d_loss"] for row in history], label="D loss")
    ax1.plot(epochs, [row["g_loss"] for row in history], label="G loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.grid(alpha=0.25)
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(epochs, [row["real_score"] for row in history], "--", color="tab:green", label="D(real)")
    ax2.plot(epochs, [row["fake_score"] for row in history], "--", color="tab:red", label="D(fake)")
    ax2.set_ylabel("sigmoid score")
    ax2.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = choose_device(args.device)

    output_dir = Path(args.output_dir)
    sample_dir = output_dir / "samples"
    ckpt_dir = output_dir / "checkpoints"
    sample_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_dataset(args)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )

    generator = ConditionalGenerator(args.z_dim, 10, args.embed_dim).to(device)
    discriminator = ConditionalDiscriminator(10).to(device)
    generator.apply(init_dcgan_weights)
    discriminator.apply(init_dcgan_weights)

    criterion = nn.BCEWithLogitsLoss()
    opt_g = optim.Adam(generator.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    opt_d = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    fixed_labels = torch.arange(10, device=device).repeat_interleave(10)
    fixed_noise = torch.randn(fixed_labels.size(0), args.z_dim, device=device)
    history = []
    start_all = time.time()

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        d_loss_sum = 0.0
        g_loss_sum = 0.0
        real_score_sum = 0.0
        fake_score_sum = 0.0
        steps = 0

        for real_images, labels in loader:
            real_images = real_images.to(device)
            labels = labels.to(device)
            batch_size = real_images.size(0)
            real_targets = torch.ones(batch_size, device=device)
            fake_targets = torch.zeros(batch_size, device=device)

            opt_d.zero_grad(set_to_none=True)
            real_logits = discriminator(real_images, labels)
            noise = torch.randn(batch_size, args.z_dim, device=device)
            fake_images = generator(noise, labels)
            fake_logits = discriminator(fake_images.detach(), labels)
            d_loss = criterion(real_logits, real_targets) + criterion(fake_logits, fake_targets)
            d_loss.backward()
            opt_d.step()

            opt_g.zero_grad(set_to_none=True)
            fake_logits_for_g = discriminator(fake_images, labels)
            g_loss = criterion(fake_logits_for_g, real_targets)
            g_loss.backward()
            opt_g.step()

            with torch.no_grad():
                d_loss_sum += d_loss.item()
                g_loss_sum += g_loss.item()
                real_score_sum += torch.sigmoid(real_logits).mean().item()
                fake_score_sum += torch.sigmoid(fake_logits).mean().item()
                steps += 1

        row = {
            "epoch": epoch,
            "d_loss": d_loss_sum / steps,
            "g_loss": g_loss_sum / steps,
            "real_score": real_score_sum / steps,
            "fake_score": fake_score_sum / steps,
            "seconds": time.time() - start,
        }
        history.append(row)
        print(
            f"epoch {epoch:03d}/{args.epochs} "
            f"d_loss={row['d_loss']:.4f} g_loss={row['g_loss']:.4f} "
            f"D(real)={row['real_score']:.3f} D(fake)={row['fake_score']:.3f}"
        )

        if epoch % args.sample_every == 0 or epoch == args.epochs:
            save_fixed_grid(generator, fixed_noise, fixed_labels, sample_dir / f"epoch_{epoch:03d}.png")

        save_checkpoint(ckpt_dir / "last_checkpoint.pt", generator, discriminator, args, history, epoch)

    save_checkpoint(ckpt_dir / "best_generator.pt", generator, discriminator, args, history, args.epochs)
    write_history(output_dir / "history.csv", history)
    plot_history(output_dir / "curves.png", history)
    metrics = {
        "dataset": args.dataset,
        "num_samples": len(dataset),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "z_dim": args.z_dim,
        "total_seconds": time.time() - start_all,
        "final_d_loss": history[-1]["d_loss"],
        "final_g_loss": history[-1]["g_loss"],
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"saved results to {output_dir}")


if __name__ == "__main__":
    main()
