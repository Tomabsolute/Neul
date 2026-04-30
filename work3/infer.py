import argparse
import sys
from pathlib import Path

import torch
from torchvision import utils

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.models import ConditionalGenerator


def parse_args():
    parser = argparse.ArgumentParser(description="Run conditional image generation.")
    parser.add_argument("--checkpoint", default="work3/results/cgan_fashion_mnist/checkpoints/best_generator.pt")
    parser.add_argument("--output", default="work3/results/figures/inference_grid.png")
    parser.add_argument("--labels", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--repeat", type=int, default=8)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    args = parse_args()
    device = choose_device(args.device)
    torch.manual_seed(args.seed)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    z_dim = int(checkpoint.get("z_dim", 100))
    embed_dim = int(checkpoint.get("embed_dim", 50))
    num_classes = int(checkpoint.get("num_classes", 10))

    generator = ConditionalGenerator(z_dim=z_dim, num_classes=num_classes, embed_dim=embed_dim).to(device)
    generator.load_state_dict(checkpoint["generator_state_dict"])
    generator.eval()

    base_labels = [int(x.strip()) for x in args.labels.split(",") if x.strip()]
    labels = torch.tensor(base_labels, dtype=torch.long, device=device).repeat_interleave(args.repeat)
    noise = torch.randn(labels.size(0), z_dim, device=device)

    with torch.no_grad():
        images = generator(noise, labels)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    utils.save_image(images, out_path, nrow=args.repeat, normalize=True, value_range=(-1, 1))
    print(f"saved generated grid to {out_path}")


if __name__ == "__main__":
    main()
