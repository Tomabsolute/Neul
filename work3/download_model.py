import argparse
import hashlib
import urllib.request
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Download a trained generator checkpoint.")
    parser.add_argument("--url", required=True, help="Checkpoint URL from Gitee Release, GitHub Release or a direct file host.")
    parser.add_argument("--output", default="work3/results/cgan_fashion_mnist/checkpoints/best_generator.pt")
    parser.add_argument("--sha256", default="", help="Optional SHA256 checksum.")
    return parser.parse_args()


def main():
    args = parse_args()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {args.url}")
    urllib.request.urlretrieve(args.url, out_path)
    if args.sha256:
        digest = hashlib.sha256(out_path.read_bytes()).hexdigest()
        if digest.lower() != args.sha256.lower():
            raise SystemExit(f"checksum mismatch: got {digest}, expected {args.sha256}")
    print(f"saved checkpoint to {out_path}")


if __name__ == "__main__":
    main()
