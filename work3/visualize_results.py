import argparse
import csv
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Plot CGAN training curves from history.csv.")
    parser.add_argument("--history", default="work3/results/cgan_fashion_mnist/history.csv")
    parser.add_argument("--output", default="work3/results/cgan_fashion_mnist/curves.png")
    return parser.parse_args()


def main():
    args = parse_args()
    history_path = Path(args.history)
    rows = []
    with history_path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({k: float(v) for k, v in row.items()})

    import matplotlib.pyplot as plt

    epochs = [int(row["epoch"]) for row in rows]
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(epochs, [row["d_loss"] for row in rows], label="D loss")
    ax1.plot(epochs, [row["g_loss"] for row in rows], label="G loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.grid(alpha=0.25)
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(epochs, [row["real_score"] for row in rows], "--", label="D(real)")
    ax2.plot(epochs, [row["fake_score"] for row in rows], "--", label="D(fake)")
    ax2.set_ylabel("sigmoid score")
    ax2.legend(loc="upper right")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    print(f"saved curves to {out_path}")


if __name__ == "__main__":
    main()
