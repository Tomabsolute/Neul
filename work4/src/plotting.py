from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_lm_history(history: list[dict], output: str | Path) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    val_loss = [row["val_loss"] for row in history]
    val_ppl = [row["val_ppl"] for row in history]

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(epochs, train_loss, label="train loss")
    ax1.plot(epochs, val_loss, label="val loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("cross entropy")
    ax1.grid(alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(epochs, val_ppl, color="tab:green", label="val ppl")
    ax2.set_ylabel("perplexity")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [line.get_label() for line in lines], loc="best")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
