from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


def choose_cjk_font() -> str | None:
    preferred = [
        "Noto Sans CJK SC",
        "Noto Sans CJK TC",
        "Noto Serif CJK SC",
        "WenQuanYi Micro Hei",
        "WenQuanYi Zen Hei",
        "Source Han Sans SC",
        "SimHei",
        "Microsoft YaHei",
        "PingFang SC",
        "Heiti SC",
        "Arial Unicode MS",
    ]
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in preferred:
        if name in available:
            return name
    return None


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


def plot_embedding_neighbors(neighbors: dict[str, list[tuple[str, float]]], output: str | Path) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    labels = []
    scores = []
    for query, rows in neighbors.items():
        for token, score in rows[:5]:
            labels.append(f"{query}->{token}")
            scores.append(score)
    if not labels:
        return
    fig, ax = plt.subplots(figsize=(8, max(3, len(labels) * 0.22)))
    cjk_font = choose_cjk_font()
    font_kwargs = {"fontproperties": cjk_font} if cjk_font else {}
    y = list(range(len(labels)))
    ax.barh(y, scores, color="#4C78A8")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, **font_kwargs)
    ax.invert_yaxis()
    ax.set_xlabel("cosine similarity")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
