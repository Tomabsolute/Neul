#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def read_summary_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Summary CSV not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def to_float(row: Dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except Exception:
        return float("nan")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_metric_bars(rows: List[Dict[str, str]], out_dir: Path) -> None:
    labels = [r["experiment"] for r in rows]
    test_acc = [to_float(r, "test_acc") for r in rows]
    test_f1 = [to_float(r, "test_macro_f1") for r in rows]
    train_h = [to_float(r, "train_hours") for r in rows]

    x = np.arange(len(labels))
    width = 0.35

    # 1) Accuracy + F1
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, test_acc, width, label="test_acc")
    ax.bar(x + width / 2, test_f1, width, label="test_macro_f1")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Work2: Accuracy / Macro-F1 Comparison")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "compare_acc_f1.png", dpi=160)
    plt.close(fig)

    # 2) Train time
    fig, ax = plt.subplots(figsize=(10, 4.5))
    bars = ax.bar(x, train_h)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Hours")
    ax.set_title("Work2: Training Time Comparison")
    ax.grid(axis="y", alpha=0.25)

    for b, v in zip(bars, train_h):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.2f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_dir / "compare_train_time.png", dpi=160)
    plt.close(fig)


def plot_mode_pairwise(rows: List[Dict[str, str]], out_dir: Path) -> None:
    # Build pairs for each model: scratch vs finetune
    grouped: Dict[str, Dict[str, Dict[str, str]]] = {}
    for r in rows:
        model = r["model"]
        mode = r["mode"]
        grouped.setdefault(model, {})[mode] = r

    for model, modes in grouped.items():
        if "scratch" not in modes or "finetune" not in modes:
            continue

        scratch = modes["scratch"]
        finetune = modes["finetune"]

        labels = ["scratch", "finetune"]
        acc = [to_float(scratch, "test_acc"), to_float(finetune, "test_acc")]
        f1 = [to_float(scratch, "test_macro_f1"), to_float(finetune, "test_macro_f1")]
        hours = [to_float(scratch, "train_hours"), to_float(finetune, "train_hours")]

        x = np.arange(2)
        width = 0.35

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        axes[0].bar(x - width / 2, acc, width, label="test_acc")
        axes[0].bar(x + width / 2, f1, width, label="macro_f1")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(labels)
        axes[0].set_ylim(0, 1)
        axes[0].set_title(f"{model}: score")
        axes[0].legend()
        axes[0].grid(axis="y", alpha=0.25)

        axes[1].bar(x, hours, width=0.5)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(labels)
        axes[1].set_title(f"{model}: train hours")
        axes[1].grid(axis="y", alpha=0.25)

        fig.tight_layout()
        fig.savefig(out_dir / f"pair_{model}.png", dpi=160)
        plt.close(fig)


def generate_table_md(rows: List[Dict[str, str]], out_path: Path) -> None:
    lines = []
    lines.append("| experiment | best_val_acc | test_acc | test_macro_f1 | train_hours |")
    lines.append("|---|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            f"| {r['experiment']} | {r['best_val_acc']} | {r['test_acc']} | {r['test_macro_f1']} | {r['train_hours']} |"
        )

    total_h = sum(to_float(r, "train_hours") for r in rows)
    lines.append("")
    lines.append(f"Total training hours: **{total_h:.3f}**")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_json(rows: List[Dict[str, str]], out_path: Path) -> None:
    payload = {
        "experiments": rows,
        "total_train_hours": sum(to_float(r, "train_hours") for r in rows),
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize work2 summary results")
    parser.add_argument("--summary", type=str, default="./work2/results/summary.csv")
    parser.add_argument("--out-dir", type=str, default="./work2/results/figures")
    args = parser.parse_args()

    summary_path = Path(args.summary)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    rows = read_summary_csv(summary_path)
    if not rows:
        raise RuntimeError("Summary CSV is empty.")

    plot_metric_bars(rows, out_dir)
    plot_mode_pairwise(rows, out_dir)
    generate_table_md(rows, out_dir / "result_table.md")
    export_json(rows, out_dir / "summary_export.json")

    print("Visualization files generated:")
    print(f"- {out_dir / 'compare_acc_f1.png'}")
    print(f"- {out_dir / 'compare_train_time.png'}")
    print(f"- {out_dir / 'pair_resnext50_32x4d.png'} (if both modes exist)")
    print(f"- {out_dir / 'pair_densenet121.png'} (if both modes exist)")
    print(f"- {out_dir / 'result_table.md'}")
    print(f"- {out_dir / 'summary_export.json'}")


if __name__ == "__main__":
    main()
