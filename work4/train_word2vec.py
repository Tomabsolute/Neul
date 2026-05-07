from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from src.data_utils import Vocab, char_tokens, normalize_text, read_text
from src.models import SkipGramNegSampling
from src.plotting import plot_embedding_neighbors


class SkipGramDataset(Dataset):
    def __init__(self, token_ids: list[int], window_size: int) -> None:
        pairs = []
        for i, center in enumerate(token_ids):
            start = max(0, i - window_size)
            end = min(len(token_ids), i + window_size + 1)
            for j in range(start, end):
                if i != j:
                    pairs.append((center, token_ids[j]))
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[int, int]:
        return self.pairs[idx]


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_negative_distribution(ids: list[int], vocab_size: int) -> torch.Tensor:
    counts = torch.ones(vocab_size)
    counter = Counter(ids)
    for idx, count in counter.items():
        counts[idx] = float(count)
    return counts.pow(0.75) / counts.pow(0.75).sum()


@torch.no_grad()
def nearest_neighbors(model: SkipGramNegSampling, vocab: Vocab, queries: list[str], topk: int = 8) -> dict[str, list[tuple[str, float]]]:
    emb = model.in_embed.weight.detach().cpu()
    emb = torch.nn.functional.normalize(emb, dim=1)
    result = {}
    for query in queries:
        if query not in vocab.stoi:
            continue
        idx = vocab.stoi[query]
        scores = emb @ emb[idx]
        values, indices = torch.topk(scores, k=min(topk + 1, emb.size(0)))
        rows = []
        for score, token_id in zip(values.tolist(), indices.tolist()):
            token = vocab.itos[token_id]
            if token == query or token.startswith("<"):
                continue
            rows.append((token, float(score)))
            if len(rows) >= topk:
                break
        result[query] = rows
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Skip-gram word2vec-style embeddings on 九章算术.")
    parser.add_argument("--data", default="work4/九章算经.txt")
    parser.add_argument("--output-dir", default="work4/results/word2vec")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--window-size", type=int, default=4)
    parser.add_argument("--negative-samples", type=int, default=8)
    parser.add_argument("--min-freq", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    text = normalize_text(read_text(args.data))
    tokens = char_tokens(text)
    vocab = Vocab.build([tokens], min_freq=args.min_freq)
    vocab.save(output_dir / "vocab.json")
    ids = vocab.encode(tokens)
    dataset = SkipGramDataset(ids, args.window_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

    device = choose_device(args.device)
    model = SkipGramNegSampling(len(vocab.itos), embedding_dim=args.embedding_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    neg_dist = build_negative_distribution(ids, len(vocab.itos)).to(device)

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0
        for center, context in loader:
            center = center.to(device)
            context = context.to(device)
            neg = torch.multinomial(neg_dist, center.numel() * args.negative_samples, replacement=True)
            neg = neg.view(center.numel(), args.negative_samples)
            optimizer.zero_grad(set_to_none=True)
            loss = model(center, context, neg)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * center.numel()
            total += center.numel()
        avg_loss = total_loss / max(1, total)
        history.append({"epoch": epoch, "loss": avg_loss})
        print(f"epoch {epoch:03d} loss={avg_loss:.4f}")

    torch.save({"model_state": model.state_dict(), "config": vars(args), "vocab_size": len(vocab.itos)}, output_dir / "skipgram_embeddings.pt")
    with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "loss"])
        writer.writeheader()
        writer.writerows(history)

    queries = ["田", "畝", "分", "率", "方", "粟", "米", "術", "答", "問", "步", "尺"]
    neighbors = nearest_neighbors(model, vocab, queries)
    (output_dir / "neighbors.json").write_text(json.dumps(neighbors, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output_dir / "neighbors.md").open("w", encoding="utf-8") as f:
        for query, rows in neighbors.items():
            f.write(f"## {query}\n")
            for token, score in rows:
                f.write(f"- {token}: {score:.4f}\n")
            f.write("\n")
    plot_embedding_neighbors(neighbors, output_dir / "neighbors.png")

    metrics = {
        "model": "SkipGramNegativeSampling",
        "vocab_size": len(vocab.itos),
        "num_tokens": len(ids),
        "num_pairs": len(dataset),
        "final_loss": history[-1]["loss"],
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "embedding_dim": args.embedding_dim,
        "window_size": args.window_size,
        "negative_samples": args.negative_samples,
        "device": str(device),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

