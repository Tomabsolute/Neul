from __future__ import annotations

import argparse

import torch

from src.data_utils import Vocab
from src.models import SkipGramNegSampling
from train_lm import choose_device
from train_word2vec import nearest_neighbors


def main() -> None:
    parser = argparse.ArgumentParser(description="Query nearest neighbors from trained 九章算术 Skip-gram embeddings.")
    parser.add_argument("--checkpoint", default="work4/results/word2vec/skipgram_embeddings.pt")
    parser.add_argument("--vocab", default="work4/results/word2vec/vocab.json")
    parser.add_argument("--queries", default="田,畝,分,率,粟,米,步,尺")
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    vocab = Vocab.load(args.vocab)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = checkpoint["config"]
    model = SkipGramNegSampling(len(vocab.itos), embedding_dim=config["embedding_dim"])
    model.load_state_dict(checkpoint["model_state"])
    device = choose_device(args.device)
    model.to(device)

    queries = [item.strip() for item in args.queries.split(",") if item.strip()]
    neighbors = nearest_neighbors(model, vocab, queries, topk=args.topk)
    for query in queries:
        print(f"## {query}")
        rows = neighbors.get(query, [])
        if not rows:
            print("- not in vocabulary")
        for token, score in rows:
            print(f"- {token}: {score:.4f}")
        print()


if __name__ == "__main__":
    main()

