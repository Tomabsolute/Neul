from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.data_utils import Vocab, char_tokens
from src.models import CharLSTMModel
from train_lm import choose_device, generate


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an answer with a trained 九章算术 CharLSTM model.")
    parser.add_argument("--checkpoint", default="work4/results/lstm_lm/best_lstm_lm.pt")
    parser.add_argument("--vocab", default="work4/results/lstm_lm/vocab.json")
    parser.add_argument("--prompt", required=True, help="Problem text. The script appends 答曰： if it is missing.")
    parser.add_argument("--max-new-chars", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.0, help="Use 0 for greedy decoding.")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    vocab = Vocab.load(args.vocab)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = checkpoint["config"]
    model = CharLSTMModel(
        len(vocab.itos),
        embedding_dim=config["embedding_dim"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
    )
    model.load_state_dict(checkpoint["model_state"])
    device = choose_device(args.device)
    model.to(device)

    prompt = args.prompt.strip()
    if "答曰" not in prompt:
        prompt += "答曰："
    elif not prompt.endswith(("：", ":")):
        prompt += "："
    output = generate(model, vocab, prompt, device, max_new_chars=args.max_new_chars, temperature=args.temperature)
    print(prompt + output)


if __name__ == "__main__":
    main()
