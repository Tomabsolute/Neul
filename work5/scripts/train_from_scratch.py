from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None


class CharTokenizer:
    def __init__(self, text: str) -> None:
        chars = sorted(set(text))
        self.itos = ["<pad>", "<unk>"] + chars
        self.stoi = {ch: i for i, ch in enumerate(self.itos)}

    def encode(self, text: str) -> list[int]:
        unk = self.stoi["<unk>"]
        return [self.stoi.get(ch, unk) for ch in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids if i >= 2)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({"itos": self.itos}, ensure_ascii=False, indent=2), encoding="utf-8")


class TinyDecoderLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        n_layer: int = 6,
        n_head: int = 8,
        n_embd: int = 384,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        layer = nn.TransformerEncoderLayer(
            d_model=n_embd,
            nhead=n_head,
            dim_feedforward=4 * n_embd,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=n_layer)
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        batch, seq_len = idx.shape
        pos = torch.arange(seq_len, device=idx.device)
        x = self.token_embedding(idx) + self.position_embedding(pos)[None, :, :]
        mask = torch.triu(torch.ones(seq_len, seq_len, device=idx.device), diagonal=1).bool()
        x = self.blocks(x, mask=mask)
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_batch(data: torch.Tensor, batch_size: int, block_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix]).to(device)
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix]).to(device)
    return x, y


@torch.no_grad()
def estimate_loss(model: nn.Module, data: torch.Tensor, batch_size: int, block_size: int, device: torch.device, eval_iters: int) -> float:
    model.eval()
    losses = []
    for _ in range(eval_iters):
        x, y = get_batch(data, batch_size, block_size, device)
        _, loss = model(x, y)
        losses.append(float(loss.item()))
    model.train()
    return sum(losses) / len(losses)


@torch.no_grad()
def generate(model: nn.Module, tokenizer: CharTokenizer, prompt: str, device: torch.device, max_new_tokens: int) -> str:
    model.eval()
    idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.block_size :]
        logits, _ = model(idx_cond)
        next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        idx = torch.cat([idx, next_id], dim=1)
    return tokenizer.decode(idx[0].tolist())


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a tiny decoder-only LM from scratch on 孙子算经.")
    parser.add_argument("--train-file", default="work5/data/pretrain.txt")
    parser.add_argument("--output-dir", default="work5/results/scratch")
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--n-layer", type=int, default=6)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--n-embd", type=int, default=384)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    text = Path(args.train_file).read_text(encoding="utf-8")
    tokenizer = CharTokenizer(text)
    ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    split = int(len(ids) * 0.9)
    train_data, val_data = ids[:split], ids[split:]

    device = choose_device(args.device)
    model = TinyDecoderLM(
        len(tokenizer.itos),
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    history = []
    for step in range(1, args.max_steps + 1):
        x, y = get_batch(train_data, args.batch_size, args.block_size, device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step == 1 or step % args.eval_interval == 0 or step == args.max_steps:
            train_loss = estimate_loss(model, train_data, args.batch_size, args.block_size, device, args.eval_iters)
            val_loss = estimate_loss(model, val_data, args.batch_size, args.block_size, device, args.eval_iters)
            row = {"step": step, "train_loss": train_loss, "val_loss": val_loss, "val_ppl": math.exp(min(20.0, val_loss))}
            history.append(row)
            print(json.dumps(row, ensure_ascii=False))

    torch.save({"model_state": model.state_dict(), "args": vars(args), "vocab_size": len(tokenizer.itos)}, output_dir / "scratch_lm.pt")
    tokenizer.save(output_dir / "char_vocab.json")
    (output_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")

    sample = generate(model, tokenizer, "今有物不知其數，三三數之剩二，五五數之剩三，七七數之剩二。問物幾何？答曰：", device, max_new_tokens=40)
    (output_dir / "sample_generation.txt").write_text(sample, encoding="utf-8")

    if history and plt is not None:
        plt.figure(figsize=(6, 4))
        plt.plot([r["step"] for r in history], [r["train_loss"] for r in history], label="train")
        plt.plot([r["step"] for r in history], [r["val_loss"] for r in history], label="val")
        plt.xlabel("step")
        plt.ylabel("loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "curves.png", dpi=160)
        plt.close()

    metrics = {
        "model": "TinyDecoderLM-from-scratch",
        "vocab_size": len(tokenizer.itos),
        "num_chars": len(text),
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "block_size": args.block_size,
        "n_layer": args.n_layer,
        "n_head": args.n_head,
        "n_embd": args.n_embd,
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "final_val_loss": history[-1]["val_loss"] if history else None,
        "final_val_ppl": history[-1]["val_ppl"] if history else None,
        "device": str(device),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
