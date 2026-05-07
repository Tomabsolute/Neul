from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from src.data_utils import EOS, Vocab, char_tokens, extract_qa_pairs, normalize_text, read_text, split_train_val
from src.models import CharLSTMModel
from src.plotting import plot_lm_history


class QAPairDataset(Dataset):
    def __init__(self, pairs: list[dict[str, str]], vocab: Vocab, max_len: int) -> None:
        self.samples = []
        self.vocab = vocab
        self.max_len = max_len
        bos_id = vocab.stoi["<bos>"]
        eos_id = vocab.stoi[EOS]
        for pair in pairs:
            prompt_tokens = char_tokens(pair["prompt"])
            answer_tokens = char_tokens(pair["answer"])
            ids = [bos_id] + vocab.encode(prompt_tokens + answer_tokens) + [eos_id]
            prompt_boundary = 1 + len(prompt_tokens)
            if len(ids) > max_len + 1:
                dropped = len(ids) - (max_len + 1)
                ids = ids[dropped:]
                prompt_boundary = max(0, prompt_boundary - dropped)
            self.samples.append((ids, prompt_boundary))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        ids, prompt_boundary = self.samples[idx]
        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor(ids[1:], dtype=torch.long)
        positions = torch.arange(1, len(ids), dtype=torch.long)
        y = y.masked_fill(positions < prompt_boundary, -100)
        return x, y


def collate_qa_batch(batch: list[tuple[torch.Tensor, torch.Tensor]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    xs, ys = zip(*batch)
    x = pad_sequence(xs, batch_first=True, padding_value=pad_id)
    y = pad_sequence(ys, batch_first=True, padding_value=-100)
    return x, y


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate_loss(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits, _ = model(x)
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        valid_tokens = (y != -100).sum().item()
        total_loss += loss.item() * valid_tokens
        total_tokens += valid_tokens
    return total_loss / max(1, total_tokens)


@torch.no_grad()
def generate(
    model: CharLSTMModel,
    vocab: Vocab,
    prompt: str,
    device: torch.device,
    max_new_chars: int = 40,
    temperature: float = 0.8,
) -> str:
    model.eval()
    ids = vocab.encode(char_tokens(prompt))
    if not ids:
        return ""
    x = torch.tensor([ids], dtype=torch.long, device=device)
    logits, hidden = model(x)
    current = x[:, -1:]
    generated: list[int] = []
    for _ in range(max_new_chars):
        logits, hidden = model(current, hidden)
        next_logits = logits[:, -1, :] / max(temperature, 1e-4)
        probs = torch.softmax(next_logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        token = vocab.itos[next_id.item()]
        if token == EOS:
            break
        generated.append(next_id.item())
        current = next_id
    return vocab.decode(generated)


def answer_hit(pred: str, gold: str) -> bool:
    pred = pred.split(EOS, 1)[0]
    gold_chars = [ch for ch in gold.replace("。", "") if ch not in "，,；; "]
    if not gold_chars:
        return False
    key_chars = [ch for ch in gold_chars if ch in "一二三四五六七八九十百千萬万億亿分步畝亩頃顷尺寸升斗斛錢钱兩两銖铢"]
    if len(key_chars) < 2:
        key_chars = gold_chars[: min(4, len(gold_chars))]
    matched = sum(1 for ch in key_chars if ch in pred)
    return matched / max(1, len(set(key_chars))) >= 0.5


def write_samples(path: Path, samples: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for i, row in enumerate(samples, 1):
            f.write(f"## 样例 {i}\n")
            f.write(f"Prompt: {row['prompt']}\n")
            f.write(f"Gold: {row['gold']}\n")
            f.write(f"Generated: {row['generated']}\n")
            f.write(f"Hit: {row['hit']}\n\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a character-level LSTM language model on 九章算术.")
    parser.add_argument("--data", default="work4/九章算经.txt")
    parser.add_argument("--output-dir", default="work4/results/lstm_lm")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    text = normalize_text(read_text(args.data))
    qa_pairs = extract_qa_pairs(text)
    if len(qa_pairs) < 10:
        raise RuntimeError("Too few QA pairs were extracted. Check the input text format.")

    train_pairs, val_pairs = split_train_val(qa_pairs, val_ratio=0.15, seed=args.seed)
    vocab_sequences = []
    for pair in qa_pairs:
        vocab_sequences.append(char_tokens(pair["prompt"]) + char_tokens(pair["answer"]) + [EOS])
    vocab = Vocab.build(vocab_sequences, min_freq=1)
    vocab.save(output_dir / "vocab.json")
    train_dataset = QAPairDataset(train_pairs, vocab, args.seq_len)
    val_dataset = QAPairDataset(val_pairs, vocab, args.seq_len)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        collate_fn=lambda batch: collate_qa_batch(batch, vocab.stoi["<pad>"]),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_qa_batch(batch, vocab.stoi["<pad>"]),
    )

    device = choose_device(args.device)
    model = CharLSTMModel(
        len(vocab.itos),
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    history: list[dict] = []
    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_tokens = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(x)
            loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            valid_tokens = (y != -100).sum().item()
            total_loss += loss.item() * valid_tokens
            total_tokens += valid_tokens

        train_loss = total_loss / max(1, total_tokens)
        val_loss = evaluate_loss(model, val_loader, criterion, device)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_ppl": math.exp(min(20.0, val_loss)),
        }
        history.append(row)
        print(f"epoch {epoch:03d} train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_ppl={row['val_ppl']:.2f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": vars(args),
                    "vocab_size": len(vocab.itos),
                },
                output_dir / "best_lstm_lm.pt",
            )

    with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "val_ppl"])
        writer.writeheader()
        writer.writerows(history)
    plot_lm_history(history, output_dir / "curves.png")

    checkpoint = torch.load(output_dir / "best_lstm_lm.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    samples = []
    hits = 0
    for pair in val_pairs[: min(30, len(val_pairs))]:
        gen = generate(model, vocab, pair["prompt"], device=device, max_new_chars=50, temperature=0.75)
        hit = answer_hit(gen, pair["answer"])
        hits += int(hit)
        samples.append({"prompt": pair["prompt"], "gold": pair["answer"], "generated": gen, "hit": str(hit)})
    write_samples(output_dir / "generation_samples.md", samples)

    metrics = {
        "model": "CharLSTM",
        "num_qa_pairs": len(qa_pairs),
        "num_train_pairs": len(train_pairs),
        "num_val_pairs": len(val_pairs),
        "vocab_size": len(vocab.itos),
        "best_val_loss": best_val,
        "best_val_ppl": math.exp(min(20.0, best_val)),
        "answer_keyword_hit_rate": hits / max(1, len(samples)),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "embedding_dim": args.embedding_dim,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "device": str(device),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
