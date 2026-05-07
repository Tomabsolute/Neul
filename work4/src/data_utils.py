from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


PAD = "<pad>"
UNK = "<unk>"
BOS = "<bos>"
EOS = "<eos>"


def read_text(path: str | Path) -> str:
    data = Path(path).read_bytes()
    for encoding in ("utf-8", "gb2312", "gb18030", "gbk"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="ignore")


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("荅曰", "答曰")
    text = text.replace("\u3000", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_train_val(items: list, val_ratio: float = 0.15, seed: int = 42) -> tuple[list, list]:
    import random

    rng = random.Random(seed)
    shuffled = list(items)
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_ratio)) if len(shuffled) > 1 else 0
    return shuffled[n_val:], shuffled[:n_val]


def extract_qa_pairs(text: str) -> list[dict[str, str]]:
    text = normalize_text(text)
    block_re = re.compile(r"(〔[一二三四五六七八九十百零〇]+〕.*?)(?=\n〔[一二三四五六七八九十百零〇]+〕|\Z)", re.S)
    pairs: list[dict[str, str]] = []
    for match in block_re.finditer(text):
        block = match.group(1).strip()
        answer_match = re.search(r"答曰[:：]\s*([^\n。；;]+[。]?)", block)
        if not answer_match:
            continue
        answer = answer_match.group(1).strip()
        prompt = block[: answer_match.start()].strip()
        prompt = re.sub(r"\s+", "", prompt)
        answer = re.sub(r"\s+", "", answer)
        if len(prompt) < 6 or len(answer) < 1:
            continue
        pairs.append({"prompt": prompt + "答曰：", "answer": answer, "block": block})
    return pairs


@dataclass
class Vocab:
    stoi: dict[str, int]
    itos: list[str]

    @classmethod
    def build(cls, sequences: Iterable[Iterable[str]], min_freq: int = 1, specials: list[str] | None = None) -> "Vocab":
        specials = specials or [PAD, UNK, BOS, EOS]
        counter: Counter[str] = Counter()
        for seq in sequences:
            counter.update(seq)
        itos = list(specials)
        for token, count in counter.most_common():
            if count >= min_freq and token not in specials:
                itos.append(token)
        return cls({token: i for i, token in enumerate(itos)}, itos)

    def encode(self, tokens: Iterable[str], add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = []
        if add_bos:
            ids.append(self.stoi[BOS])
        ids.extend(self.stoi.get(token, self.stoi[UNK]) for token in tokens)
        if add_eos:
            ids.append(self.stoi[EOS])
        return ids

    def decode(self, ids: Iterable[int], skip_special: bool = True) -> str:
        specials = {PAD, UNK, BOS, EOS}
        tokens = []
        for idx in ids:
            if idx < 0 or idx >= len(self.itos):
                continue
            token = self.itos[idx]
            if skip_special and token in specials:
                continue
            tokens.append(token)
        return "".join(tokens)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({"itos": self.itos}, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Vocab":
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
        itos = obj["itos"]
        return cls({token: i for i, token in enumerate(itos)}, itos)


def char_tokens(text: str) -> list[str]:
    return [ch for ch in text if not ch.isspace()]
