from __future__ import annotations

import argparse
import json
import random
import re
import urllib.request
from pathlib import Path


DEFAULT_SOURCE_URL = "https://www.gutenberg.org/ebooks/24038.txt.utf-8"


def ensure_input_file(path: Path, source_url: str) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading source text from {source_url}")
    with urllib.request.urlopen(source_url, timeout=60) as response:
        data = response.read()
    path.write_bytes(data)


def read_text(path: str) -> str:
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
    text = text.replace("问", "問").replace("为", "為")
    text = re.sub(r"\*\*\* START OF (THE|THIS) PROJECT GUTENBERG.*?\*\*\*", "", text, flags=re.S)
    text = re.sub(r"\*\*\* END OF (THE|THIS) PROJECT GUTENBERG.*", "", text, flags=re.S)
    text = re.sub(r"(?is)START: FULL LICENSE.*", "", text)
    start = text.find("《孫子算經》")
    if start >= 0:
        text = text[start:]
    text = re.sub(r"[ \t]+", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_qa_pairs(text: str) -> list[dict[str, str]]:
    marker = re.compile(r"(?=〔[一二三四五六七八九十百千万萬〇零兩两]+〕|(?<![一-龥])今有)")
    blocks = [block.strip() for block in marker.split(text) if block.strip()]
    pairs: list[dict[str, str]] = []
    for block in blocks:
        if "答曰" not in block:
            continue
        prompt, answer = block.split("答曰", 1)
        prompt = re.sub(r"\n+", "", prompt).strip()
        answer = answer.lstrip("：:").strip()
        answer = re.split(r"\n術曰|\n其術曰|\n又術曰", answer, maxsplit=1)[0]
        answer = re.sub(r"\n+", "", answer).strip()
        if len(prompt) >= 8 and len(answer) >= 2:
            pairs.append({"prompt": prompt + "答曰：", "answer": answer})
    return pairs


def corrupt_answer(answer: str) -> str:
    replacements = [
        ("一", "二"),
        ("二", "三"),
        ("三", "四"),
        ("十", "九"),
        ("百", "十"),
        ("畝", "步"),
        ("尺", "寸"),
        ("斗", "升"),
        ("升", "斗"),
        ("分", "尺"),
    ]
    for src, dst in replacements:
        if src in answer:
            return answer.replace(src, dst, 1)
    if len(answer) > 4:
        return answer[: max(1, len(answer) // 2)] + "。"
    return "不合題意。"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare 孙子算经 datasets for Work5.")
    parser.add_argument("--input", default="work5/data/sunzi_suanjing.txt")
    parser.add_argument("--source-url", default=DEFAULT_SOURCE_URL)
    parser.add_argument("--output-dir", default="work5/data")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ensure_input_file(Path(args.input), args.source_url)
    text = normalize_text(read_text(args.input))
    pairs = extract_qa_pairs(text)
    random.shuffle(pairs)

    (output_dir / "pretrain.txt").write_text(text, encoding="utf-8")
    write_jsonl(output_dir / "pretrain.jsonl", [{"text": text}])

    sft_rows = []
    dpo_rows = []
    for pair in pairs:
        question = pair["prompt"].removesuffix("答曰：")
        answer = pair["answer"]
        sft_rows.append(
            {
                "messages": [
                    {
                        "role": "system",
                        "content": "你是熟悉《孙子算经》的古文数学助手。请按原书风格用简洁答案作答。",
                    },
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer},
                ]
            }
        )
        dpo_rows.append(
            {
                "prompt": question,
                "chosen": answer,
                "rejected": corrupt_answer(answer),
            }
        )

    split = max(1, int(len(sft_rows) * 0.9))
    write_jsonl(output_dir / "sft.jsonl", sft_rows[:split])
    write_jsonl(output_dir / "sft_val.jsonl", sft_rows[split:])
    write_jsonl(output_dir / "dpo.jsonl", dpo_rows[:split])
    write_jsonl(output_dir / "dpo_val.jsonl", dpo_rows[split:])

    metrics = {
        "source": args.input,
        "num_pairs": len(pairs),
        "num_sft_train": split,
        "num_sft_val": len(sft_rows) - split,
        "num_dpo_train": split,
        "num_dpo_val": len(dpo_rows) - split,
        "num_chars": len(text),
    }
    (output_dir / "dataset_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
