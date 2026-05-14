from __future__ import annotations

import unsloth

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from unsloth import FastLanguageModel


DEFAULT_PROMPTS = [
    {
        "name": "物不知数",
        "prompt": "今有物不知其數，三三數之剩二，五五數之剩三，七七數之剩二。問物幾何？",
        "gold": "二十三。",
    },
    {
        "name": "鸡兔同笼",
        "prompt": "今有雉、兔同籠，上有三十五頭，下九十四足。問雉、兔各幾何？",
        "gold": "雉二十三。兔一十二。",
    },
    {
        "name": "三人共车",
        "prompt": "今有三人共車，二車空；二人共車，九人步。問人與車各幾何？",
        "gold": "一十五車。三十九人。",
    },
    {
        "name": "佛书字数",
        "prompt": "今有佛書凡二十九章，章六十三字。問字幾何？",
        "gold": "一千八百二十七。",
    },
    {
        "name": "长安洛阳",
        "prompt": "今有長安、洛陽相去九百里。車輪一匝一丈八尺。欲自洛陽至長安，問輪匝幾何？",
        "gold": "九萬匝。",
    },
]


def load_model(model_name: str, adapter: str, max_seq_length: int):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    model = PeftModel.from_pretrained(model, adapter)
    FastLanguageModel.for_inference(model)
    return model, tokenizer


@torch.no_grad()
def generate(model, tokenizer, prompt: str, max_new_tokens: int, repetition_penalty: float) -> str:
    messages = [
        {"role": "system", "content": "你是熟悉《孙子算经》的古文数学助手。请按原书风格用简洁答案作答。"},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=6,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    new_tokens = outputs[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def strip_think(text: str) -> str:
    return text.replace("<think>", "").replace("</think>", "").strip()


def write_markdown(path: Path, rows: list[dict[str, str]]) -> None:
    lines = [
        "| 题型 | 标准答案 | SFT 输出 | DPO 输出 |",
        "|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['name']} | {row['gold']} | {row['sft_output']} | {row['dpo_output']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare SFT and DPO adapters on fixed 孙子算经 prompts.")
    parser.add_argument("--model-name", default="unsloth/Qwen3-0.6B-unsloth-bnb-4bit")
    parser.add_argument("--sft-adapter", default="work5/results/qwen3_sft")
    parser.add_argument("--dpo-adapter", default="work5/results/qwen3_dpo")
    parser.add_argument("--output-dir", default="work5/results/eval")
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--repetition-penalty", type=float, default=1.15)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sft_model, sft_tokenizer = load_model(args.model_name, args.sft_adapter, args.max_seq_length)
    rows: list[dict[str, str]] = []
    for item in DEFAULT_PROMPTS:
        rows.append(
            {
                "name": item["name"],
                "prompt": item["prompt"],
                "gold": item["gold"],
                "sft_output": strip_think(generate(sft_model, sft_tokenizer, item["prompt"], args.max_new_tokens, args.repetition_penalty)),
            }
        )
    del sft_model
    torch.cuda.empty_cache()

    dpo_model, dpo_tokenizer = load_model(args.model_name, args.dpo_adapter, args.max_seq_length)
    for row in rows:
        row["dpo_output"] = strip_think(generate(dpo_model, dpo_tokenizer, row["prompt"], args.max_new_tokens, args.repetition_penalty))

    (output_dir / "comparison.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(output_dir / "comparison.md", rows)
    print((output_dir / "comparison.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
