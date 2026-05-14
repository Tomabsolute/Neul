from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from peft import PeftModel
from trl import DPOConfig, DPOTrainer
from unsloth import FastLanguageModel, PatchDPOTrainer, is_bfloat16_supported


def build_prompt(tokenizer, question: str) -> str:
    messages = [
        {"role": "system", "content": "你是熟悉《孙子算经》的古文数学助手。请按原书风格用简洁答案作答。"},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="DPO post-training for Qwen3-0.6B SFT adapter.")
    parser.add_argument("--model-name", default="unsloth/Qwen3-0.6B-unsloth-bnb-4bit")
    parser.add_argument("--data", default="work5/data/dpo.jsonl")
    parser.add_argument("--sft-adapter", default="work5/results/qwen3_sft")
    parser.add_argument("--output-dir", default="work5/results/qwen3_dpo")
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    PatchDPOTrainer()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    model = PeftModel.from_pretrained(model, args.sft_adapter, is_trainable=True)

    raw_dataset = load_dataset("json", data_files=args.data, split="train")

    def format_dpo(row):
        return {
            "prompt": build_prompt(tokenizer, row["prompt"]),
            "chosen": row["chosen"],
            "rejected": row["rejected"],
        }

    dataset = raw_dataset.map(format_dpo, remove_columns=raw_dataset.column_names)

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=DPOConfig(
            output_dir=str(output_dir),
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            warmup_steps=10,
            max_steps=args.max_steps,
            learning_rate=args.lr,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=10,
            optim="adamw_8bit",
            lr_scheduler_type="cosine",
            beta=args.beta,
            seed=args.seed,
            report_to="none",
            save_strategy="steps",
            save_steps=max(50, args.max_steps),
            max_length=args.max_seq_length,
            max_prompt_length=args.max_seq_length // 2,
        ),
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    stats = trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    metrics = {
        "stage": "dpo",
        "framework": "Unsloth",
        "base_model": args.model_name,
        "sft_adapter": args.sft_adapter,
        "dataset": args.data,
        "num_examples": len(dataset),
        "max_seq_length": args.max_seq_length,
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "learning_rate": args.lr,
        "beta": args.beta,
        "train_runtime": stats.metrics.get("train_runtime"),
        "train_loss": stats.metrics.get("train_loss"),
        "cuda": torch.cuda.is_available(),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
