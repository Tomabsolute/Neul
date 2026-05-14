from __future__ import annotations

import unsloth

import argparse

import torch
from peft import PeftModel
from unsloth import FastLanguageModel


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference with Qwen3 adapter.")
    parser.add_argument("--model-name", default="unsloth/Qwen3-0.6B-unsloth-bnb-4bit")
    parser.add_argument("--adapter", default="work5/results/qwen3_dpo")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.15)
    args = parser.parse_args()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    model = PeftModel.from_pretrained(model, args.adapter)
    FastLanguageModel.for_inference(model)

    messages = [
        {"role": "system", "content": "你是熟悉《孙子算经》的古文数学助手。请按原书风格用简洁答案作答。"},
        {"role": "user", "content": args.prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature if args.temperature > 0 else None,
            do_sample=args.temperature > 0,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=6,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    new_tokens = outputs[0, inputs["input_ids"].shape[1] :]
    print(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())


if __name__ == "__main__":
    main()
