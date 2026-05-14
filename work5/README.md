# Work5: Unsloth 框架下的《孙子算经》领域语言模型训练报告

本作业选择 `Unsloth` 作为主要框架，数据集使用 Project Gutenberg 的 `孫子算經` UTF-8 文本，默认下载到 `work5/data/sunzi_suanjing.txt`，目标是让模型学习《孙子算经》中“题目 -> 答曰”形式的古文数学问答。

包含三类算例：

- 从头训练：随机初始化一个小型 Decoder-only Transformer，验证数据清洗和自回归建模流程。
- Qwen3 微调：基于 `unsloth/Qwen3-0.6B-unsloth-bnb-4bit` 做 LoRA/QLoRA SFT。
- 后训练：基于 SFT adapter 做 DPO 偏好优化，使模型偏好包含古文答案格式、单位和计算结果的回答。

## 目录

```text
work5/
├── README.md
├── requirements.txt
├── run_all_experiments.sh
├── report.tex
├── scripts/
│   ├── prepare_sunzi_dataset.py
│   ├── train_from_scratch.py
│   ├── train_qwen3_sft.py
│   ├── train_qwen3_dpo.py
│   └── infer_qwen3.py
├── data/
└── results/
```

## 服务器运行

建议先在服务器上建独立环境：

```bash
cd /path/to/Neul
python3 -m venv .venv-work5
source .venv-work5/bin/activate
pip install -r work5/requirements.txt
```

一键小规模试跑：

```bash
bash work5/run_all_experiments.sh
```

默认参数偏保守，主要用于先跑通流程。正式训练可增加步数：

```bash
SCRATCH_STEPS=2000 SFT_STEPS=800 DPO_STEPS=400 \
  bash work5/run_all_experiments.sh
```

当前数据准备脚本已可抽取《孙子算经》题答对 65 条，其中 SFT/DPO 训练集 58 条、验证集 7 条。

## 单独运行

准备数据：

```bash
python3 work5/scripts/prepare_sunzi_dataset.py \
  --output-dir work5/data
```

从头训练小模型：

```bash
python3 work5/scripts/train_from_scratch.py \
  --train-file work5/data/pretrain.txt \
  --output-dir work5/results/scratch \
  --max-steps 1000
```

Qwen3 0.6B SFT：

```bash
python3 work5/scripts/train_qwen3_sft.py \
  --data work5/data/sft.jsonl \
  --output-dir work5/results/qwen3_sft \
  --max-steps 400
```

DPO 后训练：

```bash
python3 work5/scripts/train_qwen3_dpo.py \
  --data work5/data/dpo.jsonl \
  --sft-adapter work5/results/qwen3_sft \
  --output-dir work5/results/qwen3_dpo \
  --max-steps 200
```

推理：

```bash
python3 work5/scripts/infer_qwen3.py \
  --adapter work5/results/qwen3_dpo \
  --prompt "今有物不知其數，三三數之剩二，五五數之剩三，七七數之剩二。問物幾何？"
```

## 报告

LaTeX 报告文件为 `work5/report.tex`。服务器训练完成后，可把 `results/*/metrics.json` 中的实际数值填回报告的结果表。

```bash
cd work5
xelatex report.tex
```
