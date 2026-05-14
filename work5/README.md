# Work5: Unsloth 框架下的《孙子算经》领域语言模型训练

本作业选择 `Unsloth` 作为主要框架，数据集使用 Project Gutenberg 的《孙子算经》UTF-8 文本，默认下载到 `work5/data/sunzi_suanjing.txt`。目标是让模型学习《孙子算经》中“题目 -> 答曰”形式的古文数学问答。

实验包含三类算例：

- 从头训练：随机初始化小型 Decoder-only Transformer。
- Qwen3 微调：基于 `unsloth/Qwen3-0.6B-unsloth-bnb-4bit` 做 LoRA/QLoRA SFT。
- 后训练：基于 SFT adapter 做短步数 DPO，偏好原文答案而非扰动答案。

## 目录

```text
work5/
├── README.md
├── requirements.txt
├── requirements-no-unsloth.txt
├── run_all_experiments.sh
├── report.tex
├── scripts/
│   ├── prepare_sunzi_dataset.py
│   ├── train_from_scratch.py
│   ├── train_qwen3_sft.py
│   ├── train_qwen3_dpo.py
│   ├── infer_qwen3.py
│   └── evaluate_adapters.py
├── data/
└── results/
```

## 环境安装

若环境中还没有 PyTorch：

```bash
pip install -r work5/requirements.txt
```

若环境中已有可用 CUDA PyTorch，避免重复下载 PyTorch：

```bash
python3 -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
pip install -r work5/requirements-no-unsloth.txt
pip install --no-deps unsloth unsloth_zoo
```

如需使用镜像源：

```bash
pip install -r work5/requirements-no-unsloth.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install --no-deps unsloth unsloth_zoo -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 一键运行

```bash
bash work5/run_all_experiments.sh
```

默认最终实验参数：

| 阶段 | 参数 |
|---|---|
| 从头训练 | `SCRATCH_STEPS=2000` |
| Qwen3 SFT | `SFT_STEPS=800` |
| DPO | `DPO_STEPS=50`, `DPO_LR=1e-5` |

运行完成后会生成：

- `work5/results/scratch/metrics.json`
- `work5/results/qwen3_sft/metrics.json`
- `work5/results/qwen3_dpo/metrics.json`
- `work5/results/qwen3_dpo/sample_generation.txt`
- `work5/results/eval/comparison.md`
- `work5/results/eval/comparison.json`

## 单独运行

准备数据：

```bash
python3 work5/scripts/prepare_sunzi_dataset.py --output-dir work5/data
```

从头训练：

```bash
python3 work5/scripts/train_from_scratch.py \
  --train-file work5/data/pretrain.txt \
  --output-dir work5/results/scratch \
  --max-steps 2000
```

Qwen3 0.6B SFT：

```bash
python3 work5/scripts/train_qwen3_sft.py \
  --data work5/data/sft.jsonl \
  --output-dir work5/results/qwen3_sft \
  --max-steps 800
```

DPO 后训练：

```bash
python3 work5/scripts/train_qwen3_dpo.py \
  --data work5/data/dpo.jsonl \
  --sft-adapter work5/results/qwen3_sft \
  --output-dir work5/results/qwen3_dpo \
  --max-steps 50 \
  --lr 1e-5
```

单题推理：

```bash
python3 work5/scripts/infer_qwen3.py \
  --adapter work5/results/qwen3_dpo \
  --prompt "今有物不知其數，三三數之剩二，五五數之剩三，七七數之剩二。問物幾何？"
```

批量对比 SFT 与 DPO：

```bash
python3 work5/scripts/evaluate_adapters.py \
  --sft-adapter work5/results/qwen3_sft \
  --dpo-adapter work5/results/qwen3_dpo \
  --output-dir work5/results/eval
```

## 当前实验结果

数据准备脚本抽取《孙子算经》题答对 65 条，其中 SFT/DPO 训练集 58 条、验证集 7 条。

已完成的主要结果：

| 阶段 | 指标 | 数值 |
|---|---|---:|
| 从头训练 | 最终验证 loss | 6.9084 |
| 从头训练 | 最终验证 PPL | 1000.64 |
| Qwen3 SFT | train loss | 0.1377 |
| Qwen3 DPO | step / lr | 50 / 1e-5 |
| 物不知数生成 | DPO 输出 | 二十三。 |

## 报告

LaTeX 报告文件为 `work5/report.tex`：

```bash
cd work5
xelatex report.tex
```
