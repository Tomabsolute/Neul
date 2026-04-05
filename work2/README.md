# Work2（精简版）

只保留你要用的配置：

1. 数据集：Food101（多分类）
2. 模型：ResNeXt50_32x4d、DenseNet121
3. 对比：from scratch vs fine-tune（共4组）

## 需要的文件

- `work2/train_flowers102.py`：主训练脚本（Food101专用）
- `work2/run_all_experiments.sh`：一键跑4组
- `work2/visualize_results.py`：汇总生成对比图
- `work2/report_template.md`：报告模板

## 依赖

```bash
pip install torch torchvision scipy matplotlib numpy
```

## 一键跑4组

```bash
bash work2/run_all_experiments.sh
```

可选环境变量（2x4090 推荐）：

```bash
PYTHON=python3 BATCH_SIZE=64 NUM_WORKERS=8 AMP=1 MULTI_GPU=1 bash work2/run_all_experiments.sh
```

## 单组命令示例

```bash
python3 work2/train_flowers102.py \
  --model resnext50_32x4d \
  --mode finetune \
  --batch-size 64 \
  --num-workers 8 \
  --amp \
  --multi-gpu
```

## 输出结果

每组目录：`work2/results/food101_<model>_<mode>/`

- `history.csv`
- `curves.png`（训练结束后保存）
- `metrics.json`
- `best_model.pt`

总表：

- `work2/results/summary.csv`

四组对比图：

- `work2/results/figures/compare_acc_f1.png`
- `work2/results/figures/compare_train_time.png`
- `work2/results/figures/pair_resnext50_32x4d.png`
- `work2/results/figures/pair_densenet121.png`
