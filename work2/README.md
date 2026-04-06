## 一键跑4组

```bash
bash work2/run_all_experiments.sh
```

可选环境变量：

```bash
PYTHON=python3 BATCH_SIZE=64 NUM_WORKERS=8 AMP=1 MULTI_GPU=1 bash work2/run_all_experiments.sh
```

## 单组命令示例

```bash
python3 work2/train_food101.py \
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
- `curves.png`
- `metrics.json`
- `best_model.pt`

总表：

- `work2/results/summary.csv`

四组对比图：

- `work2/results/figures/compare_acc_f1.png`
- `work2/results/figures/compare_train_time.png`
- `work2/results/figures/pair_resnext50_32x4d.png`
- `work2/results/figures/pair_densenet121.png`
