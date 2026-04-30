# 基于条件 GAN 的小样本服饰图像生成

本项目实现一个轻量级 Conditional DCGAN，用类别标签控制 Fashion-MNIST 服饰图像生成。任务目标是课程作业中的“基于 GAN 的生成模型挑战”：小样本、合适训练量、代码可获得、包含 train 脚本、模型可加载并运行推理。

参考 notebook 位于 `work3/demo-gan/`，本项目将其中 MNIST GAN / CGAN 的思路整理为可复现实验代码。

## 环境

```bash
pip install -r work3/requirements.txt
```

## 一键训练与推理

```bash
bash work3/run_all_experiments.sh
```

可选环境变量：

```bash
PYTHON=python3 EPOCHS=20 BATCH_SIZE=128 NUM_SAMPLES=6000 NUM_WORKERS=4 bash work3/run_all_experiments.sh
```

## 单独训练

```bash
python3 work3/train_cgan.py \
  --dataset fashion-mnist \
  --download \
  --epochs 20 \
  --batch-size 128 \
  --num-samples 6000 \
  --num-workers 4
```

主要输出：

- `work3/results/cgan_fashion_mnist/checkpoints/best_generator.pt`
- `work3/results/cgan_fashion_mnist/checkpoints/last_checkpoint.pt`
- `work3/results/cgan_fashion_mnist/history.csv`
- `work3/results/cgan_fashion_mnist/curves.png`
- `work3/results/cgan_fashion_mnist/samples/epoch_*.png`

## 推理生成

```bash
python3 work3/infer.py \
  --checkpoint work3/results/cgan_fashion_mnist/checkpoints/best_generator.pt \
  --labels 0,1,2,3,4,5,6,7,8,9 \
  --repeat 8 \
  --output work3/results/figures/inference_grid.png
```

Fashion-MNIST 标签含义：

| 标签 | 类别 |
|---:|---|
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

## 模型下载

训练完成后，将 `work3/results/cgan_fashion_mnist/checkpoints/best_generator.pt` 上传到 Gitee Release、GitHub Release 或网盘直链即可。下载脚本如下：

```bash
python3 work3/download_model.py \
  --url "https://your-release-url/best_generator.pt" \
  --output work3/results/cgan_fashion_mnist/checkpoints/best_generator.pt
```

下载后直接运行上面的推理命令。

## 报告

LaTeX 报告文件：

- `work3/report.tex`

编译示例：

```bash
cd work3
xelatex report.tex
```
