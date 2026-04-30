# Work1：手写 BP 神经网络分类 Fashion-MNIST

本目录是课程第一次作业，实现了一个不依赖 PyTorch 训练框架的多层前馈神经网络，用反向传播在 Fashion-MNIST 小样本上做 10 类分类实验。

核心代码在 `ann.py`，支持：

- 激活函数：`sigmoid`、`tanh`、`RELU`、`RELU3`
- 优化方法：`GD`、`SGD`、`ADAM`
- CPU：NumPy
- GPU：如果安装了 CuPy，会自动使用 CuPy
- 输出训练精度曲线图

## 文件说明

| 文件 | 说明 |
|---|---|
| `ann.py` | 手写神经网络、前向传播、反向传播、GD/SGD/ADAM 训练代码 |
| `download.py` | 使用 `torchvision` 下载 Fashion-MNIST |
| `work1.tex` | LaTeX 报告源码 |
| `work1.pdf` | 已编译报告 |
| `figures/*.png` | 不同激活函数 / 优化器配置下的训练曲线 |
| `figures/计算图.jpg` | 报告中使用的计算图 |

## 环境

基础运行需要：

```bash
pip install numpy matplotlib
```

如果要使用 `download.py` 下载数据，需要额外安装：

```bash
pip install torch torchvision
```

如果服务器支持 CUDA，并希望使用 GPU，可安装与 CUDA 版本匹配的 CuPy，例如：

```bash
pip install cupy-cuda12x
```

没有 CuPy 时，`ann.py` 会自动回退到 NumPy CPU。

## 数据下载

运行：

```bash
python3 work1/download.py
```

注意：当前 `ann.py` 默认读取的数据路径是：

```text
./data/FashionMNIST/raw
```

如果用 `download.py` 下载后目录不一致，可以把数据移动到上述路径，或者修改 `ann.py` 中的：

```python
data_root = './data/FashionMNIST/raw'
```

建议从仓库根目录运行脚本。

## 运行实验

默认运行：

```bash
python3 work1/ann.py
```

当前 `ann.py` 主程序默认配置为：

```python
num = 500
test_num = 100
layers = np.array([784, 128, 64, 10])
Net = NeuralNetwork(layers, 'sigmoid', 'GD', use_gpu=True)
Net.train(X, Y, 1, 0.1, 10000)
```

含义：

- 使用 500 张训练图片
- 使用 100 张测试图片
- 网络结构为 `784 -> 128 -> 64 -> 10`
- 激活函数为 `sigmoid`
- 优化器为 `GD`
- 学习率为 `0.1`
- 训练 `10000` 轮

训练结束后会输出：

- 每轮训练误差 `perf`
- 每轮训练集精度 `precision`
- 测试集正确数量与测试精度
- 精度曲线图，例如 `figures/sigmoid_GD.png`

## 修改实验配置

可以在 `ann.py` 底部主程序中修改以下内容：

### 更换激活函数

```python
Net = NeuralNetwork(layers, 'RELU', 'SGD', use_gpu=True)
```

可选值：

```text
sigmoid, tanh, RELU, RELU3
```

### 更换优化器

```python
Net = NeuralNetwork(layers, 'sigmoid', 'ADAM', use_gpu=True)
```

可选值：

```text
GD, SGD, ADAM
```

### 调整网络结构

```python
layers = np.array([784, 128, 64, 10])
```

例如改成一层隐藏层：

```python
layers = np.array([784, 128, 10])
```

### 调整训练轮数和学习率

```python
Net.train(X, Y, 1, 0.1, 10000)
```

其中 `0.1` 是学习率，`10000` 是 epoch 数。

## 已保存结果

`figures/` 目录中已有多组曲线图，例如：

- `figures/sigmoid_GD.png`
- `figures/sigmoid_SGD.png`
- `figures/sigmoid_ADAM_0.001.png`
- `figures/sigmoid_ADAM_0.01.png`
- `figures/sigmoid_ADAM_0.1.png`
- `figures/tanh_SGD.png`
- `figures/RELU_SGD.png`
- `figures/RELU3_SGD.png`

这些图可用于报告中对比不同激活函数、优化器和学习率对收敛速度与最终精度的影响。

## 编译报告

```bash
cd work1
xelatex work1.tex
```

生成：

```text
work1.pdf
```
