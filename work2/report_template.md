# 《人工神经网络》Work2 报告

题目：从头训练 V.S. 微调（以 ResNeXt 与 DenseNet 为例）

姓名：
学号：
日期：
Git 仓库地址：

## 1. 实验任务与要求

- 任务：比较从头训练（from scratch）与微调（fine-tune）在多分类任务上的差异。
- 模型：ResNeXt50_32x4d、DenseNet121。
- 数据集：（填写你使用的数据集，例如 Food101 / ImageFolder 自定义大数据集）

## 2. 数据集与预处理

- 数据集简介：
- 训练/验证/测试划分规模：
- 输入尺寸：224x224
- 数据增强：RandomResizedCrop、RandomHorizontalFlip
- 标准化：ImageNet mean/std

## 3. 实验设置

统一设置：

- 优化器：SGD（momentum=0.9, weight_decay=1e-4）
- 损失函数：CrossEntropyLoss
- 指标：Top-1 Accuracy、Macro-F1
- 设备：

四组实验：

1. ResNeXt50_32x4d - scratch
2. ResNeXt50_32x4d - finetune
3. DenseNet121 - scratch
4. DenseNet121 - finetune

微调策略：

- 阶段1：冻结 backbone，仅训练分类头
- 阶段2：解冻全网，backbone 与 head 使用不同学习率

## 4. 实验结果

### 4.1 总表

| 实验 | best val acc | test acc | test macro-F1 | train time (s) | train time (h) |
|---|---:|---:|---:|---:|---:|
| ResNeXt-scratch |  |  |  |  |  |
| ResNeXt-finetune |  |  |  |  |  |
| DenseNet-scratch |  |  |  |  |  |
| DenseNet-finetune |  |  |  |  |  |

训练总时长（4组累计）：`_____` 小时（需 >= 2 小时）

### 4.2 曲线图

- ResNeXt-scratch：loss/acc 曲线
- ResNeXt-finetune：loss/acc 曲线
- DenseNet-scratch：loss/acc 曲线
- DenseNet-finetune：loss/acc 曲线

## 5. 结果分析

建议围绕以下点展开：

1. 微调相比从头训练在收敛速度上的优势。
2. 微调相比从头训练在最终精度与泛化性能上的变化。
3. ResNeXt 与 DenseNet 在该数据集上的表现差异与可能原因。
4. 不同训练策略在训练时间成本上的权衡。

## 6. 结论

- 主要结论1：
- 主要结论2：
- 主要结论3：

## 7. 附：可复现实验命令（不粘贴代码）

```bash
bash work2/run_all_experiments.sh
```

以及单组命令示例：

```bash
python3 work2/train_flowers102.py --model resnext50_32x4d --mode scratch
python3 work2/train_flowers102.py --model resnext50_32x4d --mode finetune
python3 work2/train_flowers102.py --model densenet121 --mode scratch
python3 work2/train_flowers102.py --model densenet121 --mode finetune
```
