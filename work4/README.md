# 基于神经网络的《九章算术》语言模型与文本嵌入

本目录完成课程任务中的两类模型：

- 序列生成模型：字符级 LSTM 语言模型，学习题面到 `答曰：` 后答案文本的续写。
- 文本嵌入模型：Skip-gram Negative Sampling，形式上等价于轻量 word2vec，用《九章算术》字符上下文学习语义相近的数学词汇。

原始附件为 `work4/九章算经.txt`。该文件编码方式为 `GB 2312`，脚本会自动尝试 `utf-8` / `gb2312` / `gb18030` / `gbk` 读取，并把原文中的 `荅曰` 统一为 `答曰`。

## 一键运行

```bash
bash work4/run_all_experiments.sh
```

## 单独训练 LSTM 语言模型

```bash
python3 work4/train_lm.py \
  --data work4/九章算经.txt \
  --epochs 80 \
  --batch-size 256 \
  --seq-len 96 \
  --embedding-dim 128 \
  --hidden-dim 256 \
  --num-layers 2
```

输出文件：

- `work4/results/lstm_lm/best_lstm_lm.pt`
- `work4/results/lstm_lm/vocab.json`
- `work4/results/lstm_lm/history.csv`
- `work4/results/lstm_lm/curves.png`
- `work4/results/lstm_lm/generation_samples.md`
- `work4/results/lstm_lm/metrics.json`

## 生成答案

```bash
python3 work4/generate_answer.py \
  --checkpoint work4/results/lstm_lm/best_lstm_lm.pt \
  --vocab work4/results/lstm_lm/vocab.json \
  --prompt "〔示例〕今有田廣十五步，從十六步。問為田幾何？"
```

## 训练文本嵌入

```bash
python3 work4/train_word2vec.py \
  --data work4/九章算经.txt \
  --epochs 40 \
  --embedding-dim 128 \
  --window-size 4 \
  --negative-samples 8
```

输出文件：

- `work4/results/word2vec/skipgram_embeddings.pt`
- `work4/results/word2vec/vocab.json`
- `work4/results/word2vec/history.csv`
- `work4/results/word2vec/neighbors.md`
- `work4/results/word2vec/neighbors.png`
- `work4/results/word2vec/metrics.json`

## 查询文本嵌入近邻

word2vec/Skip-gram 是嵌入模型，不做答案续写。它的示例输出是给定一个字或术语，返回向量空间中最相近的邻居：

```bash
python3 work4/query_embeddings.py \
  --checkpoint work4/results/word2vec/skipgram_embeddings.pt \
  --vocab work4/results/word2vec/vocab.json \
  --queries "田,畝,分,率,粟,米,步,尺"
```

一键脚本会自动把示例结果写到：

- `work4/results/word2vec/manual_neighbors.md`

## 报告

LaTeX 报告文件为：

- `work4/report.tex`

编译：

```bash
cd work4
xelatex report.tex
```

服务器训练完成后，报告中的图表会直接引用 `results/lstm_lm/curves.png` 和 `results/word2vec/neighbors.png`。若需要写入最终数值，可从两个 `metrics.json` 中复制。
