# L11: Recurrent Neural Networks and Transformer

**课程**: DDA3020 Machine Learning, CUHK-SZ  
**主讲**: Baoyuan Wu  
**日期**: March 30 / April 1, 2026

---

## 目录 (Table of Contents)

1. [序列数据分析：动机与挑战](#1-序列数据分析动机与挑战)
2. [RNN基础：为什么普通神经网络不够用](#2-rnn基础为什么普通神经网络不够用)
3. [Basic RNN 架构](#3-basic-rnn-架构)
   - 3.1 [RNN结构与前向传播](#31-rnn结构与前向传播)
   - 3.2 [RNN计算图与展开](#32-rnn计算图与展开)
   - 3.3 [序列建模类型](#33-序列建模类型)
4. [RNN的训练：BPTT](#4-rnn的训练bptt)
   - 4.1 [代价函数](#41-代价函数)
   - 4.2 [Backpropagation Through Time (BPTT)](#42-backpropagation-through-time-bptt)
   - 4.3 [Truncated BPTT](#43-truncated-bptt)
5. [梯度消失与梯度爆炸](#5-梯度消失与梯度爆炸)
6. [LSTM (Long Short-Term Memory)](#6-lstm-long-short-term-memory)
   - 6.1 [LSTM的动机](#61-lstm的动机)
   - 6.2 [LSTM门控机制详解](#62-lstm门控机制详解)
   - 6.3 [LSTM前向传播公式](#63-lstm前向传播公式)
   - 6.4 [为什么LSTM缓解梯度消失](#64-为什么lstm缓解梯度消失)
7. [GRU (Gated Recurrent Unit)](#7-gru-gated-recurrent-unit)
8. [RNN的扩展（选修）](#8-rnn的扩展选修)
9. [RNN及其变体的局限性](#9-rnn及其变体的局限性)
10. [Transformer架构](#10-transformer架构)
    - 10.1 [Transformer总览](#101-transformer总览)
    - 10.2 [Input Embeddings 与 Positional Encoding](#102-input-embeddings-与-positional-encoding)
    - 10.3 [Attention机制](#103-attention机制)
    - 10.4 [Self-Attention vs Cross-Attention](#104-self-attention-vs-cross-attention)
    - 10.5 [Multi-Head Attention](#105-multi-head-attention)
    - 10.6 [Feed-Forward Network](#106-feed-forward-network)
    - 10.7 [Layer Normalization 与残差连接](#107-layer-normalization-与残差连接)
    - 10.8 [Encoder-Decoder结构](#108-encoder-decoder结构)
11. [Transformer的优势 vs RNN](#11-transformer的优势-vs-rnn)
12. [Transformer的应用](#12-transformer的应用)
13. [总结](#13-总结)
14. [补充资料 (Supplementary Resources)](#14-补充资料-supplementary-resources)

---

## 1. 序列数据分析：动机与挑战

### 什么是序列数据

**序列数据 (Sequential Data)**: 数据按照有序序列排列，**顺序很重要**。与表格数据或图像不同，序列数据中每个元素的含义依赖于上下文。

### 序列数据的典型任务

| 任务类型 | 输入 → 输出 | 示例 |
|----------|-------------|------|
| 时间序列预测 | 序列 → 序列 | 股票价格预测、气象预报 |
| 语音识别 | 音频 → 文本 | Google Assistant, Siri |
| 机器翻译 | 文本 → 文本 | Google Translate ("Je suis étudiant" → "I am a student") |
| 图像描述 | 图像 → 文本 | 看图说话（Image Captioning） |
| 语言建模 | 文本 → 文本 | ChatGPT、SORA |
| 生物序列分析 | DNA/蛋白质序列 → 功能 | AlphaFold |
| 医学影像时序 | 时序扫描 → 诊断 | 病情进展追踪 |

> **直觉**: "I visited Paris in 2014" 和 "In 2014, I visited Paris" 语义相同，但词序不同。处理序列数据的模型必须能理解这种顺序依赖关系。

---

## 2. RNN基础：为什么普通神经网络不够用

### MLP 和 CNN 的局限

- **MLP** 处理**表格数据**：固定维度输入，无法处理变长序列
- **CNN** 处理**图像数据**：保留空间结构，但对时序/序列无法自然处理

MLP/CNN 无法处理序列数据的根本原因：
1. **固定输入长度**：真实序列长度各不相同
2. **无状态记忆**：无法"记住"之前的信息
3. **不共享参数**：不同时间步的参数各自独立，无法利用时间上的统计规律

---

## 3. Basic RNN 架构

### 3.1 RNN结构与前向传播

RNN 引入**循环连接**，允许网络在时间步之间传递信息：

> **RNN 前向传播公式**:
> $$h_t = f_W(h_{t-1}, x_t) = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$$
> $$\hat{y}_t = g_{W'}(h_t) = W_{hy} h_t + b_y$$

其中：
- $h_t \in \mathbb{R}^d$: 时刻 $t$ 的**隐藏状态** (hidden state)，是 RNN 的"记忆"
- $x_t$: 时刻 $t$ 的输入
- $\hat{y}_t$: 时刻 $t$ 的预测输出
- $W_{hh}$: 隐藏状态到隐藏状态的权重矩阵（**时间步间共享**）
- $W_{xh}$: 输入到隐藏状态的权重矩阵（**时间步间共享**）
- $W_{hy}$: 隐藏状态到输出的权重矩阵

**关键特点：参数共享 (Parameter Sharing)**  
所有时间步使用**同一组**参数 $\{W_{hh}, W_{xh}, W_{hy}\}$，这使得：
- 模型可以处理任意长度的序列
- 大幅减少参数量
- 具有时间平移不变性（在不同位置出现的同一模式被同样处理）

### 3.2 RNN计算图与展开

**压缩图** (folded form):
```
      ┌──────┐
  ┌──▶│  RNN │──▶ ŷ_t
  │   └──┬───┘
  │      │ h_t
  │      ▼
  h_{t-1}  x_t
```

**展开图** (unfolded form，按时间步展开):
```
x_1         x_2         x_3  ...  x_T
 ↓           ↓           ↓          ↓
h_0 → [RNN] → h_1 → [RNN] → h_2 → [RNN] → ... → h_T
        ↓            ↓            ↓                 ↓
       ŷ_1          ŷ_2          ŷ_3              ŷ_T
```

> **直觉**: 展开图揭示了 RNN 本质上是一个**参数共享的深度前馈网络**，深度等于序列长度。这也解释了为何 RNN 训练时会遇到梯度问题（序列越长，"深度"越深）。

### 3.3 序列建模类型

RNN 可以灵活地处理不同的输入-输出序列关系：

| 类型 | 结构 | 应用 |
|------|------|------|
| **One-to-One** | 固定输入→固定输出 | 普通分类（退化为MLP）|
| **One-to-Many** | 固定输入→序列输出 | 图像描述（输入图像，输出文字）|
| **Many-to-One** | 序列输入→固定输出 | 情感分析（输入文本，输出情感）|
| **Many-to-Many (同步)** | 序列→同长度序列 | 视频帧标注 |
| **Many-to-Many (异步)** | 序列→不同长度序列 | 机器翻译（Seq2Seq）|

---

## 4. RNN的训练：BPTT

### 4.1 代价函数

对于 Many-to-Many 任务，代价函数是所有时间步损失的平均：

> $$E(\theta) = \frac{1}{T} \sum_{t=1}^{T} L(y_t, \hat{y}_t), \quad \theta = \{W, W'\}$$

其中 $L(y_t, \hat{y}_t)$ 是时刻 $t$ 的损失（如交叉熵），$W$ 是 RNN 权重，$W'$ 是输出层权重。

### 4.2 Backpropagation Through Time (BPTT)

BPTT 是标准反向传播在时间维度上的扩展。由于 RNN 展开后等价于深层前馈网络，BPTT 沿时间步反向传播梯度。

**梯度计算**（以 $W_{hh}$ 为例）:

$$\frac{\partial E}{\partial W_{hh}} = \sum_{t=1}^{T} \frac{\partial L_t}{\partial W_{hh}}$$

对每个时刻 $t$ 的损失，梯度需要沿时间反向传递到更早的时间步：

$$\frac{\partial L_t}{\partial W_{hh}} = \sum_{k=1}^{t} \frac{\partial L_t}{\partial h_t} \left(\prod_{j=k+1}^{t} \frac{\partial h_j}{\partial h_{j-1}}\right) \frac{\partial h_k}{\partial W_{hh}}$$

其中 $\frac{\partial h_j}{\partial h_{j-1}} = W_{hh}^T \cdot \text{diag}(\tanh'(h_{j-1}))$

**关键观察**: 梯度中包含 $\prod_{j=k+1}^{t} \frac{\partial h_j}{\partial h_{j-1}}$ — 这是一个从时刻 $t$ 到时刻 $k$ 的**矩阵乘积链**，正是梯度问题的根源。

### 4.3 Truncated BPTT

由于完整 BPTT 对长序列代价极高，实践中使用**截断 BPTT (Truncated BPTT)**：

- **前向传播**: 隐藏状态 $h_t$ 一直向前传播（不截断）
- **反向传播**: 只反向传播固定步数（如 50 步）

类比于普通神经网络的 mini-batch 训练：前向传播维持"记忆"，反向传播限制范围以控制计算代价。

---

## 5. 梯度消失与梯度爆炸

这是 RNN 训练中最核心的挑战：

| 问题 | 描述 | 原因 | 影响 |
|------|------|------|------|
| **梯度爆炸 (Gradient Exploding)** | 梯度值指数级增大 | $\|W_{hh}\| > 1$，矩阵连乘后幂次增大 | 训练数值不稳定，loss 出现 NaN |
| **梯度消失 (Gradient Vanishing)** | 梯度值指数级衰减至接近0 | $\|W_{hh}\| < 1$ 或 $\tanh'$ 的饱和区梯度趋零 | 无法学习长期依赖关系 |

### 数学分析

BPTT 中的梯度链：
$$\prod_{j=k+1}^{t} \frac{\partial h_j}{\partial h_{j-1}} = \prod_{j=k+1}^{t} W_{hh}^T \cdot \text{diag}(\tanh'(\cdot))$$

- 当 $\lambda_{\max}(W_{hh}) > 1$：$t - k$ 步后梯度 $\approx \lambda_{\max}^{t-k} \to \infty$（爆炸）
- 当 $\lambda_{\max}(W_{hh}) < 1$：$t - k$ 步后梯度 $\approx \lambda_{\max}^{t-k} \to 0$（消失）

### 解决方案

1. **梯度裁剪 (Gradient Clipping)**: 设置梯度上下限阈值，专门针对**梯度爆炸**
   $$\mathbf{g} \leftarrow \min\left(1, \frac{\text{threshold}}{\|\mathbf{g}\|}\right) \cdot \mathbf{g}$$

2. **LSTM / GRU**: 通过门控机制缓解**梯度消失**（见下节）

3. **更好的激活函数**: ReLU 在正区间梯度为1，减轻消失

4. **Highway Networks / ResNet**: 残差连接提供梯度"高速公路"

> **直觉**: 梯度消失意味着网络"遗忘"了很早之前的信息。例如，在翻译长句时，当模型生成最后一个词时，可能已经完全忘记了句子开头的主语是什么。

---

## 6. LSTM (Long Short-Term Memory)

### 6.1 LSTM的动机

Basic RNN 只有一种"记忆"：隐藏状态 $h_t$。

LSTM（由 Hochreiter 和 Schmidhuber 于 **1997 年**提出，2013-2015 年广泛应用）引入了两种记忆：
- **Cell State** $C_t$：长期记忆，负责存储跨时间步的信息
- **Hidden State** $h_t$：短期记忆，用于当前时间步的输出

LSTM 的核心思想：**通过门控机制**精细地控制信息的保留、丢弃和更新。

### 6.2 LSTM门控机制详解

LSTM 有三个门（gate），每个门都是一个值在 $(0,1)$ 之间的向量（由 sigmoid 函数产生）：

#### Forget Gate（遗忘门）$f_t$

**功能**: 决定从 Cell State 中**丢弃**哪些信息

$$f_t = \sigma(W_f [h_{t-1}, x_t] + b_f)$$

- $f_t \approx 0$：完全遗忘
- $f_t \approx 1$：完全保留

#### Input Gate（输入门）$i_t$ 与候选值 $\tilde{C}_t$

**功能**: 决定向 Cell State 中**写入**哪些新信息

$$i_t = \sigma(W_i [h_{t-1}, x_t] + b_i)$$
$$\tilde{C}_t = \tanh(W_C [h_{t-1}, x_t] + b_C)$$

- $i_t$：控制更新的强度
- $\tilde{C}_t$：候选新信息（$\tanh$ 将值压缩到 $[-1, 1]$）

#### Cell State 更新

$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$

> **直觉**: Cell State 就像一条"信息高速公路"——遗忘门选择性地清除旧信息，输入门选择性地写入新信息。$\odot$ 表示逐元素乘法。

#### Output Gate（输出门）$o_t$

**功能**: 决定从 Cell State 中**读取**哪些信息作为当前输出

$$o_t = \sigma(W_o [h_{t-1}, x_t] + b_o)$$
$$h_t = o_t \odot \tanh(C_t)$$

### 6.3 LSTM前向传播公式

> **LSTM完整前向传播**（将上述公式整理）:
> $$f_t = \sigma(W_f [h_{t-1}, x_t] + b_f) \quad \text{(遗忘门)}$$
> $$i_t = \sigma(W_i [h_{t-1}, x_t] + b_i) \quad \text{(输入门)}$$
> $$\tilde{C}_t = \tanh(W_C [h_{t-1}, x_t] + b_C) \quad \text{(候选 Cell State)}$$
> $$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t \quad \text{(Cell State 更新)}$$
> $$o_t = \sigma(W_o [h_{t-1}, x_t] + b_o) \quad \text{(输出门)}$$
> $$h_t = o_t \odot \tanh(C_t) \quad \text{(Hidden State 输出)}$$

**架构图（文字描述）**:
```
         C_{t-1} ──────────────────────────────────── C_t
                    ×(f_t)    +(i_t × C̃_t)
                    │               │
h_{t-1} ──┬─────── forget gate     ↑
           │         (σ)     input gate+tanh  output gate
           │                                    │
x_t  ──────┴──────────────────────────────── h_t = o_t ⊙ tanh(C_t)
```

**LSTM 参数量**（相比 Basic RNN 约 4 倍）:
- 4 个门各需要 $W_*$ 矩阵：$4 \times (d_h \times (d_h + d_x)) + 4 \times d_h$（偏置）
- 其中 $d_h$ 是隐藏状态维度，$d_x$ 是输入维度

### 6.4 为什么LSTM缓解梯度消失

关键在于 Cell State 的更新是**加性**（additive）的：

$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$

梯度通过 Cell State 反传时：
$$\frac{\partial C_t}{\partial C_{t-1}} = f_t$$

这是一个逐元素乘法（而非矩阵乘法），且 $f_t$ 在 $(0,1)$ 之间，当 $f_t \approx 1$ 时（遗忘门接近1，选择记忆），梯度接近 1，不会指数衰减！

> **重要说明**: LSTM **不能保证**完全消除梯度消失/爆炸，但提供了**更容易学习长期依赖**的路径。

---

## 7. GRU (Gated Recurrent Unit)

**提出时间**: 2014年  
**作者**: Cho et al. (*"Learning phrase representations using RNN encoder-decoder for statistical machine translation"*)

### GRU的设计

GRU 是 LSTM 的简化版，合并了遗忘门和输入门，去掉了独立的 Cell State：

> **GRU前向传播公式**:
> $$z_t = \sigma(W_z [h_{t-1}, x_t] + b_z) \quad \text{(Update Gate 更新门)}$$
> $$r_t = \sigma(W_r [h_{t-1}, x_t] + b_r) \quad \text{(Reset Gate 重置门)}$$
> $$\tilde{h}_t = \tanh(W_h [r_t \odot h_{t-1}, x_t] + b_h) \quad \text{(候选 Hidden State)}$$
> $$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t \quad \text{(Hidden State 更新)}$$

**门的含义**:
- **Update Gate** $z_t$: 控制保留多少过去信息 vs 写入多少新信息（相当于 LSTM 中遗忘门+输入门的合并）
- **Reset Gate** $r_t$: 控制过去信息对候选状态计算的影响程度

### LSTM vs GRU 对比

| 特性 | LSTM | GRU |
|------|------|-----|
| 参数量 | 较多（4组参数矩阵） | 较少（3组参数矩阵） |
| 门数量 | 3个（遗忘、输入、输出）| 2个（更新、重置）|
| 记忆 | 分离的 $C_t$ 和 $h_t$ | 只有 $h_t$ |
| 计算速度 | 较慢 | 较快 |
| 性能 | 在大数据集上通常稍好 | 在小数据集上与LSTM相当 |
| 复杂度 | 较高 | 较低 |

> **实践建议**: 资源受限时优先尝试 GRU；数据充足时 LSTM 通常略胜一筹。现代实践中两者已逐渐被 Transformer 取代。

---

## 8. RNN的扩展（选修）

### Multi-layer RNN（多层RNN）

将多个 RNN 层堆叠，每层的输出作为下一层的输入：

```
x_t → RNN Layer 1 → h^(1)_t → RNN Layer 2 → h^(2)_t → ... → 输出
```

每层 RNN 独立维护自己的隐藏状态，可以学习不同层次的时序特征。常用 2-4 层堆叠。

### Bidirectional RNN（双向RNN）

在某些任务中（如情感分析、命名实体识别），每个时间步的预测不仅依赖过去，也依赖未来：

```
x_1 → x_2 → x_3 → ... (前向)
x_T ← x_{T-1} ← ... (后向)
```

将前向和后向的隐藏状态拼接：$h_t = [\overrightarrow{h}_t; \overleftarrow{h}_t]$

---

## 9. RNN及其变体的局限性

### 主要局限

1. **长期依赖困难 (Long-term Dependency)**
   - 即使是 LSTM/GRU，超过几百个时间步的依赖仍然难以学习
   - 梯度问题没有被根本解决
   
2. **无法并行化 (Limited Parallelization)**
   - RNN 必须按时间步**顺序**计算：$h_t$ 依赖 $h_{t-1}$
   - 无法像 CNN/FC 那样批量并行处理
   - 导致训练速度慢，尤其是对长序列

> 正是这两个根本性局限，催生了 **Transformer** 架构的出现。

---

## 10. Transformer架构

### 10.1 Transformer总览

**提出时间**: 2017年  
**论文**: Vaswani et al., *"Attention Is All You Need"*, NeurIPS 2017

Transformer 是一种基于**纯注意力机制**的深度学习架构，由 Google 的研究人员提出，彻底抛弃了循环和卷积，仅依赖注意力机制来建模序列中的依赖关系。

**核心组成部分**:
1. **Input Embeddings**: 将离散 token 映射为连续向量
2. **Positional Encoding**: 编码序列中的位置信息
3. **Multi-Head Self-Attention**: 核心机制，捕获全局依赖
4. **Feed-Forward Network**: 逐位置的全连接层
5. **Add & Norm**: 残差连接 + Layer Normalization

### 10.2 Input Embeddings 与 Positional Encoding

#### Input Embeddings

将序列中每个 token（如单词）映射为固定维度 $d$ 的向量：
- 对于文本：词嵌入（Word Embedding）
- 字符级：one-hot 编码
- 图像 patch：CNN 特征

设 $X \in \mathbb{R}^{n \times d}$，其中 $n$ 是序列长度，$d$ 是嵌入维度。

#### Positional Encoding

由于 Transformer 没有循环结构，无法自动感知位置信息，需要显式注入：

> **正弦余弦 Positional Encoding**:
> $$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right)$$
> $$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$

其中 $pos$ 是位置，$i$ 是维度索引，$d$ 是模型维度。

将 positional encoding 与 input embedding 相加：$X' = X + PE$

> **直觉**: 不同频率的正弦/余弦函数组合，使得每个位置有唯一的编码，且相邻位置的编码相似。

参考资料: https://kazemnejad.com/blog/transformer_architecture_positional_encoding/

### 10.3 Attention机制

Attention 机制的核心思想：**根据输入内容，动态地分配不同位置的注意力权重**。

#### Query, Key, Value 框架

设输入嵌入矩阵 $X \in \mathbb{R}^{n \times d}$，定义三个投影矩阵：
$$W_Q, W_K, W_V \in \mathbb{R}^{d \times d'}$$

计算 Query、Key、Value：
$$Q = XW_Q, \quad K = XW_K, \quad V = XW_V$$

#### Self-Attention 公式

> **Scaled Dot-Product Attention**:
> $$Z = \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d'}}\right) V$$

其中 $A := \text{softmax}\left(\frac{QK^\top}{\sqrt{d'}}\right) \in \mathbb{R}^{n \times n}$ 被称为 **Attention Map**（注意力矩阵）。

**逐步推导**:

1. **计算相似度**: $S = QK^\top \in \mathbb{R}^{n \times n}$，$S_{ij} = q_i \cdot k_j$ 衡量位置 $i$ 对位置 $j$ 的关注度

2. **缩放 (Scaling)**: 除以 $\sqrt{d'}$，防止内积值过大导致 softmax 进入梯度消失区域
   $$S_{ij} = \frac{q_i \cdot k_j}{\sqrt{d'}}$$

3. **归一化**: $A = \text{softmax}(S)$，将每行归一化为概率分布

4. **加权聚合**: $Z = AV$，用注意力权重对 Value 做加权求和

> **直觉**: 查询 (Query) 是"我想找什么"，键 (Key) 是"我有什么"，值 (Value) 是"我给出什么"。Attention 就像一个软性的信息检索系统。

#### 为什么要除以 $\sqrt{d'}$

当 $d'$ 较大时，$QK^\top$ 的点积量级约为 $\sqrt{d'}$。若不缩放，softmax 的输入值很大，梯度接近0（饱和区），导致梯度消失。除以 $\sqrt{d'}$ 使点积方差接近1，保持良好的梯度流动。

### 10.4 Self-Attention vs Cross-Attention

| 类型 | Q, K, V 来源 | 用途 |
|------|-------------|------|
| **Self-Attention** | 均来自同一输入序列 $X$ | Encoder 内部；Decoder 对自身编码 |
| **Cross-Attention** | Q 来自 Decoder，K/V 来自 Encoder 输出 | Decoder 参考 Encoder 信息 |

**Self-Attention 示例**:
```
句子: "The animal didn't cross the street because it was too tired"
"it" 的 Self-Attention 能学到 "it" 更多关注 "animal"（而非 "street"）
```

**Cross-Attention 示例（机器翻译）**:
```
生成英语 "student" 时，Decoder 通过 Cross-Attention 关注法语输入中的 "étudiant"
```

### 10.5 Multi-Head Attention

单个 Attention head 只能关注一种类型的依赖关系。**Multi-Head Attention** 并行运行多个 Attention 头，每个头学习不同类型的依赖：

> **Multi-Head Attention公式**:
> $$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O$$
> $$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

其中 $W_i^Q \in \mathbb{R}^{d \times d_k}$，$W_i^K \in \mathbb{R}^{d \times d_k}$，$W_i^V \in \mathbb{R}^{d \times d_v}$，$W^O \in \mathbb{R}^{hd_v \times d}$。

通常设 $d_k = d_v = d / h$，保持总计算量不变。

**架构图（文字描述）**:
```
         输入 X
         /  |  \
        /   |   \
    Head1 Head2 ... Head_h
    (各自有独立的W^Q,W^K,W^V)
        \   |   /
         拼接 (Concat)
              ↓
         线性变换 W^O
              ↓
         输出 Z
```

> **直觉**: 不同的 head 可以同时关注不同类型的关系——一个 head 关注句法关系，另一个关注语义关系，另一个关注指代关系等。

### 10.6 Feed-Forward Network

在每个 Attention 层之后，有一个**逐位置**（position-wise）的全连接前馈网络：

> $$\text{FFN}(z) = \max(0, z W_1 + b_1) W_2 + b_2$$

其中 $W_1 \in \mathbb{R}^{d \times d_{ff}}$，$W_2 \in \mathbb{R}^{d_{ff} \times d}$，通常 $d_{ff} = 4d$。

**作用**: Attention 层负责跨位置的信息交互，FFN 层负责在每个位置上对特征做非线性变换（增加模型表达能力）。

### 10.7 Layer Normalization 与残差连接

每个 Attention 层和 FFN 层都包含：
1. **残差连接 (Residual Connection)**: $y = x + \text{Sublayer}(x)$
2. **Layer Normalization**: 对最后一维（特征维度）做归一化

> **Add & Norm 公式**:
> $$y = \text{LayerNorm}(x + \text{Sublayer}(x))$$

**Layer Norm vs Batch Norm**:
- Batch Norm: 在 batch 维度归一化（适合固定尺寸输入）
- Layer Norm: 在特征维度归一化（适合变长序列，不依赖 batch size）

Transformer 原论文中的 Pre-LN vs Post-LN 变体：
- **Post-LN** (原论文): $y = \text{LN}(x + F(x))$
- **Pre-LN** (更稳定): $y = x + F(\text{LN}(x))$

### 10.8 Encoder-Decoder结构

以机器翻译（法语→英语）为例：

**Encoder**（处理源语言）:
```
输入: "Je suis étudiant" (token 化 + embedding + positional encoding)
    ↓
[Self-Attention → Add & Norm → FFN → Add & Norm] × N层
    ↓
编码器输出 (上下文表示 H)
```

**Decoder**（生成目标语言）:
```
输入: 已生成的目标语言 token (+ positional encoding)
    ↓
[Masked Self-Attention → Add & Norm    ← 防止看到未来的token
    ↓
 Cross-Attention (Q来自Decoder, K/V来自Encoder H) → Add & Norm
    ↓
 FFN → Add & Norm] × N层
    ↓
Linear → Softmax → 预测下一个 token
```

**Masked Self-Attention**: Decoder 在训练时（teacher forcing）知道完整目标序列，但不应看到未来的 token（自回归生成要求），因此用 mask 将未来位置的注意力设为 $-\infty$（softmax 后变为0）。

---

## 11. Transformer的优势 vs RNN

| 对比维度 | RNN/LSTM | Transformer |
|----------|----------|-------------|
| **并行化** | ❌ 必须顺序计算 | ✅ 所有位置可并行计算 |
| **长期依赖** | ❌ 梯度消失，难以捕获长距离依赖 | ✅ 任意两个位置直接相互注意，路径长度为 $O(1)$ |
| **模型容量** | 受限于隐藏状态维度 | 更大的 $d$，$h$ heads，$N$ 层 |
| **训练速度** | 慢（序列长度 $T$ 步递推） | 快（矩阵并行计算） |
| **计算复杂度** | $O(nd^2)$ | Self-Attention: $O(n^2 d)$（$n$ 较大时代价高）|
| **对序列长度的扩展性** | 较好（$O(n)$） | 对超长序列有挑战（$O(n^2)$）|

> **核心优势总结（来自讲义）**:
> 1. **更容易并行化 (Easier Parallelization)**
> 2. **更有效地处理长距离依赖 (More Effective in Handling Long-Range Dependencies)**
> 3. **更高的模型容量与灵活性 (Higher Model Capacity and Flexibility)**

---

## 12. Transformer的应用

Transformer 已成为 NLP、CV、多模态 AI 的核心架构：

| 领域 | 模型 | 特点 |
|------|------|------|
| **NLP - 语言理解** | BERT | Encoder-only，双向 |
| **NLP - 语言生成** | GPT 系列 | Decoder-only，自回归 |
| **NLP - Seq2Seq** | T5, BART | Encoder-Decoder |
| **NLP - 超大规模** | GPT-4, ChatGPT | 极大规模 Decoder |
| **CV - 图像分类** | ViT (Vision Transformer) | 将图像分成 patch 序列 |
| **CV - 目标检测** | DETR | Transformer 检测 |
| **视频生成** | SORA | 时空 Transformer |
| **多模态** | CLIP, DALL-E | 文字-图像联合 |

---

## 13. 总结

| 模型 | 主要解决的问题 | 核心机制 |
|------|---------------|----------|
| **Basic RNN** | 序列数据的顺序建模 | 循环隐藏状态 |
| **LSTM** | 梯度消失 + 长期依赖 | 门控 Cell State |
| **GRU** | LSTM 的简化版 | 更新门 + 重置门 |
| **Transformer** | 并行化 + 超长依赖 | Self-Attention |

**发展脉络**: 
$$\text{RNN} \xrightarrow{\text{解决梯度消失}} \text{LSTM/GRU} \xrightarrow{\text{解决并行化+长依赖}} \text{Transformer}$$

---

## 14. 补充资料 (Supplementary Resources)

### 教材参考

1. **Goodfellow, Bengio, Courville** - *Deep Learning* (2016)
   - Chapter 10: Sequence Modeling: Recurrent and Recursive Nets
   - 在线免费: https://www.deeplearningbook.org/

2. **Murphy** - *Machine Learning: A Probabilistic Perspective* (MLAPP, 2012)
   - Chapter 28.3: Recurrent Neural Networks

3. **Bishop** - *Pattern Recognition and Machine Learning* (PRML, 2006)
   - Chapter 5: Neural Networks (基础知识)

### 原始论文

- LSTM 原始论文: Hochreiter & Schmidhuber, 1997 — https://www.bioinf.jku.at/publications/older/2604.pdf
- GRU: Cho et al., 2014 — https://arxiv.org/abs/1406.1078
- **Attention Is All You Need** (Transformer): Vaswani et al., 2017 — https://arxiv.org/abs/1706.03762
- BERT: Devlin et al., 2018 — https://arxiv.org/abs/1810.04805
- GPT-3: Brown et al., 2020 — https://arxiv.org/abs/2005.14165

### 在线资源

- **The Illustrated Transformer** (Jay Alammar): https://jalammar.github.io/illustrated-transformer/
  （最好的 Transformer 可视化教程之一）
- **The Illustrated LSTM**: https://colah.github.io/posts/2015-08-Understanding-LSTMs/
  （LSTM 的经典直觉解释）
- Positional Encoding 详解: https://kazemnejad.com/blog/transformer_architecture_positional_encoding/
- CS224n: Natural Language Processing with Deep Learning (Stanford): https://web.stanford.edu/class/cs224n/

### 代码实现

- PyTorch 内置 RNN/LSTM/GRU: `torch.nn.RNN`, `torch.nn.LSTM`, `torch.nn.GRU`
- PyTorch Transformer: `torch.nn.Transformer`, `torch.nn.MultiheadAttention`
- Hugging Face Transformers 库: https://huggingface.co/transformers/

---

*笔记整理自 DDA3020 L11 讲义 (Baoyuan Wu, CUHK-SZ, March 30/April 1, 2026)*  
*部分幻灯片改编自 Justin Johnson 的课程材料*
