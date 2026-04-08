# Lecture 09: Neural Networks（神经网络）
**DDA3020 Machine Learning — CUHK-SZ**  
**讲师**: Baoyuan Wu | **日期**: March 18/23, 2026

---

## 目录 (Table of Contents)

1. [前置回顾：分类模型](#1-前置回顾分类模型)
2. [生物神经元 → 人工神经元](#2-生物神经元--人工神经元)
3. [激活函数 (Activation Functions)](#3-激活函数-activation-functions)
4. [Perceptron 感知机](#4-perceptron-感知机)
5. [多层感知机 (MLP / Feedforward Neural Networks)](#5-多层感知机-mlp--feedforward-neural-networks)
6. [前向传播 (Forward Propagation)](#6-前向传播-forward-propagation)
7. [损失函数 (Loss Functions)](#7-损失函数-loss-functions)
8. [反向传播 (Backpropagation)](#8-反向传播-backpropagation)
9. [计算图 (Computational Graph)](#9-计算图-computational-graph)
10. [权重初始化 (Weight Initialization)](#10-权重初始化-weight-initialization)
11. [优化器 (Optimizers)](#11-优化器-optimizers)
12. [正则化技术 (Regularization)](#12-正则化技术-regularization)
13. [通用近似定理与深层网络](#13-通用近似定理与深层网络)
14. [深度神经网络与CNN简介](#14-深度神经网络与cnn简介)
15. [补充资料](#15-补充资料)

---

## 1. 前置回顾：分类模型

### 1.1 已学模型对比

| 模型 | 假设函数 | 损失函数 | 学习算法 |
|------|---------|---------|---------|
| **Logistic Regression** | $h_\mathbf{w}(\mathbf{x}) = \sigma(\mathbf{w}^\top \mathbf{x})$ | $-\log \sigma(y \cdot \mathbf{w}^\top \mathbf{x})$ | 梯度下降 |
| **SVM** | $h_{\mathbf{w}}(\mathbf{x}) = \mathbf{w}^\top \mathbf{x} + b$ | $\max(0, 1 - y(\mathbf{w}^\top \mathbf{x}+b))$ | Lagrange 对偶/KKT |

**以上模型的共同局限**：默认输入特征 $\mathbf{x}$ 已给定为向量形式。但在图像分类、文本分类等实际任务中，如何从原始数据提取 $\mathbf{x}$？

- 传统方法：手工设计特征（SIFT、HOG、光流特征等），**特征提取与分类器训练相互独立**
- 神经网络：**端到端 (End-to-End)** 地将特征学习与分类器训练统一

---

## 2. 生物神经元 → 人工神经元

### 2.1 生物神经元

人脑有约 $10^{11}$ 个神经元，每个神经元与约 $10^4$ 个其他神经元相连。

**生物神经元工作原理**：
- 每个神经元有正/负电位
- 当正电位数量超过某个阈值时，神经元被激活（发放信号）
- 激活后产生化学物质传递到相连神经元，改变其电位

### 2.2 M-P 神经元模型

1943年，McCulloch 和 Pitts 受生物神经元启发，提出了 **M-P 神经元模型**：

$$y = f\left(\sum_{i=1}^n w_i x_i - \theta\right)$$

其中 $\theta$ 为阈值，$f(\cdot)$ 为激活函数。

### 2.3 人工神经元

现代神经网络使用更简洁的模型：

$$y = f(\mathbf{w}^\top \mathbf{x} + b)$$

与 Logistic Regression 对比：$y = \sigma(\mathbf{w}^\top \mathbf{x} + b)$，本质上是同一结构，区别在于激活函数的选择。

> **关键洞察**：将大量"极简"的神经元堆叠组合，即可实现强大的非线性计算能力！

---

## 3. 激活函数 (Activation Functions)

激活函数为神经网络引入**非线性**，使其能学习复杂模式。

### 3.1 常用激活函数一览

| 激活函数 | 公式 | 输出范围 | 特点 |
|---------|------|---------|------|
| **Linear（线性）** | $y = z$ | $(-\infty, +\infty)$ | 无非线性，多层等价于单层 |
| **Hard Threshold（硬阈值）** | $y = \mathbf{1}[z > 0]$ | $\{0, 1\}$ | 不可微，历史感知机使用 |
| **Sigmoid（Logistic）** | $y = \frac{1}{1+e^{-z}}$ | $(0, 1)$ | 光滑，但有梯度消失问题 |
| **Tanh（双曲正切）** | $y = \frac{e^z - e^{-z}}{e^z + e^{-z}}$ | $(-1, 1)$ | 零均值，比 Sigmoid 好 |
| **ReLU** | $y = \max(0, z)$ | $[0, +\infty)$ | 计算高效，缓解梯度消失 |
| **Leaky ReLU** | $y = \max(\alpha z, z)$（$\alpha \approx 0.01$） | $(-\infty, +\infty)$ | 解决 ReLU 死亡问题 |
| **Soft ReLU（Softplus）** | $y = \log(1 + e^z)$ | $(0, +\infty)$ | ReLU 的光滑版本 |

### 3.2 Softmax（多分类输出层）

用于多分类任务的输出层，将 $K$ 个原始分数（logits）$z_1, \ldots, z_K$ 转换为概率分布：

> $$\text{Softmax}(z_k) = \frac{e^{z_k}}{\sum_{j=1}^K e^{z_j}}, \quad k = 1, \ldots, K$$

**性质**：输出之和为 1，每个输出非负，可解释为概率。

### 3.3 Sigmoid 的导数（反向传播重要性质）

$$\sigma'(z) = \sigma(z)(1 - \sigma(z))$$

**推导**：

$$\sigma'(z) = \frac{e^{-z}}{(1+e^{-z})^2} = \frac{1}{1+e^{-z}} \cdot \frac{e^{-z}}{1+e^{-z}} = \sigma(z)(1-\sigma(z))$$

### 3.4 激活函数的选择建议

- **隐藏层**：优先使用 **ReLU**（训练快，性能好，是当前默认选择）
- **二分类输出层**：**Sigmoid**（将输出压缩到 $(0,1)$，解释为正类概率）
- **多分类输出层**：**Softmax**（输出 $K$ 个概率）
- **回归输出层**：**Linear**（无激活）
- **RNN/LSTM 中**：**Tanh**（零均值，输出范围对称）

---

## 4. Perceptron 感知机

### 4.1 感知机定义

**Perceptron（感知机）** 是最简单的神经网络，由 Rosenblatt 于 1957 年提出：

- 一个输入层接收输入信号
- 一个输出层包含一个 M-P 神经元（阈值逻辑单元）

**公式**：$y = f(\mathbf{w}^\top \mathbf{x} + b) = \text{Sgn}(\mathbf{w}^\top \mathbf{x} + b)$

**目标函数**（均方误差）：

$$J(\mathbf{w}) = \frac{1}{2}(y - t)^2 = \frac{1}{2}(\text{Sgn}(\mathbf{w}^\top \mathbf{x} + b) - t)^2$$

**梯度下降学习规则**（将 $\text{Sgn}(\cdot)$ 的梯度近似为 1）：

$$\mathbf{w} \leftarrow \mathbf{w} - \eta(y - t)\mathbf{x}$$

**预测**：若 $\mathbf{w}^\top \mathbf{x} + b > 0$，预测 $+1$；否则预测 $-1$。

### 4.2 感知机的能力与局限

**能力**：感知机可以模拟任意简单的二元布尔门：

| 布尔函数 | 是否可用感知机模拟 |
|---------|-----------------|
| AND | ✅ 可以（线性可分） |
| OR | ✅ 可以（线性可分） |
| NOT | ✅ 可以（线性可分） |
| **XOR** | ❌ **不能**（非线性可分） |

> **感知机局限**：单层感知机只能学习**线性可分**的函数。XOR 问题无法用一条直线（超平面）分开，感知机无法解决。

这正是多层网络存在的动机。

---

## 5. 多层感知机 (MLP / Feedforward Neural Networks)

### 5.1 定义与结构

**多层前馈神经网络 (Multi-layer Feedforward Neural Network / MLP)** 的结构：
- 一个输入层（Input Layer）
- 一或多个隐藏层（Hidden Layer）
- 一个输出层（Output Layer）
- **全连接 (Fully Connected)**：相邻两层之间所有神经元相互连接
- **单向传播**：信息从输入层到输出层，无反馈连接（无环）
- **同层无连接**：同一层内神经元间无连接
- **无跳连接**（基础 MLP，非残差网络）

**数学表示**（两层 MLP）：

$$\mathbf{y} = g_1(\mathbf{w}^\top \mathbf{h} + b)$$
$$\mathbf{h} = g_2(\mathbf{W}\mathbf{x} + \mathbf{c})$$

### 5.2 XOR 问题的解决

**用符号激活函数的解法**：

$$\mathbf{y} = g_1(\mathbf{w}^\top \mathbf{h} + b) = \text{Sgn}\left(\begin{bmatrix}1 \\ -1\end{bmatrix}^\top \mathbf{h} - 0.5\right)$$

$$\mathbf{h} = g_2(\mathbf{W}\mathbf{x} + \mathbf{c}) = \text{Sgn}\left(\begin{bmatrix}1 & 1 \\ 1 & 1\end{bmatrix}\mathbf{x} - \begin{bmatrix}0.5 \\ 1.5\end{bmatrix}\right)$$

**核心直觉**：通过隐藏层将数据变换到新的特征空间，使其在新空间中**线性可分**，再用线性分类器。

### 5.3 多层网络的模块化视角

每层计算一个函数，整体为函数复合：

$$\mathbf{h}^{(1)} = f^{(1)}(\mathbf{x})$$
$$\mathbf{h}^{(2)} = f^{(2)}(\mathbf{h}^{(1)})$$
$$\vdots$$
$$\mathbf{y} = f^{(L)}(\mathbf{h}^{(L-1)})$$

等价于：

$$\mathbf{y} = f^{(L)} \circ \cdots \circ f^{(1)}(\mathbf{x})$$

**神经网络的特征学习视角**：每一隐藏层学习将数据变换为更适合下一步分类/预测的特征表示，目标是将非线性可分数据变换为线性可分。

### 5.4 数字识别例子

对 $28 \times 28 = 784$ 像素的手写数字图像：
- 输入层：784 维向量
- 第一隐藏层：每个神经元计算 $\sigma(\mathbf{w}_j^\top \mathbf{x})$，起**特征检测器**的作用
- 可视化权重 $\mathbf{w}_j$（reshape 为 $28\times28$）能看到该神经元响应的笔划模式

---

## 6. 前向传播 (Forward Propagation)

### 6.1 通用符号

对 $L$ 层神经网络（第 0 层为输入），定义：
- $\mathbf{a}^{(l)}$：第 $l$ 层的激活值（输出）
- $\mathbf{z}^{(l)}$：第 $l$ 层的预激活值（线性组合）
- $\mathbf{W}^{(l)}$：第 $l$ 层的权重矩阵（$m_l \times m_{l-1}$）
- $\mathbf{b}^{(l)}$：第 $l$ 层的偏置向量

### 6.2 前向传播公式

$$\mathbf{a}^{(0)} = \mathbf{x} \quad \text{（输入）}$$

$$\mathbf{z}^{(l)} = \mathbf{W}^{(l)} \mathbf{a}^{(l-1)} + \mathbf{b}^{(l)}, \quad l = 1, \ldots, L$$

$$\mathbf{a}^{(l)} = f^{(l)}(\mathbf{z}^{(l)}), \quad l = 1, \ldots, L$$

最终输出：$\hat{\mathbf{y}} = \mathbf{a}^{(L)}$

---

## 7. 损失函数 (Loss Functions)

### 7.1 均方误差 (MSE) — 回归

$$L = \frac{1}{2}(\hat{y} - y)^2$$

更一般地，对 $m$ 个样本：$J = \frac{1}{m}\sum_{i=1}^m \frac{1}{2}(\hat{y}_i - y_i)^2$

### 7.2 交叉熵损失 (Cross-Entropy) — 分类

**二分类**：

$$L = -[y\log\hat{y} + (1-y)\log(1-\hat{y})]$$

**多分类**（与 Softmax 配合）：

$$L = -\sum_{k=1}^K y_k \log \hat{y}_k$$

其中 $y_k$ 为 one-hot 编码，$\hat{y}_k = \text{Softmax}(z_k)$。

### 7.3 Hinge Loss — SVM 视角

$$L = \max(0, 1 - y \cdot \hat{y})$$

---

## 8. 反向传播 (Backpropagation)

### 8.1 为什么需要反向传播？

神经网络的损失函数是参数的复合函数，对每个参数求导（梯度）是训练的关键。直接代入复合函数求解析导数在参数很多时极为繁琐。

**反向传播 (Backpropagation, BP)** 是一种高效计算梯度的算法，基于**链式法则 (Chain Rule)**。

### 8.2 链式法则

**标量版本**：若 $f(h)$ 和 $h(w)$ 均为单变量函数，则：

$$\frac{df(h(w))}{dw} = \frac{df}{dh} \cdot \frac{dh}{dw}$$

**向量版本**（雅可比矩阵）：若 $\mathbf{y} = f(\mathbf{z})$，$\mathbf{z} = g(\mathbf{x})$，则：

$$\frac{\partial \mathbf{y}}{\partial \mathbf{x}} = \frac{\partial \mathbf{y}}{\partial \mathbf{z}} \cdot \frac{\partial \mathbf{z}}{\partial \mathbf{x}}$$

### 8.3 简单例子：单神经元梯度计算

设损失函数：$L = \frac{1}{2}(\sigma(wx + b) - t)^2$（标量，$w, x, b, t$ 均为标量）

**直接代入求导法**：

$$\frac{\partial L}{\partial w} = (\sigma(wx+b) - t) \cdot \sigma'(wx+b) \cdot x$$

$$\frac{\partial L}{\partial b} = (\sigma(wx+b) - t) \cdot \sigma'(wx+b)$$

### 8.4 结构化计算（前向+反向）

将计算分解为若干中间变量：

| 步骤 | 前向计算 | 反向传播（梯度） |
|------|---------|---------------|
| 1 | $z = wx + b$ | $\frac{\partial L}{\partial z} = \frac{\partial L}{\partial y}\sigma'(z)$ |
| 2 | $y = \sigma(z)$ | $\frac{\partial L}{\partial y} = y - t$ |
| 3 | $L = \frac{1}{2}(y-t)^2$ | — |

参数梯度：
$$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial z} \cdot \frac{\partial z}{\partial w} = \frac{\partial L}{\partial z} \cdot x$$

$$\frac{\partial L}{\partial b} = \frac{\partial L}{\partial z} \cdot \frac{\partial z}{\partial b} = \frac{\partial L}{\partial z}$$

> **核心思想**：不追求闭合解析解，而是编写能高效计算导数的程序。

### 8.5 多层网络的反向传播

设网络中相邻两层的激活值为 $y_i$（前层）和 $y_j$（后层），连接权重为 $w_{ij}$：

**后层预激活值梯度**：

$$\frac{\partial L}{\partial z_j} = \frac{dy_j}{dz_j} \cdot \frac{\partial L}{\partial y_j} = y_j(1 - y_j) \cdot \frac{\partial L}{\partial y_j}$$

**前层激活值梯度**（从后层反传回前层）：

$$\frac{\partial L}{\partial y_i} = \sum_j \frac{dz_j}{dy_i} \cdot \frac{\partial L}{\partial z_j} = \sum_j w_{ij} \cdot \frac{\partial L}{\partial z_j}$$

**权重梯度**：

$$\frac{\partial L}{\partial w_{ij}} = \frac{\partial z_j}{\partial w_{ij}} \cdot \frac{\partial L}{\partial z_j} = y_i \cdot \frac{\partial L}{\partial z_j}$$

> **反向传播三步公式总结**：
> 1. $\frac{\partial L}{\partial z_j} = y_j(1-y_j)\frac{\partial L}{\partial y_j}$（激活函数导数×上游梯度）
> 2. $\frac{\partial L}{\partial y_i} = \sum_j w_{ij}\frac{\partial L}{\partial z_j}$（权重加权求和）
> 3. $\frac{\partial L}{\partial w_{ij}} = y_i \frac{\partial L}{\partial z_j}$（前层激活×后层梯度）

结合梯度下降：$\mathbf{w} \leftarrow \mathbf{w} - \alpha \frac{\partial J(\mathbf{w})}{\partial \mathbf{w}}$

---

## 9. 计算图 (Computational Graph)

### 9.1 定义

**计算图 (Computational Graph, CG)** 将计算过程用有向无环图（DAG）表示：
- **节点**：所有输入量和中间计算量
- **边**：哪些节点直接依赖哪些节点

### 9.2 例子：$z = wx + b$

```
x ──┐
     ├─[×]─ wx ─┬─[+]─ z
w ──┘            │
b ───────────────┘
```

### 9.3 前向传播和反向传播的对应

对于计算图，可形式化地定义：

**前向传播**（按拓扑序 $v_1, \ldots, v_N$）：

$$\text{For } i = 1, \ldots, N: \quad v_i = \text{func}(\text{Pa}(v_i))$$

**反向传播**（反向拓扑序）：

$$\overline{v}_N = 1 \quad \text{（终节点梯度初始化为 1）}$$

$$\overline{v}_i = \sum_{j \in \text{Ch}(v_i)} \overline{v}_j \cdot \frac{\partial v_j}{\partial v_i} \quad \text{（累加子节点的反向贡献）}$$

其中 $\overline{v}_i = \frac{\partial L}{\partial v_i}$，$\text{Pa}(v_i)$ 表示父节点，$\text{Ch}(v_i)$ 表示子节点。

### 9.4 计算图的操作符

| 操作 | 前向 | 反向（对各输入的梯度） |
|------|------|---------------------|
| $z = x + y$ | $z = x + y$ | $\partial L/\partial x = \partial L/\partial z$，$\partial L/\partial y = \partial L/\partial z$ |
| $z = x \cdot y$ | $z = xy$ | $\partial L/\partial x = y \cdot \partial L/\partial z$，$\partial L/\partial y = x \cdot \partial L/\partial z$ |
| $z = \sigma(x)$ | $z = \sigma(x)$ | $\partial L/\partial x = z(1-z) \cdot \partial L/\partial z$ |
| $z = \max(0, x)$ | $z = \text{ReLU}(x)$ | $\partial L/\partial x = \mathbf{1}[x>0] \cdot \partial L/\partial z$ |

### 9.5 反向传播的计算代价

设网络有 $m$ 个隐藏单元，输入维度 $d$，权重矩阵 $\mathbf{W} \in \mathbb{R}^{m \times d}$：

**前向传播代价**：$O_F = O(md + m)$（矩阵乘法 + 激活）

**反向传播代价**：$O_B \approx O(2md + 4m)$

> **结论**：反向传播的计算代价约为前向传播的 **2倍**，与隐藏层宽度的平方成比例（对于矩阵运算部分）。

---

## 10. 权重初始化 (Weight Initialization)

> **注**：课程讲义未详细展开，以下为重要补充内容（Murphy MLAPP Chapter 16.3，Bishop PRML Chapter 5.5）。

### 10.1 为什么初始化重要？

- **全零初始化**：所有神经元计算相同输出，梯度相同，永远对称（"对称性打破"问题）
- **过大初始值**：激活值饱和（Sigmoid/Tanh），梯度消失
- **过小初始值**：信号太弱，传播中消失

### 10.2 Xavier 初始化（适用于 Sigmoid/Tanh）

$$w \sim \mathcal{U}\left(-\sqrt{\frac{6}{n_\text{in} + n_\text{out}}}, \sqrt{\frac{6}{n_\text{in} + n_\text{out}}}\right)$$

**原理**：保持每层输出的方差与输入相同，防止信号在前向和反向传播中放大或缩小。

### 10.3 He 初始化（适用于 ReLU）

$$w \sim \mathcal{N}\left(0, \frac{2}{n_\text{in}}\right)$$

**原理**：考虑 ReLU 将约一半的激活值置零，因此方差需要加倍。

### 10.4 初始化方法比较

| 方法 | 公式 | 适合激活函数 |
|------|------|------------|
| **随机小正态** | $\mathcal{N}(0, 0.01^2)$ | 浅层网络 |
| **Xavier/Glorot** | $\mathcal{U}(-\sqrt{6/(n_{in}+n_{out})}, +\cdot)$ | Sigmoid, Tanh |
| **He/Kaiming** | $\mathcal{N}(0, 2/n_{in})$ | ReLU |

---

## 11. 优化器 (Optimizers)

> **注**：课程讲义提到梯度下降，以下为系统性补充。

### 11.1 随机梯度下降 (SGD)

**批量梯度下降 (Batch GD)**：每次用全部训练数据计算梯度。

$$\mathbf{w} \leftarrow \mathbf{w} - \eta \frac{1}{m}\sum_{i=1}^m \nabla_\mathbf{w} L_i$$

**随机梯度下降 (SGD)**：每次用单个样本（或 mini-batch）计算梯度。

$$\mathbf{w} \leftarrow \mathbf{w} - \eta \nabla_\mathbf{w} L_i$$

| 方法 | 稳定性 | 速度 | 内存 |
|------|--------|------|------|
| Batch GD | 稳定 | 慢（大数据集） | 需全部数据 |
| SGD | 噪声大 | 快 | 低 |
| Mini-batch SGD | 折中 | 折中 | 折中 |

### 11.2 Momentum（动量）

为 SGD 添加"惯性"，积累历史梯度方向：

$$\mathbf{v} \leftarrow \beta \mathbf{v} + \nabla_\mathbf{w} L$$
$$\mathbf{w} \leftarrow \mathbf{w} - \eta \mathbf{v}$$

**直觉**：像在碗中滚动的球，惯性能帮助越过小的局部极小值，并在一致方向上加速。

### 11.3 Adam（自适应矩估计）

Adam 结合了 Momentum 和 RMSProp，为每个参数自适应调整学习率：

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) \nabla L \quad \text{（一阶矩：梯度均值）}$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) (\nabla L)^2 \quad \text{（二阶矩：梯度方差）}$$
$$\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t} \quad \text{（偏差修正）}$$
$$\mathbf{w} \leftarrow \mathbf{w} - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon}\hat{m}_t$$

**默认参数**：$\beta_1 = 0.9$，$\beta_2 = 0.999$，$\epsilon = 10^{-8}$

**优点**：对大多数任务开箱即用，无需精细调参学习率。

### 11.4 优化器对比

| 优化器 | 学习率 | 内存开销 | 适用场景 |
|--------|-------|---------|---------|
| **SGD** | 固定 | 最小 | 需要最优最终性能 |
| **SGD + Momentum** | 固定 | $O(n_\text{params})$ | 图像分类 |
| **Adam** | 自适应 | $2 \times O(n_\text{params})$ | 通用首选 |
| **AdaGrad** | 自适应（单调减） | $O(n_\text{params})$ | 稀疏特征 |
| **RMSProp** | 自适应 | $O(n_\text{params})$ | RNN |

---

## 12. 正则化技术 (Regularization)

### 12.1 L1 / L2 正则化

**L2 正则化（Weight Decay）**：在损失函数中添加参数平方和惩罚：

$$J_\text{reg} = J + \frac{\lambda}{2}\|\mathbf{w}\|^2$$

**L1 正则化**：在损失函数中添加参数绝对值和惩罚（产生稀疏解）：

$$J_\text{reg} = J + \lambda\|\mathbf{w}\|_1$$

### 12.2 Dropout

**Dropout** 是神经网络特有的强正则化技术（Srivastava et al., 2014）：

**训练时**：在每次前向传播中，以概率 $p$（通常 $p=0.5$）随机将神经元输出置零：

$$\tilde{a}_i = m_i \cdot a_i, \quad m_i \sim \text{Bernoulli}(1-p)$$

**测试时**：使用所有神经元，但将权重乘以 $(1-p)$（期望值修正）。

**直觉**：
- 防止神经元之间的**共适应 (Co-adaptation)**——不能依赖特定神经元的存在
- 等价于对指数多个不同网络架构进行集成平均
- 增加了对噪声的鲁棒性

**实践建议**：全连接层后使用 Dropout（$p=0.5$）；卷积层一般不用或用较小的 $p$。

### 12.3 Batch Normalization（批归一化）

**Batch Normalization (BN)** 在每层激活前（或后）对每个 mini-batch 归一化：

$$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$

$$y_i = \gamma \hat{x}_i + \beta$$

其中 $\mu_B, \sigma_B^2$ 是 mini-batch 的均值和方差，$\gamma, \beta$ 是**可学习参数**。

**效果**：
- 减少**内部协变量偏移 (Internal Covariate Shift)**，每层输入分布更稳定
- 允许使用更大的学习率，加速训练
- 有轻微正则化效果（减轻了对 Dropout 的需求）
- 减少了对权重初始化的敏感性

### 12.4 Early Stopping（早停）

**思想**：在验证集性能停止提升时停止训练，防止过拟合。

**算法**：
1. 在每个 epoch 后评估验证集损失
2. 若验证集损失连续 $k$ 个 epoch 不下降（patience = $k$），停止训练
3. 返回验证集损失最低时的模型参数

**直觉**：训练误差持续下降，但验证误差在某点后开始上升（过拟合起始点）。最优点在验证误差最低处。

| 正则化方法 | 原理 | 适用场景 |
|----------|------|---------|
| **L2 正则化** | 抑制权重过大 | 通用 |
| **L1 正则化** | 促进稀疏权重 | 特征选择 |
| **Dropout** | 随机屏蔽神经元 | 全连接层 |
| **Batch Norm** | 归一化激活分布 | 深层网络 |
| **Early Stopping** | 监控验证集停止 | 通用 |

---

## 13. 通用近似定理与深层网络

### 13.1 通用近似定理 (Universal Approximation Theorem)

> **定理（Hornik et al., 1989）**：具有单个隐藏层、足够多神经元、非线性激活函数的前馈神经网络，可以以任意精度近似任意连续函数。

**意义**：单隐层 MLP（甚至使用 ReLU 这种"几乎线性"的激活）是**通用函数近似器**。

**重要提醒**：定理保证了近似**表示**的能力，但没有保证能**学习**到这样的网络（训练可能困难）。

### 13.2 为什么需要深层网络？

若单隐层已足够表示任意函数，为何要用深层网络？

**理由一：宽度 vs. 深度**
- 浅网络可能需要**指数多的神经元**才能表示深网络能轻松表示的函数
- 深层网络参数效率更高

**理由二：层次化特征学习**
- 深层网络自然学习**层次化特征**：低层学边缘，中层学形状，高层学语义概念
- 这与人类视觉系统的工作方式类似

**理由三：更好的泛化**
- 实验证明，在相同参数量下，深网络比浅网络泛化能力更强

**深度学习的直觉**：

| 层次 | 学到的特征（图像） |
|------|----------------|
| 第 1 层 | 边缘、颜色梯度 |
| 第 2 层 | 纹理、角点 |
| 第 3 层 | 局部形状（眼睛、轮子） |
| 第 4+ 层 | 对象部件、语义概念 |

### 13.3 深度线性网络

注意：

$$\mathbf{y} = \mathbf{W}^{(3)}\mathbf{W}^{(2)}\mathbf{W}^{(1)}\mathbf{x} \equiv \mathbf{W}'\mathbf{x}$$

**多层线性网络等价于单层线性网络**！深度本身不提供额外表达能力——**激活函数的非线性是关键**。

---

## 14. 深度神经网络与CNN简介

### 14.1 深层全连接网络的局限性

将全连接网络扩展到高维输入（如大图像）时，参数数量爆炸：
- $1000 \times 1000$ 图像，第一层 $1000$ 个神经元：参数数 = $10^9$！

**解决方案**：两种重要思想——稀疏连接 + 参数共享

### 14.2 稀疏连接 (Sparse Connection)

每个输出神经元只连接到输入的一个**局部区域（感受野 Receptive Field）**，而非所有输入。

- 每层感受野大小固定（如 $3\times3$）
- 随着网络深度增加，感受野**等效范围扩大**（深层神经元间接"看到"更大范围的输入）

### 14.3 参数共享 (Parameter Sharing / Tied Weights)

不同空间位置使用**相同的权重**（卷积核/滤波器）。

**参数量对比**：
- 全连接：$5 \times 5 = 25$ 个参数（连接到 5 个输入，每个位置独立）
- 卷积：仅 **3 个参数**（卷积核大小），在所有位置共享

**直觉**：图像中的特征（边缘、纹理）在不同位置出现，同一个检测器可以复用。

### 14.4 卷积神经网络（CNN）

卷积神经网络 (Convolutional Neural Networks, CNNs) 将**稀疏连接 + 参数共享**系统化，适合处理网格结构数据（图像、序列）。详细内容在下一讲介绍。

---

## 15. 补充资料

### 教材参考

- **Murphy "MLAPP"**：Chapter 16.3 — Feedforward neural networks; Chapter 16.4 — Backpropagation（含完整推导）
- **Bishop "PRML"**：Chapter 5 — Neural Networks（逐步推导前向传播和反向传播）；Chapter 5.3 — Error backpropagation
- **Burkov "Hundred-Page ML Book"**：Chapter 6 — Neural Networks and Deep Learning（简洁综述）

### 经典论文

- **反向传播**：Rumelhart, Hinton, Williams (1986), "Learning representations by back-propagating errors," *Nature*, 323, 533–536
- **通用近似定理**：Hornik, Stinchcombe, White (1989), "Multilayer feedforward networks are universal approximators," *Neural Networks*, 2(5), 359–366
- **Dropout**：Srivastava et al. (2014), "Dropout: A Simple Way to Prevent Neural Networks from Overfitting," *JMLR*, 15, 1929–1958
- **Batch Normalization**：Ioffe & Szegedy (2015), "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift," *ICML*
- **Adam**：Kingma & Ba (2015), "Adam: A Method for Stochastic Optimization," *ICLR*

### 在线资源

- Universal Approximation (Video 1): https://www.youtube.com/watch?v=KKT2VkTdFyc
- Why Deep Learning (Video 2): https://www.youtube.com/watch?v=FN8jclCrqY0
- Deep Learning Blog: https://medium.com/@jacklindsai/why-is-deep-learning-deep-d4305e596b77

### 关键公式速查

| 公式 | 内容 |
|------|------|
| $\mathbf{z}^{(l)} = \mathbf{W}^{(l)}\mathbf{a}^{(l-1)} + \mathbf{b}^{(l)}$ | 前向传播（预激活） |
| $\mathbf{a}^{(l)} = f(\mathbf{z}^{(l)})$ | 前向传播（激活） |
| $\frac{\partial L}{\partial z_j} = y_j(1-y_j)\frac{\partial L}{\partial y_j}$ | 反向传播（Sigmoid 层） |
| $\frac{\partial L}{\partial y_i} = \sum_j w_{ij}\frac{\partial L}{\partial z_j}$ | 反向传播（误差传递） |
| $\frac{\partial L}{\partial w_{ij}} = y_i \frac{\partial L}{\partial z_j}$ | 权重梯度 |
| $\text{ReLU}(z) = \max(0, z)$ | ReLU 激活函数 |
| $\sigma(z) = \frac{1}{1+e^{-z}}$ | Sigmoid 激活函数 |
| $\text{Softmax}(z_k) = \frac{e^{z_k}}{\sum_j e^{z_j}}$ | Softmax（多分类） |

### 学习路线建议

```
感知机 → 多层感知机 → 前向传播 → 反向传播
       → 激活函数选择 → 权重初始化 → 优化器
       → 正则化（Dropout, BN） → 深层网络 → CNN
```
