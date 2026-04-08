# L10: Convolutional Neural Networks (CNN)

**课程**: DDA3020 Machine Learning, CUHK-SZ  
**主讲**: Baoyuan Wu  
**日期**: March 25, 2026

---

## 目录 (Table of Contents)

1. [CNN的历史回顾](#1-cnn的历史回顾)
2. [为什么需要CNN：全连接网络的局限](#2-为什么需要cnn全连接网络的局限)
3. [CNN总体结构预览](#3-cnn总体结构预览)
4. [卷积层 (Convolutional Layer)](#4-卷积层-convolutional-layer)
   - 4.1 [卷积运算的直觉](#41-卷积运算的直觉)
   - 4.2 [Stride（步幅）与 Padding（填充）](#42-stride步幅与-padding填充)
   - 4.3 [输出尺寸计算公式](#43-输出尺寸计算公式)
   - 4.4 [多通道卷积](#44-多通道卷积)
   - 4.5 [参数量计算](#45-参数量计算)
   - 4.6 [卷积层超参数汇总](#46-卷积层超参数汇总)
5. [池化层 (Pooling Layer)](#5-池化层-pooling-layer)
6. [全连接层 (Fully Connected Layer)](#6-全连接层-fully-connected-layer)
7. [CNN完整架构](#7-cnn完整架构)
8. [经典CNN架构](#8-经典cnn架构)
   - 8.1 [LeNet-5](#81-lenet-5)
   - 8.2 [AlexNet](#82-alexnet)
   - 8.3 [VGGNet](#83-vggnet)
   - 8.4 [GoogLeNet / Inception](#84-googlenet--inception)
   - 8.5 [ResNet（残差网络）](#85-resnet残差网络)
9. [CNN用于图像分类的Pipeline](#9-cnn用于图像分类的pipeline)
10. [Transfer Learning（迁移学习）](#10-transfer-learning迁移学习)
11. [Data Augmentation（数据增强）](#11-data-augmentation数据增强)
12. [计算机视觉中CNN的广泛应用](#12-计算机视觉中cnn的广泛应用)
13. [补充资料 (Supplementary Resources)](#13-补充资料-supplementary-resources)

---

## 1. CNN的历史回顾

### 时间线

CNN 的发展历程标志着深度学习从理论走向实践的重要里程碑：

- **1998年**：LeCun、Bottou、Bengio 和 Haffner 发表 *"Gradient-based learning applied to document recognition"*，提出了 LeNet，首次成功将 CNN 应用于手写数字识别（MNIST）。
- **2006年**：Hinton 和 Salakhutdinov 发表深度置信网络论文，重新激发了学界对 **Deep Learning** 的兴趣。
- **2012年**：Krizhevsky、Sutskever 和 Hinton 发表 AlexNet (*"ImageNet Classification with Deep Convolutional Neural Networks"*)，在 ImageNet 竞赛中以巨大优势夺冠，标志着深度 CNN 时代的开始。

> **直觉解释**: 传统的图像特征需要人工设计（如 SIFT、HOG），而 CNN 能够从数据中自动学习层次化的特征表示：低层学习边缘、纹理，高层学习语义概念。

---

## 2. 为什么需要CNN：全连接网络的局限

### 全连接网络的问题

回顾上一讲的神经网络：
- **线性模型**: $f(\mathbf{x}) = W\mathbf{x}$
- **两层神经网络**: $f(\mathbf{x}) = W_2 \max(0, W_1\mathbf{x})$

对于一张 $32 \times 32 \times 3$ 的彩色图像，如果展平为向量，输入维度高达 $32 \times 32 \times 3 = 3072$。用全连接层处理图像有以下问题：

| 问题 | 说明 |
|------|------|
| **参数量爆炸** | 若第一层有 1000 个神经元，仅一层就需要 $3072 \times 1000 = 3.07M$ 个参数 |
| **忽略空间结构** | 将图像展平会丢失像素间的空间位置关系 |
| **无平移不变性** | 图像中同一物体在不同位置出现，全连接网络需要重新学习 |
| **过拟合风险高** | 参数过多，泛化能力差 |

### CNN的核心思想

CNN 通过以下机制克服上述问题：
1. **局部连接 (Local Connectivity)**: 每个神经元只与输入的一个局部区域相连
2. **权值共享 (Weight Sharing)**: 同一个滤波器在整个输入上滑动，大幅减少参数
3. **空间层次结构**: 通过堆叠卷积层，逐层提取更抽象的特征

---

## 3. CNN总体结构预览

CNN 是一系列卷积层与激活函数的堆叠：

```
输入图像 → [CONV → ReLU] → [CONV → ReLU] → ... → POOL → FC → 输出
```

**示例结构**（处理 $32 \times 32 \times 3$ 图像）：
- 第1层: CONV + ReLU，使用 6 个 $5 \times 5 \times 3$ 滤波器 → 输出 $28 \times 28 \times 6$
- 第2层: CONV + ReLU，使用 10 个 $5 \times 5 \times 6$ 滤波器 → 输出 $24 \times 24 \times 10$
- 更多 CONV + ReLU 层...
- POOL 层（下采样）
- FC 层（分类输出）

> **直觉**: CNN 就像一个自动特征提取器 + 分类器，前面的卷积层提取特征，后面的全连接层进行分类。

---

## 4. 卷积层 (Convolutional Layer)

### 4.1 卷积运算的直觉

**输入**: $W_1 \times H_1 \times C$ 的三维张量（宽 × 高 × 通道数）

卷积层保留图像的空间结构，而不是将图像展平。对于 $32 \times 32 \times 3$ 的图像，使用一个 $5 \times 5 \times 3$ 的滤波器：

1. 滤波器始终覆盖输入的**完整深度**（通道数必须匹配）
2. 将滤波器在输入的空间维度上滑动
3. 在每个位置计算滤波器与对应输入区域的**点积**（dot product）

> **关键公式**（单个位置的输出）:
> $$\text{output}_{i,j} = \mathbf{w}^T \mathbf{x}_{i,j} + b$$
> 其中 $\mathbf{w}$ 是滤波器权重向量（展平后），$\mathbf{x}_{i,j}$ 是对应输入区域（展平后），$b$ 是偏置。

**例子**:
- 输入 $32 \times 32 \times 3$，一个 $5 \times 5 \times 3$ 滤波器
- 每次点积的维度: $5 \times 5 \times 3 = 75$ 维 + 1 个 bias
- 在整个图像上滑动后，得到一个 $28 \times 28 \times 1$ 的 **activation map**（激活图）

**多个滤波器**:
- 使用 6 个 $5 \times 5 \times 3$ 滤波器 → 得到 6 个激活图 → 堆叠成 $28 \times 28 \times 6$ 的输出体积

> **直觉**: 每个滤波器相当于一个"特征检测器"，学习检测某种特定模式（如水平边缘、垂直边缘、颜色梯度等）。

### 4.2 Stride（步幅）与 Padding（填充）

#### Stride（步幅）

Stride 控制滤波器每次滑动的步长：

| Stride | 对 $7 \times 7$ 输入使用 $3 \times 3$ 滤波器 |
|--------|----------------------------------------------|
| $S=1$ | 输出 $5 \times 5$ |
| $S=2$ | 输出 $3 \times 3$ |
| $S=3$ | **不合法**！$(7-3)/3 = 1.33$，无法整除 |

#### Padding（填充）

在实际中，常在输入边界补零（zero padding）以控制输出尺寸：

- 输入 $7 \times 7$，$3 \times 3$ 滤波器，stride=1，**pad=1**
- 填充后等效输入变为 $9 \times 9$
- 输出尺寸: $(7 + 2 \cdot 1 - 3)/1 + 1 = 7 \times 7$（**尺寸不变**！）

**保持尺寸不变的通用规则**: 对于 $F \times F$ 滤波器，使用 padding $P = (F-1)/2$：
- $F=3 \Rightarrow P=1$
- $F=5 \Rightarrow P=2$  
- $F=7 \Rightarrow P=3$

### 4.3 输出尺寸计算公式

> **核心公式**（输出空间尺寸）:
> $$W_2 = \frac{W_1 - F + 2P}{S} + 1$$
> $$H_2 = \frac{H_1 - F + 2P}{S} + 1$$

其中：
- $W_1, H_1$: 输入宽度和高度
- $F$: 滤波器尺寸（$F \times F$）
- $P$: 零填充量
- $S$: Stride

**具体示例**:
- 输入 $32 \times 32 \times 3$
- 10个 $5 \times 5 \times 3$ 滤波器，stride=1，pad=2
- 输出尺寸: $\frac{32 + 2 \times 2 - 5}{1} + 1 = 32$
- 最终输出体积: $32 \times 32 \times 10$

### 4.4 多通道卷积

当输入有 $C$ 个通道时，滤波器的深度也必须是 $C$：

```
输入: W × H × C
滤波器: F × F × C  (深度与输入相同)
一个滤波器的输出: W' × H' × 1  (单个激活图)
K个滤波器的输出: W' × H' × K  (K个激活图堆叠)
```

**架构图示** (文字描述):
```
[输入体积 W×H×C] → 滤波器1(F×F×C) → 激活图1(W'×H'×1)
                 → 滤波器2(F×F×C) → 激活图2(W'×H'×1)
                 ...
                 → 滤波器K(F×F×C) → 激活图K(W'×H'×1)
                                    ↓ 堆叠
                              输出体积(W'×H'×K)
```

### 4.5 参数量计算

> **参数量公式**:
> $$\text{参数总量} = F^2 \cdot C \cdot K + K$$
> 即：每个滤波器有 $F^2 \cdot C$ 个权重 + 1 个偏置，共 $K$ 个滤波器。

**示例计算**:
- 输入 $32 \times 32 \times 3$，使用 10 个 $5 \times 5 \times 3$ 滤波器
- 每个滤波器参数: $5 \times 5 \times 3 + 1 = 76$（包含偏置）
- 总参数量: $76 \times 10 = 760$

**对比全连接层**:
- 若用全连接层连接 $32 \times 32 \times 3 = 3072$ 个输入到 $28 \times 28 = 784$ 个输出神经元
- 参数量: $3072 \times 784 = 2,408,448$（约 240 万！）
- CNN 只需 760 个参数 → **权值共享的巨大优势**

### 4.6 卷积层超参数汇总

| 超参数 | 说明 | 常用设置 |
|--------|------|----------|
| $K$（滤波器数量） | 输出通道数 | 32, 64, 128, 256, 512（2的幂次） |
| $F$（滤波器尺寸） | 空间大小 | $3 \times 3$（最常见）, $5 \times 5$, $1 \times 1$ |
| $S$（Stride） | 滑动步长 | 1（最常见）, 2 |
| $P$（Padding） | 零填充量 | $(F-1)/2$（保持尺寸）|

**常见组合**:
- $F=3, S=1, P=1$ → 保持空间尺寸
- $F=5, S=1, P=2$ → 保持空间尺寸
- $F=1, S=1, P=0$ → 跨通道线性变换（改变深度不改变空间尺寸）
- $F=5, S=2, P=?$ → 下采样

---

## 5. 池化层 (Pooling Layer)

### 动机

在深层 CNN 中，如果只用卷积层：
- $32 \to 28 \to 24 \to \ldots$（缩小太慢，计算代价极高）
- 需要一种机制来**快速降低空间尺寸**，同时保留重要信息

### 池化操作

池化层**独立作用于每个激活图**，不改变深度（通道数）。

#### Max Pooling（最大池化）

最常用的池化方式。在每个局部窗口内取最大值：

```
输入区域 (2×2):
1  3
2  4
→ Max Pooling → 4
```

**直觉**: 保留局部区域中最强的激活，相当于"这个区域有没有这个特征"。具有一定的**平移不变性**。

#### Average Pooling（平均池化）

取局部窗口内的平均值。在某些场合（如 GoogLeNet 最后的全局平均池化）使用。

### 池化层参数

池化层有两个超参数：
- **窗口尺寸** $F$（常用 $2 \times 2$）
- **Stride** $S$（常用 $S=2$，即不重叠）

> **输出尺寸公式**（与卷积层相同）:
> $$W_2 = \frac{W_1 - F}{S} + 1$$
> 注意：池化层通常**不使用 padding**。

**典型设置**: $F=2, S=2$ → 空间尺寸减半，深度不变（如 $224 \times 224 \times 64 \to 112 \times 112 \times 64$）

### 池化层的特点

- **无参数可学习**（不像卷积层有权重）
- 提供**局部平移不变性**
- 减小表示尺寸 → 减少计算量 → 控制过拟合

---

## 6. 全连接层 (Fully Connected Layer)

全连接层（FC Layer）位于 CNN 的末端，将卷积/池化层提取的特征用于最终分类。

- 输入：将前一层的 3D 特征图展平（flatten）为 1D 向量
- 与普通多层感知机（MLP）完全相同
- 每个神经元连接到前一层所有神经元

**作用**：
- 整合全局特征，做出最终预测
- 通常接 Softmax 层输出类别概率

> **注意**: FC 层参数量通常占 CNN 总参数量的大部分。例如 AlexNet 中，前5个卷积层约有 240 万参数，而后3个 FC 层约有 5800 万参数！

---

## 7. CNN完整架构

典型的 CNN 图像分类架构：

```
输入图像 (e.g., 224×224×3)
    ↓
[CONV → BN → ReLU] × N  (特征提取)
    ↓
POOL  (下采样)
    ↓
[CONV → BN → ReLU] × N
    ↓
POOL
    ↓
... (重复多次)
    ↓
Flatten (展平)
    ↓
FC → ReLU → Dropout
    ↓
FC → Softmax
    ↓
输出类别概率
```

**各层作用总结**：

| 层类型 | 主要作用 | 是否有可学习参数 |
|--------|----------|-----------------|
| CONV | 提取局部特征 | 是（权重、偏置） |
| BN (Batch Norm) | 归一化，加速训练 | 是（scale, shift） |
| ReLU | 非线性激活 | 否 |
| POOL | 下采样，平移不变性 | 否 |
| FC | 分类决策 | 是（权重、偏置） |
| Softmax | 输出概率 | 否 |

---

## 8. 经典CNN架构

### 8.1 LeNet-5

**年份**: 1998  
**作者**: LeCun, Bottou, Bengio, Haffner  
**论文**: "Gradient-based learning applied to document recognition"

**架构** (输入 $32 \times 32 \times 1$ 灰度图):

```
输入(32×32×1) → CONV(6×5×5) → AvgPool → CONV(16×5×5) → AvgPool → FC(120) → FC(84) → 输出(10)
```

| 层 | 输出尺寸 | 参数量 |
|----|----------|--------|
| 输入 | 32×32×1 | — |
| CONV1 (6个5×5) | 28×28×6 | 156 |
| AvgPool | 14×14×6 | — |
| CONV2 (16个5×5) | 10×10×16 | 2,416 |
| AvgPool | 5×5×16 | — |
| FC1 | 120 | 48,120 |
| FC2 | 84 | 10,164 |
| 输出 | 10 | 850 |

**意义**: 首次展示 CNN 在实际任务（手写数字识别）中优于传统方法，奠定了现代 CNN 的基本架构。

### 8.2 AlexNet

**年份**: 2012  
**作者**: Krizhevsky, Sutskever, Hinton  
**参数量**: ~60M  
**成就**: ImageNet 2012 冠军（top-5 错误率 15.3%，第二名 26.2%）

**关键创新**:
1. **ReLU 激活函数** → 解决了 sigmoid 的梯度消失问题，训练速度快 6 倍
2. **Dropout 正则化** → 防止过拟合
3. **数据增强** → 随机裁剪、水平翻转、颜色抖动
4. **GPU 并行训练** → 首次使用双 GPU 训练
5. **Local Response Normalization (LRN)** → 侧抑制（现已较少使用）

**架构** (输入 $227 \times 227 \times 3$):
```
CONV1(96, 11×11, S=4) → MaxPool → CONV2(256, 5×5, P=2) → MaxPool
→ CONV3(384, 3×3, P=1) → CONV4(384, 3×3, P=1) → CONV5(256, 3×3, P=1)
→ MaxPool → FC(4096) → FC(4096) → FC(1000) → Softmax
```

> **历史意义**: AlexNet 的成功标志着深度学习在计算机视觉中的决定性胜利，引发了整个 AI 领域的深度学习革命。

### 8.3 VGGNet

**年份**: 2014  
**作者**: Simonyan, Zisserman (Oxford Visual Geometry Group)  
**参数量**: ~138M  
**变体**: VGG-16, VGG-19

**核心思想**: 用多个小卷积核（$3 \times 3$）叠加代替大卷积核

> **关键洞察**: 两个 $3 \times 3$ 卷积层的感受野等价于一个 $5 \times 5$ 卷积层，但参数量更少且非线性更强：
> - 两个 $3 \times 3$ 层: $2 \times (3^2 \times C^2) = 18C^2$ 个参数
> - 一个 $5 \times 5$ 层: $5^2 \times C^2 = 25C^2$ 个参数

**VGG-16 架构**（输入 $224 \times 224 \times 3$）:

```
[CONV(64, 3×3)] × 2 → MaxPool
[CONV(128, 3×3)] × 2 → MaxPool
[CONV(256, 3×3)] × 3 → MaxPool
[CONV(512, 3×3)] × 3 → MaxPool
[CONV(512, 3×3)] × 3 → MaxPool
FC(4096) → FC(4096) → FC(1000) → Softmax
```

**特点**:
- 结构简单、规则（全用 $3 \times 3$ 卷积，通道数加倍）
- 可迁移性极强，至今仍是迁移学习的常用 backbone

### 8.4 GoogLeNet / Inception

**年份**: 2014  
**作者**: Szegedy et al. (Google)  
**参数量**: ~5M（比 AlexNet 小 12 倍！）  
**成就**: ImageNet 2014 冠军

**核心创新**: **Inception Module**

传统 CNN 每层只用一个尺寸的滤波器，Inception Module 在同一层并行使用多种尺寸的卷积：

```
         输入
    ┌────┬────┬────┬────┐
  1×1  3×3  5×5  3×3池化
    └────┴────┴────┴────┘
         拼接输出
```

> **直觉**: 让网络自己学习在每一层应该用多大尺寸的感受野，而不是人为固定。

**辅助技巧**:
- **$1 \times 1$ 卷积**: 跨通道降维，减少参数量（"Network in Network"思想）
- **全局平均池化**: 最后用 Global Average Pooling 代替 FC 层，大幅减少参数

### 8.5 ResNet（残差网络）

**年份**: 2015  
**作者**: He, Zhang, Ren, Sun (Microsoft Research)  
**参数量**: ResNet-50 约 25M  
**成就**: ImageNet 2015 冠军（top-5 错误率 3.57%）

**问题背景**: 随着网络层数增加，出现"退化问题"（degradation problem）—— 更深的网络在训练集上效果反而更差（不是过拟合，而是优化困难）。

**核心创新**: **残差连接 (Residual Connection / Skip Connection)**

> **残差块 (Residual Block) 公式**:
> $$\mathbf{y} = \mathcal{F}(\mathbf{x}, \{W_i\}) + \mathbf{x}$$
> 其中 $\mathcal{F}(\mathbf{x})$ 是需要学习的残差映射，$\mathbf{x}$ 是直接跳跃连接的 identity。

**文字架构图**:
```
输入 x
  ├─────────────── skip connection ──────────────┐
  ↓                                              |
CONV(F, 3×3) → BN → ReLU → CONV(F, 3×3) → BN → (+) → ReLU
                                                输出 y = F(x) + x
```

**为什么有效**？

1. **恒等映射容易学习**: 如果某层不需要变换，只需令 $\mathcal{F}(\mathbf{x}) \approx 0$，比学习 $\mathcal{H}(\mathbf{x}) = \mathbf{x}$ 更容易。
2. **梯度直接传播**: 跳跃连接提供了梯度的"高速公路"，缓解梯度消失。
3. **理论保证**: 残差网络的深层版本不会比浅层版本更差（最差情况下 $\mathcal{F} = 0$，等价于浅层网络）。

**ResNet 变体**:

| 变体 | 层数 | 特点 |
|------|------|------|
| ResNet-18 | 18 | 基础残差块 |
| ResNet-34 | 34 | 基础残差块 |
| ResNet-50 | 50 | Bottleneck 块（$1\times1 \to 3\times3 \to 1\times1$）|
| ResNet-101 | 101 | Bottleneck 块 |
| ResNet-152 | 152 | Bottleneck 块 |

**Bottleneck Block** 设计（ResNet-50+）：
```
输入(256ch)
    ↓ 1×1 CONV (64ch, 降维)
    ↓ 3×3 CONV (64ch)
    ↓ 1×1 CONV (256ch, 升维)
    + skip connection
输出(256ch)
```
参数效率更高：$1 \times 64 \times 256 + 9 \times 64 \times 64 + 1 \times 64 \times 256 = 69,632$，而若直接用 $3 \times 3$: $9 \times 256 \times 256 = 589,824$。

---

## 9. CNN用于图像分类的Pipeline

完整的图像分类流程：

```
原始图像
    ↓
数据预处理
  - 调整尺寸 (resize)
  - 零均值化 (zero-center): x -= mean_image
  - 归一化
    ↓
数据增强 (训练时)
  - 随机裁剪、翻转、颜色抖动等
    ↓
CNN特征提取
  - 多个 [CONV → BN → ReLU → Pool] 模块
    ↓
Flatten + FC层
    ↓
Softmax 输出类别概率
    ↓
损失函数 (Cross-Entropy Loss)
    ↓
反向传播 + 优化器更新权重 (SGD/Adam)
```

**CNN各层学到的特征**（可视化研究发现）：
- **第1层**: 边缘、颜色梯度等基础纹理
- **第2-3层**: 纹理、简单形状
- **第4-5层**: 物体部件（眼睛、轮子等）
- **最后层**: 完整物体语义

---

## 10. Transfer Learning（迁移学习）

### 动机

训练一个大型 CNN 需要：
- 百万级标注数据
- 数周的 GPU 计算时间

**迁移学习**利用在大型数据集（如 ImageNet，1.2M图像，1000类）上预训练好的模型，迁移到新任务上。

### 迁移学习策略

根据目标数据集大小和与源数据集的相似度：

| 情况 | 策略 |
|------|------|
| 数据量小 + 相似度高 | 只训练最后的 FC 层，冻结其余层 |
| 数据量小 + 相似度低 | 训练更高层的 FC，甚至微调部分卷积层 |
| 数据量大 + 相似度高 | Fine-tune 整个网络（较小学习率） |
| 数据量大 + 相似度低 | 从头训练或大幅 fine-tune |

### Fine-tuning（微调）步骤

1. 下载在 ImageNet 上预训练的 CNN（如 ResNet-50）
2. 替换最后的 FC 层（将输出维度改为目标类别数）
3. 使用较小的学习率（通常是原始学习率的 $1/10$）在目标数据集上继续训练
4. 可以选择冻结浅层（保留通用特征），只更新深层（学习特定特征）

> **直觉**: ImageNet 预训练的特征（边缘、纹理、形状）对大多数视觉任务都有用，迁移学习避免了从头学习这些通用特征的开销。

**教材参考**: Goodfellow et al., *Deep Learning*, Chapter 15 (Representation Learning)

---

## 11. Data Augmentation（数据增强）

### 动机

增加训练数据的多样性，防止过拟合，提升模型鲁棒性。

### 常用增强方法

**几何变换**:
- **随机水平翻转 (Random Horizontal Flip)**: 最简单、最常用，适合不含方向信息的图像
- **随机裁剪 (Random Crop)**: 在训练时随机裁剪，测试时中心裁剪（AlexNet 使用 $224 \times 224$ 从 $256 \times 256$ 裁剪）
- **随机旋转 (Random Rotation)**: 小角度旋转
- **随机缩放 (Random Scale)**

**颜色变换**:
- **颜色抖动 (Color Jitter)**: 随机改变亮度、对比度、饱和度
- **随机灰度化 (Random Grayscale)**

**高级方法**:
- **Cutout/Random Erasing**: 随机遮挡图像的矩形区域
- **MixUp**: 将两张图像线性混合：$\tilde{x} = \lambda x_i + (1-\lambda) x_j$
- **CutMix**: 将一张图像的区域替换为另一张图像的对应区域

> **注意**: 数据增强只在**训练时**使用，测试时通常只做确定性的中心裁剪和水平翻转（有时做多尺度测试取平均）。

---

## 12. 计算机视觉中CNN的广泛应用

CNN 不仅用于图像分类，还广泛应用于：

| 任务 | 说明 | 代表模型 |
|------|------|----------|
| **图像分类** (Classification) | 判断图像属于哪个类别 | ResNet, VGG, EfficientNet |
| **目标检测** (Detection) | 找出图像中所有物体的位置和类别 | YOLO, Faster R-CNN, SSD |
| **图像分割** (Segmentation) | 对每个像素进行类别标注 | FCN, U-Net, DeepLab |
| **图像检索** (Retrieval) | 以图搜图 | 特征嵌入 + 相似度搜索 |
| **人脸识别** (Face Recognition) | 识别人脸身份 | FaceNet, ArcFace |
| **图像生成** (Generation) | 生成真实感图像 | CNN+GAN |
| **医学影像** (Medical Imaging) | 疾病诊断 | 各类医学专用模型 |

---

## 13. 补充资料 (Supplementary Resources)

### 教材参考

1. **Goodfellow, Bengio, Courville** - *Deep Learning* (2016)
   - Chapter 9: Convolutional Networks — CNN 理论的权威介绍
   - 在线免费: https://www.deeplearningbook.org/

2. **Murphy** - *Machine Learning: A Probabilistic Perspective* (MLAPP, 2012)
   - Chapter 28: Deep Learning — 包含 CNN 的概率视角

3. **Bishop** - *Pattern Recognition and Machine Learning* (PRML, 2006)
   - Chapter 5: Neural Networks — 神经网络基础

### 在线课程

- **CS231n: Deep Learning for Computer Vision** (Stanford)
  - http://cs231n.stanford.edu/
  - 本讲幻灯片部分内容来源于此课程

### 深度学习框架

- **PyTorch**: https://pytorch.org/
  - `torch.nn.Conv2d`, `torch.nn.MaxPool2d`
- **TensorFlow/Keras**: https://www.tensorflow.org/
  - `tf.keras.layers.Conv2D`, `tf.keras.layers.MaxPool2D`

### 原始论文

- LeNet: LeCun et al., 1998 — http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf
- AlexNet: Krizhevsky et al., 2012 — https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html
- VGGNet: Simonyan & Zisserman, 2014 — https://arxiv.org/abs/1409.1556
- GoogLeNet: Szegedy et al., 2014 — https://arxiv.org/abs/1409.4842
- ResNet: He et al., 2015 — https://arxiv.org/abs/1512.03385

### 可视化工具

- CNN Explainer (交互式可视化): https://poloclub.github.io/cnn-explainer/
- Feature Visualization (Distill): https://distill.pub/2017/feature-visualization/

---

*笔记整理自 DDA3020 L10 讲义 (Baoyuan Wu, CUHK-SZ, March 25, 2026)*
