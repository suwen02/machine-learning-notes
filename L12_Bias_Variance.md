# L12: Over/Under-Fitting and Bias-Variance Trade-off

**课程**: DDA3020 Machine Learning, CUHK-SZ  
**主讲**: Baoyuan Wu  
**日期**: April 1, 2026

---

## 目录 (Table of Contents)

1. [学习目标与动机](#1-学习目标与动机)
2. [过拟合与欠拟合](#2-过拟合与欠拟合)
   - 2.1 [欠拟合 (Underfitting)](#21-欠拟合-underfitting)
   - 2.2 [过拟合 (Overfitting)](#22-过拟合-overfitting)
3. [模型复杂度与泛化误差](#3-模型复杂度与泛化误差)
   - 3.1 [多项式回归不同阶数的比较](#31-多项式回归不同阶数的比较)
   - 3.2 [二维情形的欠拟合与过拟合](#32-二维情形的欠拟合与过拟合)
4. [Bias-Variance Trade-off：实验观察](#4-bias-variance-trade-off实验观察)
5. [Bias-Variance Trade-off：统计分析](#5-bias-variance-trade-off统计分析)
   - 5.1 [问题建模](#51-问题建模)
   - 5.2 [期望测试误差的定义](#52-期望测试误差的定义)
   - 5.3 [分解推导（完整数学推导）](#53-分解推导完整数学推导)
   - 5.4 [三项的含义](#54-三项的含义)
6. [Bias、Variance与模型复杂度的关系](#6-biasvariance与模型复杂度的关系)
7. [训练误差 vs 测试误差：学习曲线](#7-训练误差-vs-测试误差学习曲线)
   - 7.1 [学习曲线（随训练样本数变化）](#71-学习曲线随训练样本数变化)
   - 7.2 [Regime 1: 高方差（过拟合）](#72-regime-1-高方差过拟合)
   - 7.3 [Regime 2: 高偏差（欠拟合）](#73-regime-2-高偏差欠拟合)
8. [Ensemble方法如何降低Bias/Variance](#8-ensemble方法如何降低biasvariance)
   - 8.1 [Decision Trees 的 Bias/Variance](#81-decision-trees-的-biasvariance)
   - 8.2 [Random Forests 降低 Variance](#82-random-forests-降低-variance)
   - 8.3 [Boosting 降低 Bias](#83-boosting-降低-bias)
9. [正则化方法回顾](#9-正则化方法回顾)
10. [交叉验证与模型选择](#10-交叉验证与模型选择)
11. [综合习题解析](#11-综合习题解析)
12. [总结：偏差-方差框架的实践指导](#12-总结偏差-方差框架的实践指导)
13. [补充资料 (Supplementary Resources)](#13-补充资料-supplementary-resources)

---

## 1. 学习目标与动机

### 核心问题

机器学习的终极目标是**预测 (Prediction)**，即：用训练数据学到的模型，在未见过的测试数据上表现良好。

训练好的模型用于预测：
- **线性模型**: $f_{\hat{w}, b}(X_\text{new}) = X_\text{new} \hat{w}$
- **多项式模型**: $f_{\hat{w}, b}(X_\text{new}) = P_\text{new} \hat{w}$

但一个在训练集上完美拟合的模型，不一定在测试集上表现好。如何理解和权衡这种矛盾？

> **Bias-Variance Trade-off 是机器学习理论的核心之一**：它从统计角度精确刻画了模型泛化能力的来源，指导我们选择合适复杂度的模型。

---

## 2. 过拟合与欠拟合

### 2.1 欠拟合 (Underfitting)

**定义**: 模型对训练数据的预测效果也不好，无法捕捉数据的基本规律。

**原因**:
1. 模型过于简单（如用线性模型拟合非线性数据）
2. 特征信息量不足（特征工程不够）

**识别**:
- 训练误差高
- 训练误差和测试误差都高，且接近

**解决方案**:
- 使用更复杂的模型（提升多项式阶数、增加神经网络层数）
- 添加更多信息量丰富的特征
- 减少正则化强度

**示例（1维回归）**:  
对于非线性数据，使用线性回归（1阶多项式）拟合：
- 拟合曲线是一条直线，明显无法描述数据的曲线趋势
- 残差（红色误差线）很大
- 训练集和测试集误差均高

### 2.2 过拟合 (Overfitting)

**定义**: 模型在训练数据上表现极好，但在测试数据上表现很差。

**原因**:
1. 模型过于复杂（相对于数据量）
2. 训练样本数量少，而特征维度高
3. 训练时间过长（拟合了噪声）

**识别**:
- 训练误差极低（接近0）
- 测试误差显著高于训练误差

**解决方案**:
- 增加训练数据
- 降低模型复杂度
- 正则化（L1、L2、Dropout等）
- 早停（Early Stopping）

**示例（1维回归）**:  
对同样数据使用9阶多项式拟合：
- 曲线穿过（几乎）所有训练点，训练误差接近0
- 但曲线极度扭曲，对测试点预测失真
- 模型"记住了"训练数据（包括噪声），而非学到真实规律

---

## 3. 模型复杂度与泛化误差

### 3.1 多项式回归不同阶数的比较

对同一数据集，比较不同阶数多项式的拟合效果：

| 模型 | 训练误差 | 测试误差 | 问题 |
|------|----------|----------|------|
| 1阶（线性） | 高 | 高 | 欠拟合 |
| 2阶 | 中等 | 较低 | 较好拟合 |
| 4阶 | 低 | 最低 | 最优拟合 |
| 9阶 | 极低（≈0） | 高 | 过拟合 |

**规律**:
- 训练误差随模型复杂度**单调递减**（越复杂越能拟合训练集）
- 测试误差**先降后升**，存在最优点

### 3.2 二维情形的欠拟合与过拟合

对于2维分类任务（两类数据点）：

- **欠拟合（Underfit）**: 决策边界过于简单（如线性分界），无法区分两类
- **过拟合（Overfit）**: 决策边界极度复杂，在训练点间蜿蜒，对新数据泛化差
- **良好拟合（Good Fit）**: 决策边界形状合适，既能拟合训练数据又能泛化

> **直觉**: 机器学习的目标不是最小化训练误差（这很容易——只要模型足够复杂就行），而是最小化**泛化误差**（在未见数据上的误差）。

---

## 4. Bias-Variance Trade-off：实验观察

### 多次随机采样实验

从同一个数据分布中**多次随机采样**训练集，对每次采样训练同一类型的模型：

#### 欠拟合情形（低阶多项式）

多次试验中：
- 每次拟合的曲线**形状相似**（low variance，不同训练集得到差不多的模型）
- 但所有曲线都**偏离真实函数**（high bias，存在系统性误差）

#### 过拟合情形（高阶多项式）

多次试验中：
- 每次拟合的曲线**差异很大**（high variance，对训练集采样非常敏感）
- 不同训练集训练出完全不同形状的曲线

### 实验总结（来自讲义）

> 当进行大量实验后观察到：
> 1. 训练误差随模型复杂度增加而单调递减（趋向零）
> 2. 测试误差：
>    - **低复杂度**: 测试误差高，**高bias，低variance**（欠拟合）
>    - **高复杂度**: 测试误差先降后升，**低bias，高variance**（过拟合）
>    - 在某个复杂度之后测试误差不再下降
> 3. 这种现象就是 **Bias-Variance Trade-off**

---

## 5. Bias-Variance Trade-off：统计分析

### 5.1 问题建模

设训练数据集 $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^n$ 独立同分布地从分布 $P(X, Y)$ 采样而来。

**真实数据生成过程**:

$$y = t(x) + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2)$$
$$p(y | x) = \mathcal{N}(t(x), \sigma^2)$$

其中 $t(x)$ 是未知的真实目标函数（$y$ 的条件期望），$\epsilon$ 是不可消除的随机噪声。

**学习目标**: 使用学习算法 $A$（如线性回归、SVM），从训练集 $\mathcal{D}$ 中学出假设函数：

$$h_\mathcal{D} = A(\mathcal{D})$$

注意：不同的训练集 $\mathcal{D}$ 会产生不同的 $h_\mathcal{D}$（这正是 Variance 的来源）。

### 5.2 期望测试误差的定义

**期望假设函数**（在所有可能的训练集上取期望）：

$$\bar{h}(x) = \mathbb{E}_{\mathcal{D} \sim P^n}[h_\mathcal{D}(x)] = \int_\mathcal{D} h_\mathcal{D}(x) \, p(\mathcal{D}) \, d\mathcal{D}$$

**给定 $h_\mathcal{D}$ 时的期望测试误差**（对测试点 $(x, y) \sim P$ 求期望）：

$$\mathbb{E}_{(x,y) \sim P}\left[(h_\mathcal{D}(x) - y)^2\right]$$

**给定算法 $A$ 时的期望测试误差**（同时对测试点和训练集求期望）：

$$\mathbb{E}_{(x,y), \mathcal{D}}\left[(h_\mathcal{D}(x) - y)^2\right] = \int_\mathcal{D} \int_x \int_y (h_\mathcal{D}(x) - y)^2 \, p(x,y) \, p(\mathcal{D}) \, dx \, dy \, d\mathcal{D}$$

这个量刻画了学习算法 $A$ 相对于数据分布 $P(X,Y)$ 的整体性能。

### 5.3 分解推导（完整数学推导）

**目标**: 证明期望测试误差可以分解为 Variance + Bias² + Noise。

#### 第一步：引入 $\bar{h}(x)$，分解期望误差

$$\mathbb{E}_{(x,y), \mathcal{D}}\left[(h_\mathcal{D}(x) - y)^2\right]$$

添加和减去 $\bar{h}(x)$：

$$= \mathbb{E}\left[((h_\mathcal{D}(x) - \bar{h}(x)) + (\bar{h}(x) - y))^2\right]$$

展开平方：

$$= \mathbb{E}\left[(h_\mathcal{D}(x) - \bar{h}(x))^2\right] + 2\mathbb{E}\left[(h_\mathcal{D}(x) - \bar{h}(x))(\bar{h}(x) - y)\right] + \mathbb{E}\left[(\bar{h}(x) - y)^2\right]$$

**证明交叉项为零**:

$$\mathbb{E}_{(x,y), \mathcal{D}}\left[(h_\mathcal{D}(x) - \bar{h}(x))(\bar{h}(x) - y)\right]$$
$$= \mathbb{E}_{(x,y)}\left[\mathbb{E}_\mathcal{D}[h_\mathcal{D}(x) - \bar{h}(x)] \cdot (\bar{h}(x) - y)\right]$$
$$= \mathbb{E}_{(x,y)}\left[(\mathbb{E}_\mathcal{D}[h_\mathcal{D}(x)] - \bar{h}(x)) \cdot (\bar{h}(x) - y)\right]$$
$$= \mathbb{E}_{(x,y)}\left[0 \cdot (\bar{h}(x) - y)\right] = 0$$

因此：

$$\mathbb{E}\left[(h_\mathcal{D}(x) - y)^2\right] = \underbrace{\mathbb{E}\left[(h_\mathcal{D}(x) - \bar{h}(x))^2\right]}_{\text{Variance}} + \mathbb{E}\left[(\bar{h}(x) - y)^2\right]$$

#### 第二步：进一步分解第二项（引入 $t(x)$）

$$\mathbb{E}_{(x,y)}\left[(\bar{h}(x) - y)^2\right] = \mathbb{E}_{(x,y)}\left[((\bar{h}(x) - t(x)) + (t(x) - y))^2\right]$$

展开：

$$= \mathbb{E}_{(x,y)}\left[(t(x) - y)^2\right] + \mathbb{E}_{(x,y)}\left[(\bar{h}(x) - t(x))^2\right] + 2\mathbb{E}_{(x,y)}\left[(t(x) - y)(\bar{h}(x) - t(x))\right]$$

**证明新的交叉项为零**:

$$\mathbb{E}_{(x,y)}\left[(t(x) - y)(\bar{h}(x) - t(x))\right]$$
$$= \mathbb{E}_x\left[\mathbb{E}_{y|x}\left[(t(x) - y)(\bar{h}(x) - t(x))\right]\right]$$
$$= \mathbb{E}_x\left[(t(x) - \mathbb{E}_{y|x}[y]) \cdot (\bar{h}(x) - t(x))\right]$$
$$= \mathbb{E}_x\left[(t(x) - t(x)) \cdot (\bar{h}(x) - t(x))\right] = 0$$

（利用了 $\mathbb{E}_{y|x}[y] = t(x)$）

#### 最终分解结果

> **Bias-Variance 分解定理**:
> $$\boxed{\mathbb{E}_{(x,y), \mathcal{D}}\left[(h_\mathcal{D}(x) - y)^2\right] = \underbrace{\mathbb{E}_{\mathcal{D}}\left[(h_\mathcal{D}(x) - \bar{h}(x))^2\right]}_{\text{Variance（方差）}} + \underbrace{\mathbb{E}_x\left[(\bar{h}(x) - t(x))^2\right]}_{\text{Bias}^2(\text{偏差}^2)} + \underbrace{\mathbb{E}_{(x,y)}\left[(t(x) - y)^2\right]}_{\text{Noise（噪声）}}}$$

### 5.4 三项的含义

#### Variance（方差）

$$\text{Var} = \mathbb{E}_{\mathcal{D}}\left[(h_\mathcal{D}(x) - \bar{h}(x))^2\right]$$

> **含义**: 衡量你的分类器/回归器对不同训练集的**敏感程度**（即"过专化"程度）。

- 方差来源于**固定的训练集**（你碰巧抽到了哪些训练样本）
- 高方差 ↔ 过拟合：换一个训练集，模型会有很大差异
- 复杂模型（如深度决策树、高阶多项式）倾向于高方差

#### Bias²（偏差的平方）

$$\text{Bias}^2 = \mathbb{E}_x\left[(\bar{h}(x) - t(x))^2\right]$$

> **含义**: 即使有无限多训练数据，你的分类器/回归器依然存在的**固有误差**。

- 偏差来源于模型本身的**假设局限性**（模型归纳偏置）
- 高偏差 ↔ 欠拟合：模型无法表达真实函数形式
- 简单模型（如线性模型、浅决策树）倾向于高偏差

#### Noise（噪声）

$$\text{Noise} = \mathbb{E}_{(x,y)}\left[(t(x) - y)^2\right] = \sigma^2$$

> **含义**: 数据本身的内在噪声，是**不可消除**的误差下界。

- 噪声来源于数据分布的随机性和特征表示的模糊性
- 无论多好的模型，也无法突破噪声下界
- 对应数据生成过程中的 $\epsilon \sim \mathcal{N}(0, \sigma^2)$

**总结**:

| 误差来源 | 来源 | 可减少？ | 解决方向 |
|----------|------|----------|----------|
| Variance | 训练集的随机性 | 是 | 增加数据量、降低复杂度、正则化 |
| Bias² | 模型假设限制 | 是 | 增加模型复杂度、更好的特征 |
| Noise | 数据本身 | 否 | 不可消除 |

---

## 6. Bias、Variance与模型复杂度的关系

随模型复杂度增加（以多项式阶数为例）：

> **关键规律**:
> - **Variance**: 随模型复杂度**增加**（不同训练集训练的模型差异越来越大）
> - **Bias²**: 随模型复杂度**减少**（更复杂的模型平均预测更接近真实函数）
> - **总误差**: 先减后增，在某个复杂度处取得最小值

**曲线图描述**:
```
误差
 │   
 │  ╲ Total Error
 │   ╲         /
 │    ╲       /
 │  __ ╲___/ 
 │ /bias²  ╲
 │/         ╲ variance
 └────────────────────── 模型复杂度
    欠拟合区域 | 最优点 | 过拟合区域
```

**结论**: 应当选择**适中复杂度**的模型，使总期望误差（Bias² + Variance + Noise）最小化。

**更多例子：决策树**

| 决策树类型 | Bias | Variance | 拟合状态 |
|------------|------|----------|----------|
| 单棵剪枝树（浅） | 高 | 低 | 欠拟合 |
| 单棵深树（不剪枝） | 低 | 高 | 过拟合 |

---

## 7. 训练误差 vs 测试误差：学习曲线

### 7.1 学习曲线（随训练样本数变化）

对于**固定模型**，随着训练样本数增加，训练误差和测试误差的变化：

**文字描述**:
```
误差
│
│─── 训练误差（随样本增加上升，因为更难全部拟合）
│
│
│                         ─────── 测试误差（随样本增加下降）
│
│                    交汇区域
└─────────────────────────────── 训练样本数 n
```

- 训练样本极少时：训练误差极低（模型可以完美记住），测试误差极高
- 随样本增加：训练误差上升，测试误差下降，最终趋于稳定

根据两条曲线的相对位置，可以判断是 High Variance 还是 High Bias。

### 7.2 Regime 1: 高方差（过拟合）

**症状**:
- **训练误差** 远低于 测试误差（两者差距大）
- 训练误差低于可接受阈值 $\epsilon$
- 测试误差高于 $\epsilon$

**图形特征**: 两条曲线分开，差距大。增加训练数据后，两者会逐渐收敛（因为更多数据让过拟合更难发生）。

**解决方案**:
1. **增加训练样本数**（最根本的解决方案）
2. 降低模型复杂度（减少特征数、降低多项式阶数、减少网络层数/宽度）
3. 正则化（L1、L2、Dropout、Early Stopping）
4. 数据增强

### 7.3 Regime 2: 高偏差（欠拟合）

**症状**:
- **训练误差**本身就高于可接受阈值 $\epsilon$
- 训练误差和测试误差都高，且差距不大（两者接近）

**图形特征**: 两条曲线很快收敛但都在高处稳定。增加训练数据几乎无济于事（因为问题出在模型本身）。

**解决方案**:
1. 添加更多特征（提升特征信息量）
2. 使用更复杂的模型（核方法、非线性模型、增加网络容量）
3. 减少正则化强度（如降低 L2 惩罚系数）

---

## 8. Ensemble方法如何降低Bias/Variance

### 8.1 Decision Trees 的 Bias/Variance

单棵决策树展现出两种极端：
- **剪枝树（浅）**: 高 bias，低 variance → 欠拟合
- **完全树（深）**: 低 bias，高 variance → 过拟合

### 8.2 Random Forests 降低 Variance

**Random Forests**（随机森林）的核心思想：训练多棵随机化的决策树，对它们的预测取平均（回归）或投票（分类）。

**随机化来源**:
1. **Bootstrap 采样** (Bagging)：每棵树从训练集有放回地采样，各自训练集不同
2. **随机特征子集**：每次分裂时只从随机选择的特征子集中选最优分裂特征

**为什么能降低 Variance**?

假设训练了 $B$ 棵独立的树，每棵方差为 $\sigma^2$，则均值的方差为 $\sigma^2 / B$。

虽然随机森林中的树并非完全独立（使用了相同数据分布），但随机化使相关性降低，方差确实显著减小。

> **随机森林的 Bias-Variance 特性**:
> - Variance 显著降低（相比单棵深树）
> - Bias 不保证降低（各树复杂度与单树相当，平均后 bias 没有本质改变）

### 8.3 Boosting 降低 Bias

**Boosting**（如 AdaBoost、Gradient Boosting）逐步组合**弱学习器**（high bias，low variance），通过以下方式降低 bias：

- 每一轮重点关注上一轮预测错误的样本
- 后一棵树拟合前面所有树的**残差**
- 最终模型是弱学习器的加权组合

> **Boosting 的 Bias-Variance 特性**:
> - Bias 显著降低（集成后表达能力增强）
> - Variance 不保证降低，需要谨慎调参防过拟合

**Ensemble 方法总结**:

| 方法 | 主要效果 | 降低 | 适用场景 |
|------|----------|------|----------|
| Bagging (如 Random Forest) | 降低 Variance | Variance ↓ | 高方差模型（深树）|
| Boosting (如 XGBoost) | 降低 Bias | Bias ↓ | 高偏差弱学习器 |
| Stacking | 两者都可能改善 | 两者 | 多样化基学习器 |

---

## 9. 正则化方法回顾

正则化通过**限制模型复杂度**来降低过拟合（减小 Variance）：

### L2 正则化（Ridge / Weight Decay）

$$\min_w \frac{1}{n} \sum_{i=1}^n L(y_i, h_w(x_i)) + \lambda \|w\|_2^2$$

- 对权重施加均方惩罚，使权重趋向于小的值
- 等价于对权重的高斯先验 $p(w) \propto \exp(-\lambda \|w\|^2)$
- $\lambda$ 增大 → 正则化增强 → Bias↑，Variance↓

### L1 正则化（Lasso）

$$\min_w \frac{1}{n} \sum_{i=1}^n L(y_i, h_w(x_i)) + \lambda \|w\|_1$$

- 倾向于产生稀疏解（部分权重精确为0）
- 可用于特征选择

### Dropout（神经网络专用）

训练时随机将一部分神经元置零（概率 $p$），测试时对所有神经元的输出乘以 $(1-p)$。

- 相当于隐式地训练了指数级数量的子网络并取平均
- 防止神经元间的共适应（co-adaptation）

### Early Stopping（早停）

在验证集误差开始上升时停止训练，防止模型过度拟合训练集噪声。

> **直觉**: 正则化本质上是在增加 Bias 的同时减少 Variance，从而使总误差下降。选择合适的正则化强度 $\lambda$ 至关重要。

---

## 10. 交叉验证与模型选择

### 为什么需要交叉验证

直接在测试集上调参会导致**测试集污染**（信息泄露），使测试误差不再是真实泛化误差的无偏估计。

正确流程：
1. 用**训练集**训练模型
2. 用**验证集**选择超参数（如正则化系数 $\lambda$、多项式阶数）
3. 最终用**测试集**评估最终性能（只报告一次！）

### K-fold 交叉验证 (K-fold Cross-Validation)

**步骤**：
1. 将训练集均分为 $K$ 份（folds）
2. 进行 $K$ 轮，每轮以第 $k$ 份为验证集，其余 $K-1$ 份为训练集
3. 记录每轮的验证误差 $e_k$
4. 最终估计误差为 $K$ 轮的平均: $\bar{e} = \frac{1}{K} \sum_{k=1}^K e_k$

> **常用设置**: $K=5$ 或 $K=10$

**优点**:
- 充分利用数据（每个样本都被用作验证集恰好一次）
- 估计稳定性优于单次 train/val 划分

**缺点**:
- 计算代价是单次划分的 $K$ 倍

### Leave-One-Out CV (LOOCV)

$K$-fold 的极端情况，令 $K = n$（训练集大小）：
- 每轮留出 1 个样本作为验证集
- 最低偏差（几乎用全量数据训练），但计算代价极高

**适用场景**: 数据集极小（如医学研究中样本极少）

### 模型选择流程

```
候选模型集合（如不同阶数的多项式，不同 λ 的 Ridge 回归）
    ↓
对每个候选模型，进行 K-fold CV 估计泛化误差
    ↓
选择 CV 误差最小的模型/超参数
    ↓
在全部训练集上重新训练该模型
    ↓
在（单次使用的）测试集上评估最终性能
```

---

## 11. 综合习题解析

以下是讲义中的课堂练习，通过具体数值加深对三个分解项的理解。

### 题目设定

**数据生成**:
- 真实函数: $y = t(x) + \epsilon$，$\epsilon \in \mathcal{N}(0, \sigma^2)$，$\sigma^2 = 0.5$
- 从同一分布中随机采样 10 次训练集，各自训练同类型的回归模型

**测试样本**: $(x, y) = (5, 10)$，其中 $t(x=5) = 9.5$，$\epsilon = 0.5$（实例化）

**10 个模型的预测**: $\hat{y}_1, \ldots, \hat{y}_{10} = 9, 11, 23, 6, 8, 12, 10, 4, 13, 7$

### 计算过程

**Step 1: 平均预测（期望假设函数）**

$$\bar{h}(x=5) = \frac{9 + 11 + 23 + 6 + 8 + 12 + 10 + 4 + 13 + 7}{10} = \frac{103}{10} = 10.3$$

**Step 2: Bias²**

$$\text{Bias}^2 = (\bar{h}(x) - t(x))^2 = (10.3 - 9.5)^2 = 0.8^2 = 0.64$$

**Step 3: Variance**

$$\text{Var} = \frac{1}{10} \sum_{i=1}^{10} (\hat{y}_i - \bar{h})^2$$

$$= \frac{1}{10}\left[(9-10.3)^2 + (11-10.3)^2 + (23-10.3)^2 + (6-10.3)^2 + (8-10.3)^2 \right.$$
$$\left.+ (12-10.3)^2 + (10-10.3)^2 + (4-10.3)^2 + (13-10.3)^2 + (7-10.3)^2\right]$$

$$= \frac{1}{10}\left[1.69 + 0.49 + 161.29 + 18.49 + 5.29 + 2.89 + 0.09 + 39.69 + 7.29 + 10.89\right]$$

$$= \frac{248.1}{10} = 24.81$$

**Step 4: Noise**

$$\text{Noise} = \sigma^2 = 0.5$$

（注：这里用的是分布的方差，而非实例化后的具体 $\epsilon = 0.5$）

**Step 5: 经验 MSE（作为验证）**

$$\widehat{\text{MSE}} = \frac{1}{10}\sum_{i=1}^{10}(\hat{y}_i - y)^2 = \frac{1}{10}\sum_{i=1}^{10}(\hat{y}_i - 10)^2$$

$$= \frac{1}{10}\left[1 + 1 + 169 + 16 + 4 + 4 + 0 + 36 + 9 + 9\right] = \frac{249}{10} = 24.9$$

**验证**:

$$\text{Bias}^2 + \text{Var} + \text{Noise} = 0.64 + 24.81 + 0.5 = 25.95$$

$$\approx \widehat{\text{MSE}} = 24.9 \quad \checkmark$$

（由于是经验估计，数值略有差异）

> **注意**: 本例中 Variance (24.81) 远大于 Bias² (0.64)，说明该模型处于**高方差（过拟合）**状态，主要误差来源是模型对训练集采样过于敏感。

---

## 12. 总结：偏差-方差框架的实践指导

### 诊断问题的框架

```
观察训练误差 vs 测试误差
        │
   ┌────┴────┐
   │         │
两者都高   训练误差低，测试误差高
   │         │
 High Bias  High Variance
(欠拟合)   (过拟合)
   │         │
增加模型   增加数据/
复杂度     降低复杂度/
添加特征   正则化
```

### 核心要点

1. **期望测试误差 = Variance + Bias² + Noise**（不可消除的噪声是下界）

2. **Bias-Variance 权衡**:
   - 增大模型复杂度：Bias↓，Variance↑
   - 增大训练数据量：Variance↓，Bias不变
   - 增强正则化：Bias↑，Variance↓

3. **实践建议**:
   - 首先确认是 high bias 还是 high variance（观察学习曲线）
   - High variance → 正则化、更多数据、降低复杂度
   - High bias → 更复杂模型、更好特征、减少正则化
   - 用交叉验证系统地选择超参数

4. **Ensemble 方法**:
   - Bagging（随机森林）→ 降低 Variance
   - Boosting（XGBoost）→ 降低 Bias

---

## 13. 补充资料 (Supplementary Resources)

### 教材参考

1. **Murphy** - *Machine Learning: A Probabilistic Perspective* (MLAPP, 2012)
   - Chapter 6: Frequentist Statistics — Bias-Variance tradeoff 的严格统计推导
   - Chapter 7: Linear Regression — 正则化与模型选择

2. **Bishop** - *Pattern Recognition and Machine Learning* (PRML, 2006)
   - Chapter 3.2: The Bias-Variance Decomposition — 经典推导（本讲最直接的参考）
   - Chapter 1.3: Model Selection — 过拟合与交叉验证

3. **Goodfellow, Bengio, Courville** - *Deep Learning* (2016)
   - Chapter 5: Machine Learning Basics — Section 5.4 on Bias, Variance, and the tradeoffs
   - 在线免费: https://www.deeplearningbook.org/

### 在线资源

- **Cornell CS4780 课程笔记** (Kilian Weinberger):
  https://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/
  （本讲主要参考来源之一）

- **Bias-Variance 可视化** (Scott Fortmann-Roe):
  http://scott.fortmann-roe.com/docs/BiasVariance.html
  （直觉可视化工具，强烈推荐）

- **Understanding the Bias-Variance Tradeoff** (Seema Singh):
  https://towardsdatascience.com/understanding-the-bias-variance-tradeoff-165e6942b229

### 延伸阅读

- **双下降现象 (Double Descent)**: 在深度学习中，当模型极度过参数化时，测试误差可能在过拟合区域之后再次下降，挑战了传统的 Bias-Variance 理论框架。参考:
  - Belkin et al., 2019: https://arxiv.org/abs/1812.11118

- **交叉验证实现**:
  - Scikit-learn: `sklearn.model_selection.KFold`, `cross_val_score`
  - 文档: https://scikit-learn.org/stable/modules/cross_validation.html

- **Bootstrap 与 Bagging**:
  - Breiman (1996): https://link.springer.com/article/10.1023/A:1018054314350

---

*笔记整理自 DDA3020 L12 讲义 (Baoyuan Wu, CUHK-SZ, April 1, 2026)*  
*数学推导参考: Cornell CS4780 讲义 & Bishop PRML Chapter 3*
