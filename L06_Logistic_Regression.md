# L06 Logistic Regression — DDA3020 Machine Learning

> **课程**: DDA3020 Machine Learning, CUHK-SZ  
> **讲师**: Baoyuan Wu  
> **日期**: February 2/4, 2026

---

## 目录 (Table of Contents)

1. [从回归到分类](#1-从回归到分类)
2. [Sigmoid 函数 / Logistic 函数](#2-sigmoid-函数--logistic-函数)
3. [Logistic Regression 模型定义](#3-logistic-regression-模型定义)
4. [决策边界 (Decision Boundary)](#4-决策边界-decision-boundary)
5. [损失函数：交叉熵 (Cross-Entropy)](#5-损失函数交叉熵-cross-entropy)
6. [MLE 推导](#6-mle-推导)
7. [梯度下降求解](#7-梯度下降求解)
8. [多分类：Softmax 回归](#8-多分类softmax-回归)
9. [过拟合与正则化](#9-过拟合与正则化)
10. [正则化逻辑回归 (Regularized Logistic Regression)](#10-正则化逻辑回归)
11. [概率视角下的正则化](#11-概率视角下的正则化)
12. [Linear Regression vs Logistic Regression 对比](#12-linear-regression-vs-logistic-regression-对比)
13. [补充资料](#13-补充资料)

---

## 1. 从回归到分类

### 1.1 分类任务的特点

> **分类 (Classification)**: 将输入数据划分到离散状态（类别）中。

**典型例子**:
- 邮件过滤：垃圾邮件 / 非垃圾邮件
- 天气预报：晴天 / 非晴天
- 肿瘤诊断：恶性 / 良性

**二分类设定**: 标签 $y \in \{0, 1\}$
- $y = 0$：负类 (negative class)，如非垃圾邮件、良性
- $y = 1$：正类 (positive class)，如垃圾邮件、恶性

### 1.2 直接用线性回归做分类的问题

**想法一**: 用线性假设 $f_w(x) = x^\top w$，设阈值 0.5，即：
- 若 $f_w(x) > 0.5$：预测 $y = 1$
- 若 $f_w(x) \leq 0.5$：预测 $y = 0$

**问题**: 
1. 若出现一个极端值（如很大的肿瘤）作为正样本，会导致线性拟合的斜率大幅改变，使原本正确的决策边界移位，将部分正样本误分类为负类。
2. 线性回归的输出 $f_w(x) \in (-\infty, +\infty)$，超出了 $[0, 1]$ 的范围——对于概率解释来说毫无意义。

> **需求**: 理想的假设函数应满足 $f_w(x) \in [0, 1]$，输出可解释为概率。

---

## 2. Sigmoid 函数 / Logistic 函数

### 2.1 定义

> **Sigmoid / Logistic 函数**:
> $$g(z) = \frac{1}{1 + e^{-z}}$$

**关键性质**:
- 输出范围: $g(z) \in (0, 1)$
- $g(0) = 0.5$
- 单调递增，关于 $(0, 0.5)$ 中心对称
- $g(z) \to 1$ 当 $z \to +\infty$
- $g(z) \to 0$ 当 $z \to -\infty$

### 2.2 Sigmoid 函数的导数

> $$g'(z) = g(z)(1 - g(z))$$

**推导**:
$$\frac{d}{dz} \frac{1}{1+e^{-z}} = \frac{e^{-z}}{(1+e^{-z})^2} = \frac{1}{1+e^{-z}} \cdot \frac{e^{-z}}{1+e^{-z}} = g(z)(1 - g(z))$$

这个性质在梯度计算中非常有用。

### 2.3 直觉

Sigmoid 将线性分数 $z = w^\top x$（从 $-\infty$ 到 $+\infty$）"压缩"到 $(0, 1)$，形成 S 形曲线。这样输出可以解释为概率。

---

## 3. Logistic Regression 模型定义

### 3.1 假设函数

$$f_w(x) = g(w^\top x) = \frac{1}{1 + e^{-w^\top x}}$$

其中 $w = [w_0, w_1, \ldots, w_d]^\top$，$x = [1, x_1, \ldots, x_d]^\top$（增广特征向量）。

> **概率解释**:
> $$f_w(x) = P(y = 1 \mid x; w)$$

即：$f_w(x)$ 是在给定特征 $x$ 和参数 $w$ 的条件下，$y = 1$ 的**估计概率**。

**例子**: 若 $f_w(x) = 0.8$，表示具有该特征的患者肿瘤为恶性的概率是 80%。

### 3.2 互补概率

由于 $P(y=0) + P(y=1) = 1$：
$$P(y = 0 \mid x; w) = 1 - f_w(x) = 1 - g(w^\top x)$$

---

## 4. 决策边界 (Decision Boundary)

### 4.1 二元分类规则

> **决策规则**:
> - 若 $f_w(x) \geq 0.5$（即 $w^\top x \geq 0$）：预测 $y = 1$
> - 若 $f_w(x) < 0.5$（即 $w^\top x < 0$）：预测 $y = 0$

**决策边界**: 方程 $w^\top x = 0$ 对应的超平面（或曲面）。

**为什么 $f_w(x) = 0.5$ 对应 $w^\top x = 0$?**
$$g(z) = 0.5 \iff z = 0 \iff w^\top x = 0$$

### 4.2 线性决策边界

**示例**: $f_w(x) = g(w_0 + w_1 x_1 + w_2 x_2) = g(-3 + x_1 + x_2)$

- 决策边界: $-3 + x_1 + x_2 = 0$，即 $x_1 + x_2 = 3$（直线）
- 预测 $y = 1$：若 $-3 + x_1 + x_2 \geq 0$

### 4.3 非线性决策边界

通过特征展开（类似多项式回归），可以产生非线性决策边界。

**示例**: $f_w(x) = g(w_0 + w_1 x_1 + w_2 x_2 + w_3 x_1^2 + w_4 x_2^2) = g(-1 + x_1^2 + x_2^2)$

- 决策边界: $-1 + x_1^2 + x_2^2 = 0$，即 $x_1^2 + x_2^2 = 1$（圆）
- 预测 $y = 1$：若 $x_1^2 + x_2^2 \geq 1$（圆外）

**直觉**: 决策边界的形状完全由 $w^\top x = 0$ 决定，是参数 $w$ 的线性函数，但通过特征展开可以实现任意复杂的非线性分界。

---

## 5. 损失函数：交叉熵 (Cross-Entropy)

### 5.1 为何不用 $\ell_2$ 损失？

若对 Logistic Regression 使用与线性回归相同的 $\ell_2$ 损失：
$$J(w) = \frac{1}{2m} \sum_{i=1}^m (g(w^\top x_i) - y_i)^2$$

由于 $g(\cdot)$ 是非线性函数，$J(w)$ 关于 $w$ 是**非凸的 (non-convex)**——存在多个局部极小值，梯度下降无法保证找到全局最优。

> **考试题型**: 证明 $\ell_2$ 损失对线性回归是凸的，但对逻辑回归是非凸的。

### 5.2 交叉熵 (Cross-Entropy)

> **交叉熵定义**:
> $$H(p, q) = -\sum_x p(x) \log q(x)$$

其中 $p(x)$ 是真实概率，$q(x)$ 是预测概率。

设：
- 真实后验: $y(x) = P(y=1 \mid x)$（真实标签）
- 预测后验: $f_w(x) = P(y=1 \mid x; w)$

**单样本交叉熵损失**:
$$\text{cost}(y(x), f_w(x)) = H(y(x), f_w(x))$$
$$= -P(y=1 \mid x) \log P(y=1 \mid x; w) - P(y=0 \mid x) \log P(y=0 \mid x; w)$$

对于 $y \in \{0, 1\}$（确定标签）：

$$\text{cost}(y, f_w(x)) = \begin{cases} -\log(f_w(x)), & \text{若 } y = 1 \\ -\log(1 - f_w(x)), & \text{若 } y = 0 \end{cases}$$

### 5.3 交叉熵损失的直觉

**当 $y = 1$ 时**:
- 若 $f_w(x) = 1$（预测正确）：损失 $= -\log(1) = 0$ ✓
- 若 $f_w(x) \to 0$（预测极错误）：损失 $= -\log(0) \to \infty$ ✗

**当 $y = 0$ 时**:
- 若 $f_w(x) = 0$（预测正确）：损失 $= -\log(1) = 0$ ✓
- 若 $f_w(x) \to 1$（预测极错误）：损失 $= -\log(0) \to \infty$ ✗

**直觉**: 对预测置信度越高但越错误的情况，惩罚越严重（趋向无穷大）。

### 5.4 整体损失函数

将两个分支合并为统一形式：

> **逻辑回归代价函数 (Log Loss / Binary Cross-Entropy)**:
> $$J(w) = -\frac{1}{m} \sum_{i=1}^m \left[y_i \log(f_w(x_i)) + (1 - y_i) \log(1 - f_w(x_i))\right]$$

**验证**:
- 当 $y_i = 1$：第二项消失，得 $-\log(f_w(x_i))$
- 当 $y_i = 0$：第一项消失，得 $-\log(1 - f_w(x_i))$

> **重要性质**: $J(w)$ 关于 $w$ 是**凸函数**，梯度下降可以找到全局最优。

---

## 6. MLE 推导

### 6.1 概率模型

Logistic Regression 对二分类做如下概率假设：

$$\mu(x; w) = \text{Sigmoid}(w^\top x), \quad y(x; w) \sim \text{Bernoulli}(\mu(x; w))$$

则：
$$P(y \mid x; w) = \begin{cases} \mu & \text{若 } y = 1 \\ 1 - \mu & \text{若 } y = 0 \end{cases}$$

可紧凑地写为：
$$P(y \mid x; w) = \mu^y (1-\mu)^{1-y}$$

### 6.2 对数似然函数

$$\log L(w) = \sum_{i=1}^m \log P(y_i \mid x_i; w)$$
$$= \sum_{i=1}^m \left[y_i \log \mu_i + (1 - y_i) \log(1 - \mu_i)\right]$$

因此：

> $$\max_w L(w) \equiv \min_w J(w)$$

**结论**: **最大化对数似然等价于最小化交叉熵损失**。这是逻辑回归使用交叉熵的理论依据。

---

## 7. 梯度下降求解

### 7.1 梯度计算

对 $J(w) = -\frac{1}{m} \sum_{i=1}^m [y_i \log f_w(x_i) + (1-y_i) \log(1-f_w(x_i))]$ 求梯度。

**关键推导步骤**:

首先计算 $\frac{\partial}{\partial w_j} \log f_w(x)$：

$$\frac{\partial}{\partial w_j} \log g(w^\top x) = \frac{1}{g(w^\top x)} \cdot g(w^\top x)(1 - g(w^\top x)) \cdot x_j = (1 - f_w(x)) x_j$$

类似地：$\frac{\partial}{\partial w_j} \log(1 - g(w^\top x)) = -f_w(x) x_j$

因此：

$$\frac{\partial J(w)}{\partial w_j} = -\frac{1}{m} \sum_{i=1}^m \left[y_i (1-f_w(x_i)) x_{ij} - (1-y_i) f_w(x_i) x_{ij}\right]$$

$$= -\frac{1}{m} \sum_{i=1}^m (y_i - f_w(x_i)) x_{ij}$$

$$= \frac{1}{m} \sum_{i=1}^m (f_w(x_i) - y_i) x_{ij}$$

**向量形式**:

> $$\nabla_w J(w) = \frac{1}{m} \sum_{i=1}^m (f_w(x_i) - y_i) x_i = \frac{1}{m} X^\top (f_w(X) - y)$$

### 7.2 更新规则

> **梯度下降更新**:
> $$w \leftarrow w - \alpha \nabla_w J(w) = w - \frac{\alpha}{m} \sum_{i=1}^m [f_w(x_i) - y_i] x_i$$

### 7.3 与线性回归的惊人相似性

| | 线性回归 | 逻辑回归 |
|-|---------|---------|
| 假设函数 $f_w(x)$ | $w^\top x$ | $g(w^\top x)$ |
| 梯度 $\nabla_w J(w)$ | $\frac{1}{m}\sum_i(f_w(x_i) - y_i)x_i$ | $\frac{1}{m}\sum_i(f_w(x_i) - y_i)x_i$ |
| 更新规则形式 | $w \leftarrow w - \alpha \nabla_w J$ | $w \leftarrow w - \alpha \nabla_w J$ |

**梯度形式完全相同！** 区别只在于 $f_w(x_i)$ 的计算方式不同（线性 vs Sigmoid）。这不是巧合，背后有 GLM（广义线性模型）理论支撑。

### 7.4 收敛监控

**正确做法**: 绘制**交叉熵损失** $J(w) = -\frac{1}{m}\sum_i [y_i \log f_w(x_i) + (1-y_i)\log(1-f_w(x_i))]$ 关于迭代次数的曲线，确保每次迭代都在减小。

**错误做法**: 用 $\ell_2$ 损失 $\sum_i (y_i - f_w(x_i))^2$ 监控（这不是逻辑回归的目标函数）。

---

## 8. 多分类：Softmax 回归

### 8.1 多分类问题

- **二分类**: $y \in \{0, 1\}$（如前述）
- **多分类**: $y \in \{1, \ldots, C\}$（$C$ 个类别）

**实际应用**:
- 天气预报: 晴天/多云/雨天/雪天
- 邮件标签: 工作/朋友/家庭/爱好

### 8.2 One-vs-All (OvA) 方法

**思路**: 对每个类 $j$ 训练一个二元逻辑回归 $f_{w_j}(\cdot)$，将其他所有类视为负类。

**预测**: $\hat{y} = \arg\max_j f_{w_j}(x)$

**优点**: 实现简单。  
**缺点**: 训练代价高，难以扩展到大量类别场景；各分类器独立训练，概率不归一化。

### 8.3 Softmax 回归（Multinomial Logistic Regression）

**Softmax 函数**:

> $$f_{w_j}^{(j)}(x) = \frac{\exp(w_j^\top x)}{\sum_{c=1}^C \exp(w_c^\top x)} = P(y = j \mid x; W)$$

其中 $W = [w_1, \ldots, w_C] \in \mathbb{R}^{(d+1) \times C}$，$x = [1, x_1, \ldots, x_d]^\top$。

**性质**:
- $f^{(j)}_W(x) \in (0, 1)$
- $\sum_{j=1}^C f^{(j)}_W(x) = 1$（概率归一化）
- 当 $C = 2$ 时，Softmax 退化为 Sigmoid

### 8.4 Softmax 代价函数

$$J(W) = -\frac{1}{m} \sum_{i=1}^m \sum_{j=1}^C \left[\mathbb{1}(y_i = j) \log f_{w_j}(x_i)\right]$$

其中 $\mathbb{1}(\cdot)$ 是指示函数（条件为真时为 1，否则为 0）。

**直觉**: 只对真实类别 $y_i$ 对应的对数概率惩罚。等同于多类交叉熵。

### 8.5 梯度与更新

**梯度推导结果**（类似于二分类）:

$$\frac{\partial J(W)}{\partial w_j} = \frac{1}{m} \sum_{i=1}^m (f_{w_j}(x_i) - \mathbb{1}(y_i = j)) x_i$$

**梯度下降更新**:
$$w_j \leftarrow w_j - \alpha \frac{\partial J(W)}{\partial w_j}$$

> **注意**: $\{w_c\}_{c=1}^C$ 应**并行**更新，而不是顺序更新，因为所有 $w_c$ 都出现在 Softmax 分母中，相互影响。

---

## 9. 过拟合与正则化

### 9.1 什么是过拟合

**过拟合 (Overfitting)**: 当特征过多（模型过于复杂），学习到的假设对训练数据拟合极好（低 bias），但对新样本泛化能力差（高 variance）。

**欠拟合 (Underfitting)**: 模型太简单，连训练数据都没拟合好。

**对应**:
- 欠拟合 → 高 bias（偏差）
- 过拟合 → 高 variance（方差）
- 好的模型 → bias-variance 平衡

### 9.2 解决过拟合的方法

1. **减少特征数**:
   - 手动特征选择
   - 维度规约（PCA 等，后续讲解）

2. **正则化**:
   - 保留所有特征，但限制每个参数的量级
   - 每个特征只贡献一点点

---

## 10. 正则化逻辑回归

### 10.1 L2 正则化逻辑回归（Ridge）

> $$\bar{J}(w) = J(w) + \frac{\lambda}{2m} \sum_{j=1}^d w_j^2$$

$$= -\frac{1}{m}\sum_{i=1}^m[y_i \log f_w(x_i) + (1-y_i)\log(1-f_w(x_i))] + \frac{\lambda}{2m}\sum_{j=1}^d w_j^2$$

**注意**: 偏置项 $w_0$ **不正则化**。

### 10.2 梯度下降更新（带正则化）

$$w_0 \leftarrow w_0 - \frac{\alpha}{m} \sum_{i=1}^m (f_w(x_i) - y_i) \cdot x_i^{(0)}, \quad x_i^{(0)} = 1$$

$$w_j \leftarrow w_j - \frac{\alpha}{m} \left[\sum_{i=1}^m (f_w(x_i) - y_i) \cdot x_i^{(j)} + \lambda w_j\right], \quad j = 1, \ldots, d$$

等价地（合并）:
$$w_j \leftarrow w_j \left(1 - \frac{\alpha \lambda}{m}\right) - \frac{\alpha}{m} \sum_{i=1}^m (f_w(x_i) - y_i) x_i^{(j)}$$

这里 $\left(1 - \frac{\alpha \lambda}{m}\right) < 1$，正则化项相当于每步都将 $w_j$ 缩小一点（权重衰减效果）。

### 10.3 L1 正则化逻辑回归（Lasso）

$$\bar{J}(w) = J(w) + \frac{\lambda}{2m} \sum_{j=1}^d |w_j|$$

与 Ridge 类似，但产生稀疏解（自动特征选择）。

---

## 11. 概率视角下的正则化

### 11.1 L2 正则化 ↔ 高斯先验

假设 $w \sim \mathcal{N}(w \mid 0, \sigma^2 I)$（排除偏置 $w_0$），MAP 估计：

$$\max_w L(w) + \log \mathcal{N}(w \mid 0, \sigma^2 I) \equiv \min_w J(w) + \frac{\lambda}{2m} \sum_{j=1}^d w_j^2$$

### 11.2 L1 正则化 ↔ 拉普拉斯先验

假设 $w \sim \text{Laplace}(w \mid 0, b)$（排除偏置 $w_0$），MAP 估计：

$$\max_w L(w) + \log \text{Laplace}(w \mid 0, b) \equiv \min_w J(w) + \frac{\lambda}{2m} \sum_{j=1}^d |w_j|$$

**总结**:

| 正则化方式 | 对应先验 | 特点 |
|-----------|---------|------|
| L2（Ridge）| Gaussian $\mathcal{N}(0, \sigma^2 I)$ | 小而非零参数 |
| L1（Lasso）| Laplace$(0, b)$ | 稀疏参数（特征选择） |

---

## 12. Linear Regression vs Logistic Regression 对比

| 特性 | 线性回归 (Linear Regression) | 逻辑回归 (Logistic Regression) |
|------|------------------------------|-------------------------------|
| **任务** | 回归 (Regression) | 分类 (Classification) |
| **假设函数** $f_w(x)$ | $w^\top x \in (-\infty, +\infty)$ | $g(w^\top x) \in [0, 1]$ |
| **目标函数** $J(w)$ | $\frac{1}{2m}\sum_i (y_i - w^\top x_i)^2$ | $-\frac{1}{m}\sum_i [y_i \log f_w + (1-y_i)\log(1-f_w)]$ |
| **损失函数** | $\ell_2$（均方误差） | 交叉熵（Binary Cross-Entropy） |
| **概率假设** | $y \mid x \sim \mathcal{N}(w^\top x, \sigma^2)$ | $y \mid x \sim \text{Bernoulli}(\mu(x;w))$ |
| **求解方法** | 解析解 或 梯度下降 | 梯度下降（无解析解） |
| **正则化 L2** | Ridge: $(X^\top X + \lambda I)^{-1} X^\top y$ | 加入 $\frac{\lambda}{2m}\sum w_j^2$ |
| **正则化 L1** | Lasso | 加入 $\frac{\lambda}{2m}\sum |w_j|$ |
| **输出解释** | 预测值 | 预测为正类的概率 |

> **重要说明**: 对每种线性回归/逻辑回归的变体，都可以从**确定性视角**（最小化某种损失函数）和**概率视角**（MLE 或 MAP）两个角度推导，两者等价。

**本节的延伸**: 线性回归和逻辑回归都是**广义线性模型 (Generalized Linear Models, GLM)** 的特例。感兴趣可参考 Bishop PRML Ch. 4。

---

## 常见错误与考试注意点

1. **Sigmoid 的导数**: $g'(z) = g(z)(1 - g(z))$，这是推导逻辑回归梯度的关键。
2. **交叉熵的凸性**: $J(w)$ 对逻辑回归是凸的，但 $\ell_2$ 损失对逻辑回归是非凸的。考试可能要求证明。
3. **决策边界**: 决策边界是 $w^\top x = 0$ 的集合，它是一个超平面（线性边界）；通过特征展开可以得到非线性边界，但模型本身仍是"线性"的（对参数线性）。
4. **梯度形式与线性回归相同**: 这是因为逻辑回归属于 GLM，其梯度总是 $(f_w(x_i) - y_i) x_i$ 的形式。
5. **Softmax 归一化**: $\sum_j f^{(j)}_W(x) = 1$，这保证输出是概率分布。
6. **不对偏置正则化**: $w_0$ 不参与 L1/L2 正则化惩罚，因为偏置只影响决策边界的位移，不影响模型复杂度。
7. **Softmax 并行更新**: 更新所有 $w_c$ 时要用**相同的旧参数值**计算梯度，不能顺序更新。
8. **监控梯度下降收敛**: 应绘制**正确的损失函数**（交叉熵）随迭代次数的变化曲线。

---

## 13. 补充资料

### 教材参考

- **Murphy "Machine Learning: A Probabilistic Perspective"**  
  - Ch. 8: 逻辑回归（全面覆盖 MLE、MAP、正则化、Softmax）  
  - Ch. 9.3: 广义线性模型（GLM）  

- **Bishop "Pattern Recognition and Machine Learning" (PRML)**  
  - Ch. 4: 线性分类模型（Logistic Regression、Softmax、LDA 对比）  
  - Ch. 4.2: 概率判别式模型  
  - Ch. 4.3: 迭代重加权最小二乘（IRLS，逻辑回归的牛顿法求解）  

- **Burkov "The Hundred-Page Machine Learning Book"**  
  - Ch. 5: 基本算法（逻辑回归部分）  

### 关键公式速查

> **Sigmoid 函数**:
> $$g(z) = \frac{1}{1+e^{-z}}, \quad g'(z) = g(z)(1-g(z))$$

> **二分类交叉熵损失**:
> $$J(w) = -\frac{1}{m}\sum_{i=1}^m [y_i \log f_w(x_i) + (1-y_i)\log(1-f_w(x_i))]$$

> **梯度（与线性回归同形）**:
> $$\nabla_w J(w) = \frac{1}{m}\sum_{i=1}^m (f_w(x_i) - y_i) x_i$$

> **Softmax 函数**:
> $$P(y=j \mid x; W) = \frac{e^{w_j^\top x}}{\sum_{c=1}^C e^{w_c^\top x}}$$

> **正则化 L2 目标函数**:
> $$\bar{J}(w) = J(w) + \frac{\lambda}{2m}\sum_{j=1}^d w_j^2$$

### 在线资源

- **Andrew Ng CS229 课程笔记**: [Logistic Regression](http://cs229.stanford.edu/notes/) (Notes 1)
- **Bishop PRML 官方网站**: [Cambridge PRML](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf)
- **Generalized Linear Models**: Bishop PRML Section 4, Murphy Ch. 9

### 编程实践建议

- 使用 `sklearn.linear_model.LogisticRegression` 实践（包含 L1/L2 正则化选项）
- 从头实现梯度下降版逻辑回归：理解收敛曲线、学习率影响
- 可视化决策边界：对比线性特征和多项式特征展开的效果
- 用 Softmax 在 MNIST 或 CIFAR-10 做多分类实验
