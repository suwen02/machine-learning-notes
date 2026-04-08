# DDA3020 机器学习 — Lecture 2: Probability and Information Theory（概率论与信息论）

> **课程**: DDA3020 Machine Learning, CUHK-SZ  
> **讲师**: Baoyuan Wu  
> **日期**: 2026年1月7/12日

---

## 目录 (Table of Contents)

1. [随机实验、样本空间与事件](#1-随机实验样本空间与事件)
2. [随机变量 (Random Variables)](#2-随机变量-random-variables)
3. [离散随机变量的概率](#3-离散随机变量的概率)
4. [贝叶斯定理 (Bayes' Theorem)](#4-贝叶斯定理-bayes-theorem)
5. [独立随机变量](#5-独立随机变量-independent-random-variables)
6. [期望与方差（离散）](#6-期望与方差离散-expectation--variance-discrete)
7. [连续随机变量](#7-连续随机变量-continuous-random-variables)
8. [期望与方差（连续）](#8-期望与方差连续)
9. [常见概率分布](#9-常见概率分布-common-distributions)
10. [最大似然估计 (MLE) 与最大后验估计 (MAP)](#10-最大似然估计-mle-与最大后验估计-map)
11. [信息论基础](#11-信息论基础-information-theory)
12. [补充资料](#12-补充资料-supplementary-resources)

---

## 1. 随机实验、样本空间与事件

### 1.1 随机实验 (Random Experiment)

**随机实验**是指具有随机结果的实验过程。通过其**程序**（procedure）和对结果的**观察**（observations）来描述。

**例子**: 掷硬币2次，观察每次朝上的面。

### 1.2 样本空间 (Sample Space) $S$

**样本空间**是随机实验所有可能结果的集合，记为 $S$。

$$S = \{(\text{Head, Head}),\ (\text{Head, Tail}),\ (\text{Tail, Head}),\ (\text{Tail, Tail})\}$$

### 1.3 事件 (Event)

**事件**是样本空间的子集，记为 $A \subset S$。

**例子**: 定义事件 $A$ 为"至少出现一次正面"：

$$A = \{(\text{Head, Head}),\ (\text{Head, Tail}),\ (\text{Tail, Head})\} \subset S$$

### 1.4 事件概率的基本公理

设事件 $A \subset S$，$B \subset S$，则概率需满足：

> $$P(A) \geq 0$$
> $$P(S) = 1$$
> $$\text{若 } A \cap B = \emptyset,\ \text{则 } P(A \cup B) = P(A) + P(B)$$
> $$\text{否则 } P(A \cup B) = P(A) + P(B) - P(A \cap B)$$

---

## 2. 随机变量 (Random Variables)

**随机变量**是从样本空间 $S$ 到实数空间 $\mathbb{R}$ 的实值函数：

$$X: S \to \mathbb{R}$$

**例子**: 以2次掷硬币中出现"尾面"的次数作为随机变量 $X$：

$$X((\text{H,H})) = 0, \quad X((\text{H,T})) = 1, \quad X((\text{T,H})) = 1, \quad X((\text{T,T})) = 2$$

$X$ 的**状态空间 (state space)** 为 $\mathcal{X} = \{0, 1, 2\}$。

### 随机变量的两种类型

| 类型 | 定义 | 例子 |
|------|------|------|
| **离散 (Discrete)** | 状态空间 $\mathcal{X}$ 是有限或可数无限集合 | 掷骰子点数、抛硬币结果 |
| **连续 (Continuous)** | 状态空间 $\mathcal{X}$ 是不可数集合（如实数区间） | 身高、温度、时间 |

> **直觉解释**: 随机变量是"将随机实验的结果数值化"的工具。在 ML 中，数据特征和标签都可以用随机变量建模，这使得我们可以用概率语言描述不确定性。

---

## 3. 离散随机变量的概率

### 3.1 概率质量函数 (PMF, Probability Mass Function)

离散随机变量 $X$ 取值 $x \in \mathcal{X}$ 的概率记为 $P(X = x)$，须满足：

> $$P(X = x) \geq 0, \quad \forall x \in \mathcal{X}$$
> $$\sum_{x \in \mathcal{X}} P(X = x) = 1$$

### 3.2 联合概率 (Joint Probability)

两个随机变量 $X$ 和 $Y$ 的**联合概率**：

$$P(X = x, Y = y) = P(X = x \mid Y = y)\,P(Y = y) = P(Y = y \mid X = x)\,P(X = x)$$

这就是**乘法法则 (Product Rule)**。

### 3.3 边际概率 (Marginal Probability)

通过对联合概率**求和消元（边际化，marginalization）**得到单变量的概率：

$$P(X = x) = \sum_{y \in \mathcal{Y}} P(X = x, Y = y) \quad \text{（对 }Y\text{ 求和）}$$

$$P(Y = y) = \sum_{x \in \mathcal{X}} P(X = x, Y = y) \quad \text{（对 }X\text{ 求和）}$$

这就是**加法法则 (Sum Rule)**。

### 3.4 条件概率 (Conditional Probability)

在已知 $Y = y$ 的条件下，$X$ 的概率：

$$P(X = x \mid Y = y) = \frac{P(X = x, Y = y)}{P(Y = y)}$$

$$P(Y = y \mid X = x) = \frac{P(X = x, Y = y)}{P(X = x)}$$

> **直觉解释**: 条件概率回答"在已知某事件发生的情况下，另一事件的概率是多少"。这是贝叶斯推断的核心。

---

## 4. 贝叶斯定理 (Bayes' Theorem)

### 4.1 贝叶斯公式推导

将条件概率的定义与乘法法则和加法法则结合，得到**贝叶斯定理**：

> $$P(X = x \mid Y = y) = \frac{P(X = x)\, P(Y = y \mid X = x)}{\sum_{x' \in \mathcal{X}} P(X = x')\, P(Y = y \mid X = x')}$$

或者简写为（使用 $P(Y)$ 作分母）：

$$P(X \mid Y) = \frac{P(X)\, P(Y \mid X)}{P(Y)}$$

### 4.2 贝叶斯定理的四个核心概念

| 术语 | 公式 | 含义 |
|------|------|------|
| **Prior Probability（先验概率）** | $P(Y)$ | 观察到证据之前对事件的初始信念 |
| **Likelihood（似然）** | $P(X \mid Y)$ | 在事件发生条件下，观察到证据的概率 |
| **Marginal Likelihood（边际似然）** | $P(X)$ | 证据出现的总概率 |
| **Posterior Probability（后验概率）** | $P(Y \mid X)$ | 观察到证据之后对事件的更新信念 |

> **直觉解释（因果视角）**:  
> 设 $Y$ 为"原因"（cause），$X$ 为"结果"（effect）。  
> - **Likelihood** $P(X|Y)$ 是已知原因、预测结果（正向因果）  
> - **Posterior** $P(Y|X)$ 是已知结果、反推原因（逆向推断）  
> 这正是 ML 中"从观测数据推断隐含参数/状态"的核心思路。

### 4.3 医疗诊断案例 (Bayes' Theorem 实例)

**问题设置**:
- $x = 1$: 检测结果阳性；$x = 0$: 阴性
- $y = 1$: 患有乳腺癌；$y = 0$: 未患癌

**已知数据**:
- 患癌情况下，检测阳性的概率（Likelihood）: $P(x=1 \mid y=1) = 0.8$
- 乳腺癌先验概率（Prior）: $P(y=1) = 0.13$（美国女性终生风险约13%）
- 假阳性率（False Positive Rate）: $P(x=1 \mid y=0) = 0.1$

**问题**: 检测阳性时，真正患癌的概率是多少？

**错误直觉**: 很多人会回答 $P(y=1 \mid x=1) = 0.8$，这是**错的**！忽略了先验概率。

**用贝叶斯公式计算**:

$$P(y=1 \mid x=1) = \frac{P(x=1 \mid y=1)\, P(y=1)}{P(x=1 \mid y=1)\, P(y=1) + P(x=1 \mid y=0)\, P(y=0)}$$

$$= \frac{0.8 \times 0.13}{0.8 \times 0.13 + 0.1 \times 0.87} = \frac{0.104}{0.104 + 0.087} \approx 0.5445$$

**结论**: 即使检测阳性，真正患癌的概率约为 **54.45%**，而非80%！  
这说明先验概率（baseline risk）在更新信念时至关重要。

> **在 ML 中的意义**: 贝叶斯定理是**贝叶斯分类器**、**朴素贝叶斯**、**MAP 估计**等方法的理论基础。它提供了一种在观测数据下更新参数信念（belief）的原则性方法。

---

## 5. 独立随机变量 (Independent Random Variables)

**定义**: 若 $X$ 与 $Y$ 独立（记为 $X \perp Y$），则：

> $$X \perp Y \iff P(X, Y) = P(X)\, P(Y)$$

**独立性的意义（参数数量）**:

假设 $X$ 有3个状态，$Y$ 有4个状态：
- **无独立假设**: 需要 $3 \times 4 - 1 = 11$ 个自由参数定义联合分布
- **有独立假设**: 只需 $(3-1) + (4-1) = 5$ 个自由参数

> **直觉解释**: 独立性假设极大地降低了概率模型的参数量，使建模更高效。ML 中的**朴素贝叶斯分类器**就是利用特征之间的条件独立假设来简化计算。

---

## 6. 期望与方差（离散）(Expectation & Variance, Discrete)

### 6.1 期望 (Expectation / Mean)

> $$\mathbb{E}[X] = \sum_{x \in \mathcal{X}} x\, P(X = x)$$

**函数的期望**:

$$\mathbb{E}[f(X)] = \sum_{x \in \mathcal{X}} f(x)\, P(X = x)$$

**矩 (Moments)**: $X$ 的 $k$ 阶矩定义为 $M_k = \mathbb{E}[X^k]$

### 6.2 方差 (Variance)

> $$\text{Var}(X) = \mathbb{E}[(X - \mathbb{E}[X])^2] = \mathbb{E}[X^2] - (\mathbb{E}[X])^2 = M_2 - M_1^2$$

**标准差 (Standard Deviation)**:

$$\text{Std} = \sqrt{\text{Var}(X)}$$

方差衡量随机变量围绕均值的**平均平方波动**。

> **在 ML 中的意义**: 期望是模型预测的"平均水平"，方差衡量预测的"稳定性"。Bias-Variance Trade-off（偏差-方差权衡，W9讲）正是利用这两个概念分析模型的泛化能力。

---

## 7. 连续随机变量 (Continuous Random Variables)

### 7.1 概率密度函数 (PDF, Probability Density Function)

若 $X$ 是连续随机变量，其**状态空间不可数**，因此 $P(X = x) = 0$ 对任意单点 $x$ 成立。

用**概率密度函数 (PDF)** $p_X(x)$ 描述连续概率：

> $$P(a < X < b) = \int_a^b p(x)\, dx$$
> $$P(a < X < a + dx) \approx p(a) \cdot dx$$

**累积分布函数 (CDF, Cumulative Distribution Function)**:

$$F_X(x) = P(X < x) = \int_{-\infty}^{x} p(s)\, ds$$

且 $p_X(x) = F'(x)$（PDF 是 CDF 的导数）。

### 7.2 二元连续分布：边际化、条件化与独立性

设 $X, Y$ 的联合 PDF 为 $p_{X,Y}(x, y)$，满足 $\int_x \int_y p(x,y)\, dx\, dy = 1$：

| 操作 | 公式 |
|------|------|
| **边际分布 (Marginal)** | $p(x) = \int_{-\infty}^{\infty} p(x, y)\, dy$ |
| **条件分布 (Conditional)** | $p(x \mid y) = \frac{p(x, y)}{p(y)}$ |
| **独立性 (Independence)** | $p_{X,Y}(x, y) = p_X(x)\, p_Y(y)$ |

> **注意**: 连续情况下 $P(Y=y) = 0$，条件概率 $p(x|y)$ 须通过无穷小事件推导，结果形式与离散情况一致。

---

## 8. 期望与方差（连续）

连续随机变量的期望和方差与离散情况形式类似，只需将**求和**替换为**积分**：

> **期望 (Mean)**:
> $$\mu = \mathbb{E}[X] = \int_{\mathcal{X}} x \cdot p(x)\, dx$$

> **矩 (Moments)**:
> $$M_k = \mathbb{E}[X^k] = \int_{\mathcal{X}} x^k \cdot p(x)\, dx$$

> **方差 (Variance)**:
> $$\text{Var}(X) = \mathbb{E}[(X - \mathbb{E}[X])^2] = \mathbb{E}[X^2] - (\mathbb{E}[X])^2 = M_2 - M_1^2$$

> **标准差 (Standard Deviation)**:
> $$\text{Std} = \sqrt{\text{Var}(X)}$$

---

## 9. 常见概率分布 (Common Distributions)

### 9.1 Bernoulli 分布（离散）

**背景**: 二元随机变量 $x \in \{0, 1\}$，如投掷一枚硬币。

参数 $\mu \in [0,1]$ 表示 $x=1$ 的概率：

$$p(x = 1 \mid \mu) = \mu, \quad p(x = 0 \mid \mu) = 1 - \mu$$

**PMF 的统一写法**:

> $$\text{Bern}(x \mid \mu) = \mu^x (1 - \mu)^{1-x}$$

**统计量**:

$$\mathbb{E}[x] = \mu, \quad \text{Var}[x] = \mu(1 - \mu)$$

> **ML 应用**: Binary classification 的输出层，逻辑回归（Logistic Regression）的概率输出。

---

### 9.2 Binomial 分布（离散）

**背景**: 独立重复 $N$ 次 Bernoulli 实验（每次成功概率为 $\mu$），$m$ 表示成功次数。

> $$\text{Bin}(m \mid N, \mu) = \binom{N}{m} \mu^m (1 - \mu)^{N-m}$$

其中 $\binom{N}{m} = \frac{N!}{(N-m)!\, m!}$（二项式系数）

**统计量**:

$$\mathbb{E}[m] = N\mu, \quad \text{Var}[m] = N\mu(1-\mu)$$

> **直觉解释**: Binomial 分布是 Bernoulli 分布的多次重复。当 $N=1$ 时退化为 Bernoulli 分布。随着 $N$ 增大，分布逐渐趋近于 Gaussian（中心极限定理）。

---

### 9.3 Multinomial 分布（多元离散）

**背景**: 将 Binomial 推广到 $K$ 个类别（$K > 2$）。

设 $\mathbf{x} = (x_1, \ldots, x_K)$，$x_k$ 为类别 $k$ 出现的次数，$\mu_k$ 为类别 $k$ 的概率（$\sum_k \mu_k = 1$）：

$$\text{Mult}(\mathbf{x} \mid N, \boldsymbol{\mu}) = \binom{N}{x_1, \ldots, x_K} \prod_{k=1}^K \mu_k^{x_k}$$

> **ML 应用**: 多分类问题的概率建模，Softmax 层输出服从 Multinomial 分布。

---

### 9.4 Gaussian / Normal 分布（连续）

**单变量高斯分布 (Univariate Gaussian)**:

> $$\mathcal{N}(x \mid \mu, \sigma^2) = \frac{1}{(2\pi\sigma^2)^{1/2}} \exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

其中：
- $\mu$: 均值（mean）= 分布的中心
- $\sigma^2$: 方差（variance）= 控制分布的宽度（$\sigma$ 为标准差）

**性质**: 关于均值 $\mu$ 对称，呈"钟形曲线"（bell curve），$68\%$ 的数据落在 $[\mu-\sigma, \mu+\sigma]$ 内。

---

### 9.5 多元高斯分布 (Multivariate Gaussian)

对于 $D$ 维向量 $\mathbf{x}$：

> $$\mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}, \boldsymbol{\Sigma}) = \frac{1}{(2\pi)^{D/2} |\boldsymbol{\Sigma}|^{1/2}} \exp\!\left(-\frac{(\mathbf{x} - \boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu})}{2}\right)$$

其中：
- $\boldsymbol{\mu} \in \mathbb{R}^D$: 均值向量
- $\boldsymbol{\Sigma} \in \mathbb{R}^{D \times D}$: **协方差矩阵 (covariance matrix)**，正定对称矩阵
- $|\boldsymbol{\Sigma}|$: $\boldsymbol{\Sigma}$ 的行列式
- $\boldsymbol{\Sigma}^{-1}$: $\boldsymbol{\Sigma}$ 的逆矩阵
- $(\mathbf{x} - \boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu})$: 马氏距离的平方（Mahalanobis distance）

> **直觉解释**: 协方差矩阵 $\boldsymbol{\Sigma}$ 的对角线元素控制各维度方差，非对角线元素描述不同维度之间的相关性。当 $\boldsymbol{\Sigma} = \sigma^2 \mathbf{I}$ 时，各维度独立且方差相等，分布等值线为圆形。

> **ML 应用**: 高斯混合模型（GMM）、线性判别分析（LDA）、卡尔曼滤波器、高斯过程回归等。

---

## 10. 最大似然估计 (MLE) 与最大后验估计 (MAP)

### 10.1 参数估计问题

**问题**: 给定观测数据 $\mathcal{D} = \{x_1, x_2, \ldots, x_N\}$，如何估计概率分布的参数 $\theta$？

假设数据 i.i.d. 服从参数为 $\theta$ 的分布 $p(x \mid \theta)$。

---

### 10.2 最大似然估计 (MLE, Maximum Likelihood Estimation)

**思想**: 找到使**观测数据出现概率最大**的参数 $\theta$。

**似然函数 (Likelihood Function)**:

$$\mathcal{L}(\theta) = p(\mathcal{D} \mid \theta) = \prod_{i=1}^N p(x_i \mid \theta)$$

由于连乘容易数值下溢，通常最大化**对数似然 (Log-Likelihood)**：

$$\ell(\theta) = \log \mathcal{L}(\theta) = \sum_{i=1}^N \log p(x_i \mid \theta)$$

> **MLE 估计**:
> $$\hat{\theta}_{MLE} = \arg\max_\theta \sum_{i=1}^N \log p(x_i \mid \theta)$$

**Gaussian 分布的 MLE 例子**:

设数据 $\{x_i\}$ i.i.d. ~ $\mathcal{N}(\mu, \sigma^2)$，对数似然为：

$$\ell(\mu, \sigma^2) = -\frac{N}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^N (x_i - \mu)^2$$

求偏导并令其为零，得：

$$\hat{\mu}_{MLE} = \frac{1}{N}\sum_{i=1}^N x_i \quad \text{（样本均值）}$$

$$\hat{\sigma}^2_{MLE} = \frac{1}{N}\sum_{i=1}^N (x_i - \hat{\mu})^2 \quad \text{（样本方差）}$$

> **直觉解释**: MLE 将参数视为"固定未知量"，通过最大化数据的出现概率来估计参数。它只利用**数据本身**的信息，不考虑参数的先验知识。

---

### 10.3 最大后验估计 (MAP, Maximum A Posteriori Estimation)

**思想**: 在 MLE 的基础上引入参数的**先验分布 $p(\theta)$**，找到使**后验概率最大**的参数。

由贝叶斯定理：

$$p(\theta \mid \mathcal{D}) = \frac{p(\mathcal{D} \mid \theta)\, p(\theta)}{p(\mathcal{D})} \propto p(\mathcal{D} \mid \theta)\, p(\theta)$$

> **MAP 估计**:
> $$\hat{\theta}_{MAP} = \arg\max_\theta p(\theta \mid \mathcal{D}) = \arg\max_\theta \left[\log p(\mathcal{D} \mid \theta) + \log p(\theta)\right]$$

对比 MLE：$\hat{\theta}_{MAP}$ 在对数似然的基础上**增加了 $\log p(\theta)$ 项**（先验的对数）。

> **直觉解释**: MAP 将参数视为随机变量，用先验 $p(\theta)$ 编码我们对参数的预先信念。先验就像"正则化项"——它约束参数不要偏离先验太远，防止过拟合。

---

### 10.4 MLE vs MAP 对比

| 特性 | MLE | MAP |
|------|-----|-----|
| **参数视角** | 固定未知量 | 随机变量（有先验分布）|
| **优化目标** | $\arg\max_\theta \log p(\mathcal{D} \mid \theta)$ | $\arg\max_\theta [\log p(\mathcal{D}\mid\theta) + \log p(\theta)]$ |
| **先验** | 不使用 | 使用 $p(\theta)$ |
| **数据量少时** | 可能过拟合 | 先验起正则化作用，更鲁棒 |
| **数据量多时** | 收敛至真实参数 | 先验影响减弱，趋近于 MLE |
| **对应正则化** | 无正则化 | L2 正则化（Gaussian 先验）/ L1 正则化（Laplace 先验）|
| **完整贝叶斯** | 点估计 | 点估计（后验的众数）|

> **关键联系**: 当先验 $p(\theta)$ 为各向同性高斯分布 $\mathcal{N}(0, \lambda^{-1}\mathbf{I})$ 时，MAP 等价于带 **L2 正则化（Ridge Regression）**的 MLE：
> $$\hat{\theta}_{MAP} = \arg\min_\theta \left[-\sum_{i=1}^N \log p(x_i \mid \theta) + \lambda \|\theta\|^2\right]$$

---

## 11. 信息论基础 (Information Theory)

### 11.1 信息量 (Information)

**Claude Shannon** 创立了信息论（Information Theory），定义了量化事件不确定性的度量。

对于概率为 $p_k$ 的事件 $x_k$，其**信息量**定义为：

> $$I(x_k) = \log \frac{1}{p_k} = -\log(p_k)$$

- 使用以2为底的对数时，单位为 **bits**
- $I(x_k) \geq 0$（非负）
- 当 $p_k = 1$（确定性事件）时，$I(x_k) = 0$（无不确定性，无信息）
- **越稀有的事件携带越多的信息量**（越出乎意料，越有信息）

### 11.2 熵 (Entropy)

**熵**是信息量的**期望值**，衡量整个概率分布的平均不确定性：

> $$H_P(X) = \mathbb{E}[I(x_k)] = -\sum_{x_k \in \mathcal{X}} p_k \log(p_k)$$

**二元熵（Binary Entropy）**:

$$H_P(X) = -p_0 \log p_0 - (1-p_0) \log(1-p_0)$$

- 当 $p_0 = 0.5$ 时，$H_P(X)$ 达到最大值（最大不确定性）
- 当 $p_0 = 0$ 或 $1$ 时，$H_P(X) = 0$（完全确定，无不确定性）

> **ML 应用**: 决策树（Decision Tree）使用信息增益（information gain，基于熵）选择最佳分裂特征；交叉熵是分类任务的常用损失函数。

### 11.3 交叉熵 (Cross-Entropy)

**交叉熵**衡量：用分布 $Q(X)$ 编码真实分布 $P(X)$ 的数据时，所需的**平均比特数**：

> $$H_{P,Q}(X) = -\sum_{x_k \in \mathcal{X}} P(X = x_k) \cdot \log Q(X = x_k)$$

其中 $P$ 是真实分布，$Q$ 是模型分布。

**二元交叉熵**：

$$H_{P,Q}(X) = -p_0 \log q_0 - (1-p_0) \log(1-q_0)$$

**性质**:
1. **非负性**: $H_{P,Q}(X) \geq 0$
2. **不小于熵**: $H_{P,Q}(X) \geq H_P(X)$，等号当且仅当 $P = Q$ 时成立

> **ML 应用**: 分类任务中，交叉熵损失 $H_{P,Q}$ 衡量模型输出分布 $Q$ 与真实标签分布 $P$ 的差距。最小化交叉熵 = 使模型分布尽量接近真实分布。

### 11.4 KL 散度 (Kullback-Leibler Divergence)

**KL 散度**（相对熵，Relative Entropy）衡量两个概率分布之间的"距离"：

**离散情况**:

$$D_{P,Q}(X) = \sum_{x_k \in \mathcal{X}} P(X = x_k) \cdot \log \frac{P(X = x_k)}{Q(X = x_k)}$$

**连续情况**:

$$D_{P,Q}(X) = \int_{x \in \mathcal{X}} p_X(x) \log \frac{p_X(x)}{q_X(x)}\, dx$$

**性质**:
1. **非负性**: $D_{P,Q}(X) \geq 0$（由 Jensen 不等式证明，$\log(\cdot)$ 为凹函数）
2. **不对称性**: $D_{P,Q}(X) \neq D_{Q,P}(X)$（故 KL 散度不是真正的"距离"）

**交叉熵与 KL 散度的关系**:

> $$H_{P,Q}(X) = H_P(X) + D_{P,Q}(X)$$

即：交叉熵 = 真实分布的熵 + KL 散度。由于熵 $H_P(X)$ 是固定的（不依赖模型参数），**最小化交叉熵等价于最小化 KL 散度**。

> **ML 应用**: VAE（Variational Autoencoder）使用 KL 散度正则化潜在空间；生成模型（GAN, VAE）的训练目标往往与 KL 散度或其变体相关。

---

## 12. 补充资料 (Supplementary Resources)

### 教材章节对照

| 内容 | Murphy (MLPP) | Bishop (PRML) | Burkov (100-page ML) |
|------|---------------|----------------|----------------------|
| 概率基础（事件、随机变量）| Ch. 2.1–2.2 | App. B | Ch. 1 |
| 联合/边际/条件概率 | Ch. 2.2 | App. B | Ch. 1 |
| 贝叶斯定理 | Ch. 2.2.3 | Ch. 1.2.1 | Ch. 1 |
| 期望与方差 | Ch. 2.3 | App. B | Ch. 1 |
| Bernoulli 分布 | Ch. 3.3 | Ch. 2.1 | Ch. 1 |
| Binomial 分布 | Ch. 3.3 | Ch. 2.1 | Ch. 1 |
| Gaussian/Normal 分布 | Ch. 2.3.2 | Ch. 2.3 | Ch. 1 |
| 多元 Gaussian | Ch. 4.1 | Ch. 2.3 | Ch. 1 |
| MLE | Ch. 3.5.1 | Ch. 1.2.5 | Ch. 4 |
| MAP | Ch. 3.5.2 | Ch. 1.2.6 | Ch. 4 |
| 信息论（熵、KL 散度）| Ch. 2.8 | Ch. 1.6 | — |

### 关键公式速查

> **贝叶斯定理**:
> $$P(Y \mid X) = \frac{P(X \mid Y)\, P(Y)}{P(X)} \quad \Leftrightarrow \quad \text{Posterior} \propto \text{Likelihood} \times \text{Prior}$$

> **MLE**:
> $$\hat{\theta}_{MLE} = \arg\max_\theta \sum_{i=1}^N \log p(x_i \mid \theta)$$

> **MAP**:
> $$\hat{\theta}_{MAP} = \arg\max_\theta \left[\sum_{i=1}^N \log p(x_i \mid \theta) + \log p(\theta)\right]$$

> **交叉熵损失（分类）**:
> $$\mathcal{L} = H_{P,Q} = -\sum_k P_k \log Q_k$$

> **KL 散度与交叉熵关系**:
> $$H_{P,Q} = H_P + D_{KL}(P \| Q)$$

### 补充学习资源

- **交叉熵入门**: https://machinelearningmastery.com/cross-entropy-for-machine-learning/
- **KL 散度非负性证明**: https://stats.stackexchange.com/questions/335197/why-kl-divergence-is-non-negative
- **3Blue1Brown 贝叶斯视频**: Bayes theorem, and making probability intuitive
