# L05 Linear Regression — DDA3020 Machine Learning

> **课程**: DDA3020 Machine Learning, CUHK-SZ  
> **讲师**: Baoyuan Wu  
> **日期**: January 21/26/28, 2026  

---

## 目录 (Table of Contents)

1. [数学符号与预备知识](#1-数学符号与预备知识)
2. [线性回归模型定义](#2-线性回归模型定义)
3. [确定性视角下的线性回归](#3-确定性视角下的线性回归-deterministic-perspective)
4. [概率视角下的线性回归 (MLE)](#4-概率视角下的线性回归-probabilistic-perspective--mle)
5. [解析解：正规方程](#5-解析解正规方程-normal-equation)
6. [梯度下降求解](#6-梯度下降求解-gradient-descent)
7. [解析解 vs 梯度下降：对比](#7-解析解-vs-梯度下降对比)
8. [多输出线性回归](#8-多输出线性回归-multiple-outputs)
9. [线性回归用于分类](#9-线性回归用于分类)
10. [Ridge Regression（L2 正则化）](#10-ridge-regressionl2-正则化)
11. [多项式回归 (Polynomial Regression)](#11-多项式回归-polynomial-regression)
12. [Lasso Regression（L1 正则化）](#12-lasso-regressionl1-正则化)
13. [鲁棒线性回归 (Robust Linear Regression)](#13-鲁棒线性回归-robust-linear-regression)
14. [各类线性回归总结对比](#14-各类线性回归总结对比)
15. [模型评估指标](#15-模型评估指标)
16. [补充资料](#16-补充资料)

---

## 1. 数学符号与预备知识

### 1.1 集合 (Sets)

- 有限集合: $\{1, 3, 18, 23\}$ 或 $\{x_1, x_2, \ldots, x_d\}$
- 闭区间: $[a, b]$（包含端点）；开区间: $(a, b)$（不含端点）
- 全体实数: $\mathbb{R}$
- 集合运算: 交集 $S_1 \cap S_2$，并集 $S_1 \cup S_2$

### 1.2 函数 (Functions)

- $f: \mathbb{R}^d \to \mathbb{R}$ 表示 $f$ 将 $d$ 维实向量映射到实数（标量值函数）
- $f(x) = f(x_1, x_2, \ldots, x_d)$

**线性函数 (Linear function)**:
- 满足**齐次性 (Homogeneity)**: $f(\alpha x) = \alpha f(x)$
- 满足**可加性 (Additivity)**: $f(x + y) = f(x) + f(y)$
- 即满足**叠加原理 (Superposition)**: $f(\alpha x + \beta y) = \alpha f(x) + \beta f(y)$
- 形式: $f(x) = a^\top x$

**仿射函数 (Affine function)**: $f(x) = a^\top x + b$（线性函数加常数偏置）

### 1.3 导数与梯度 (Derivative and Gradient)

**标量对向量求导（分母布局）**:

若 $f(w): \mathbb{R}^d \to \mathbb{R}$，则梯度为 $d \times 1$ 向量：

$$\frac{df(w)}{dw} = \nabla_w f = \begin{bmatrix} \frac{\partial f}{\partial w_1} \\ \vdots \\ \frac{\partial f}{\partial w_d} \end{bmatrix}$$

**向量对向量求导（Jacobian 矩阵）**:

若 $f(w): \mathbb{R}^d \to \mathbb{R}^h$，则 $J = \frac{df(w)}{dw} \in \mathbb{R}^{d \times h}$，$J_{ij} = \frac{\partial f_j}{\partial w_i}$。

**常用矩阵求导公式**（分母布局）:

> $$\frac{d(X^\top w)}{dw} = X \quad \text{（$X$ 不含 $w$）}$$
> $$\frac{d(y^\top X w)}{dw} = X^\top y$$
> $$\frac{d(w^\top X w)}{dw} = (X + X^\top) w$$

**注意**: 本课采用分母布局 (denominator layout)，所有结果均按此约定。

---

## 2. 线性回归模型定义

### 2.1 问题设置

- **数据集**: $m$ 个有标签样本 $\{(x_i, y_i)\}_{i=1}^m$
  - $x_i \in \mathbb{R}^d$：$d$ 维特征向量（feature vector）
  - $y_i \in \mathbb{R}$：实值目标（回归目标）
- **任务**: 用线性模型 $f_w$ 近似真实目标函数 $t: \mathcal{X} \to \mathcal{Y}$

### 2.2 线性假设函数 (Linear Hypothesis Function)

$$f_{w_0, w_1, \ldots, w_d}(x) = w_0 + w_1 x_1 + w_2 x_2 + \cdots + w_d x_d$$

其中 $w_0$ 是**偏置项 (bias)**，$w_1, \ldots, w_d$ 是**系数 (coefficients)**。

**向量化表示**: 令 $w = [w_0, w_1, \ldots, w_d]^\top$，并将 $x$ 增广为 $x \leftarrow [1, x_1, \ldots, x_d]^\top$，则：

> $$f_w(x) = w^\top x$$

**重要说明**: $f_w(x)$ 称为"线性"是因为它关于**参数向量 $w$** 是线性的，而不是关于原始特征 $[x_1, \ldots, x_d]^\top$ 的。

---

## 3. 确定性视角下的线性回归 (Deterministic Perspective)

### 3.1 目标函数（损失函数）

> **均方误差 (Mean Squared Error, MSE)**:
> $$\min_w \frac{1}{m} \sum_{i=1}^m (f_w(x_i) - y_i)^2$$

- $(f_w(x_i) - y_i)^2$：第 $i$ 个样本的**平方误差损失 (squared error loss)**
- 整体目标：**经验风险 (empirical risk)**，即损失的平均值
- 这个损失也叫 **残差平方和 (Residual Sum of Squares, RSS)**

**直觉**: 我们希望模型的预测值尽可能接近真实值，用平方而不是绝对值是为了保证可微性（便于求导），且对大误差的惩罚更重。

---

## 4. 概率视角下的线性回归 (Probabilistic Perspective / MLE)

### 4.1 概率模型假设

假设输入和输出的关系为：

> $$y = w^\top x + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2)$$

其中 $\epsilon$ 是**观测噪声 (observation noise)** 或**残差误差 (residual error)**，与 $x$ 独立。

因此，$y$ 也是随机变量，其条件概率为：

$$p(y \mid x, w) = \mathcal{N}(w^\top x, \sigma^2)$$

### 4.2 最大似然估计 (MLE) 推导

给定训练数据 $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^m$，参数 $w$ 的 MLE 估计为：

$$w_{\text{MLE}} = \arg\max_w \log L(w; \mathcal{D})$$

**推导步骤**:

$$\log L(w; \mathcal{D}) = \log \prod_{i=1}^m p(y_i \mid x_i, w) \tag{1}$$

$$= \sum_{i=1}^m \log \mathcal{N}(w^\top x_i, \sigma^2) \tag{2}$$

$$= \sum_{i=1}^m \left[-\log(\sigma\sqrt{2\pi}) - \frac{(y_i - w^\top x_i)^2}{2\sigma^2}\right] \tag{3}$$

$$= -m\log(\sigma\sqrt{2\pi}) - \frac{1}{2\sigma^2} \sum_{i=1}^m (y_i - w^\top x_i)^2 \tag{4}$$

去掉与 $w$ 无关的常数项，**最大化对数似然等价于最小化 RSS**:

> $$w_{\text{MLE}} = \arg\min_w \frac{1}{2} \sum_{i=1}^m (y_i - w^\top x_i)^2$$

**结论**: MLE（假设噪声为高斯分布）等价于最小二乘回归（确定性视角），两者给出相同的目标函数。这是线性回归的重要理论支撑。

---

## 5. 解析解：正规方程 (Normal Equation)

### 5.1 矩阵形式化

将 $m$ 个样本打包成矩阵形式：

$$X = \begin{bmatrix} x_1^\top \\ \vdots \\ x_m^\top \end{bmatrix} \in \mathbb{R}^{m \times (d+1)}, \quad y = \begin{bmatrix} y_1 \\ \vdots \\ y_m \end{bmatrix} \in \mathbb{R}^m$$

目标函数的矩阵形式（令 $e = Xw - y$）：

$$J(w) = \sum_{i=1}^m (f_w(x_i) - y_i)^2 = e^\top e = (Xw - y)^\top(Xw - y)$$

**展开**:
$$J(w) = w^\top X^\top X w - 2y^\top X w + y^\top y$$

### 5.2 正规方程推导

对 $w$ 求偏导并令其等于零：

$$\frac{\partial}{\partial w} J(w) = 0$$
$$\Rightarrow 2X^\top X w - 2X^\top y = 0$$
$$\Rightarrow X^\top X w = X^\top y$$

若 $X^\top X$ 可逆，则解为：

> **正规方程 (Normal Equation)**:
> $$\hat{w} = (X^\top X)^{-1} X^\top y$$

**预测**: $f_w(X_{\text{new}}) = X_{\text{new}} \hat{w}$

### 5.3 数值示例

**一维情形** ($d = 1$)：设数据点为 $\{(-2, -2.5), (0, -1), (2, 0), (4, 2)\}$，则：

$$y = -1.4375 + 0.5625x$$

预测 $x = -1$：$\hat{y} = [1, -1] \begin{bmatrix} -1.4375 \\ 0.5625 \end{bmatrix} = -2$

### 5.4 何时 $X^\top X$ 不可逆？

- 特征之间存在**线性相关 (multicollinearity)**（某些特征是另一些的线性组合）
- 特征数 $d$ 超过样本数 $m$（过定方程组，underdetermined system）

**解决方案**: Ridge Regression（第10节）通过加入正则项 $\lambda I$ 保证可逆性。

---

## 6. 梯度下降求解 (Gradient Descent)

### 6.1 梯度计算

$$J(w) = \frac{1}{2} \|Xw - y\|^2 = \frac{1}{2} \sum_{i=1}^m (x_i^\top w - y_i)^2$$

$$\frac{\partial J(w)}{\partial w} = X^\top(Xw - y)$$

**推导细节**:
$$\frac{\partial}{\partial w} (Xw - y)^\top (Xw - y) = 2X^\top(Xw - y)$$
（因此省略 $\frac{1}{2}$ 后梯度化简为 $X^\top(Xw - y)$）

### 6.2 梯度下降更新规则

> $$w \leftarrow w - \alpha \frac{\partial J(w)}{\partial w} = w - \alpha X^\top(Xw - y)$$

其中 $\alpha$ 是学习率 (learning rate / step size)。

**收敛性**: 由于线性回归的损失函数 $J(w)$ 是凸函数（$X^\top X \succeq 0$），梯度下降保证收敛到全局最优。

**学习率选择实践**:
- 绘制 $J(w)$ 关于迭代次数的曲线，确保每次迭代 $J(w)$ 都在减小
- 若 $J(w)$ 增大，说明 $\alpha$ 太大，需减小
- 若 $J(w)$ 减小非常慢，考虑增大 $\alpha$

---

## 7. 解析解 vs 梯度下降：对比

| 方法 | 公式 | 优点 | 缺点 |
|------|------|------|------|
| **解析解 (Normal Equation)** | $w = (X^\top X)^{-1} X^\top y$ | 无超参数；无需迭代 | 计算复杂度 $O(d^3 + md^2)$；$d$ 很大时慢 |
| **梯度下降** | $w \leftarrow w - \alpha X^\top(Xw - y)$ | $d$ 很大时仍有效 | 需要选择 $\alpha$；需要多次迭代；复杂度 $O(T \times md)$ |

**选择建议**:
- $d$ 很小（$d \leq 10^4$）：优先选择解析解
- $d$ 很大（如 NLP, CV 特征）：选择梯度下降

---

## 8. 多输出线性回归 (Multiple Outputs)

### 8.1 问题设置

当输出 $y_i \in \mathbb{R}^h$（$h > 1$ 个输出维度），模型变为：

$$f_W(x) = x^\top W, \quad W = [w_1, \ldots, w_h] \in \mathbb{R}^{(d+1) \times h}$$

### 8.2 目标函数与解析解

$$J(W) = \text{trace}\left[(XW - Y)^\top(XW - Y)\right]$$

其中 $Y \in \mathbb{R}^{m \times h}$，若 $X^\top X$ 可逆：

> $$\hat{W} = (X^\top X)^{-1} X^\top Y$$

**假设**: 各输出维度的误差项相互独立，即对所有 $k = 1, \ldots, h$，$e_k^\top e_k$ 相互独立。

---

## 9. 线性回归用于分类

### 9.1 二分类

标签 $y_i \in \{-1, +1\}$，学习 $\hat{w} = (X^\top X)^{-1} X^\top y$，预测：

$$f_w(x_{\text{new}}) = \text{sgn}(x_{\text{new}}^\top \hat{w})$$

### 9.2 多分类

使用 **one-hot 编码**: 第 $c$ 类的标签向量 $y_i^\top = [0, \ldots, 1, \ldots, 0]$（第 $c$ 位为 1）。

学习 $\hat{W} = (X^\top X)^{-1} X^\top Y$，预测：

$$f_W(x_{\text{new}}) = \arg\max_{c=1,\ldots,C}(x_{\text{new}}^\top \hat{W})$$

**直觉**: 用最大输出分数对应的类别作为预测结果。

---

## 10. Ridge Regression（L2 正则化）

### 10.1 动机一：$X^\top X$ 不可逆

正规方程中若 $X^\top X$ 奇异，无法求逆。加入正则项后：

$$(X^\top X + \lambda \hat{I}_d)w = X^\top y$$

其中 $\hat{I}_d$ 是将 $d+1$ 维单位矩阵的 $(1,1)$ 元素设为 $0$ 的矩阵（**不惩罚偏置 $w_0$**），因为：

> **注意**: 偏置项 $w_0$ 仅影响函数的高度（平移），不影响复杂度，故不加入正则化。

当 $\lambda > 0$ 时，$(X^\top X + \lambda \hat{I}_d)$ 保证可逆。

### 10.2 动机二：防止过拟合 (Overfitting)

当模型过于复杂（如高阶多项式），参数值会非常大（正负交替），使得模型对训练数据拟合极好，但泛化性差。

### 10.3 MAP 估计推导（概率视角）

假设参数 $w$（不含偏置）服从**高斯先验**：

$$p(w) = \mathcal{N}(w \mid 0, \tau^2 I)$$

则 MAP 估计：

$$w_{\text{MAP}} = \arg\max_w \left[\sum_{i=1}^m \log p(y_i \mid x_i, w) + \log p(w)\right] \tag{10}$$

$$= \arg\max_w \left[\sum_{i=1}^m \log \mathcal{N}(x_i^\top w, \sigma^2) + \log \mathcal{N}(w \mid 0, \tau^2 I)\right] \tag{11}$$

$$\equiv \arg\min_w \left[\sum_{i=1}^m (x_i^\top w - y_i)^2 + \lambda \|w\|_2^2\right] \tag{12}$$

其中 $\lambda = \sigma^2 / \tau^2$。

**解析解**:

> **Ridge Regression 闭式解**:
> $$w_{\text{MAP}} = (\lambda I + X^\top X)^{-1} X^\top y$$

**L2 正则化** 也称为 **权重衰减 (weight decay)**。

### 10.4 正则化强度 $\lambda$ 的效果

- $\lambda$ 越大 → 参数越小 → 模型越平滑 → 可能欠拟合
- $\lambda$ 越小 → 参数越大 → 模型越复杂 → 可能过拟合
- $\lambda = 0$：退化为普通最小二乘（OLS）

**Ridge 目标函数**:

> $$J_{\text{ridge}}(w) = \sum_{i=1}^m (f_w(x_i) - y_i)^2 + \lambda \|w\|_2^2$$

---

## 11. 多项式回归 (Polynomial Regression)

### 11.1 动机

某些数据不是线性可分的（如 XOR 数据），线性模型无法拟合。需要**非线性映射**将特征展开。

### 11.2 基展开 (Basis Expansion)

**二次模型**（加入所有配对乘积项）:
$$f_w(x) = w_0 + \sum_{i=1}^d w_i x_i + \sum_{i=1}^d \sum_{j=1}^d w_{ij} x_i x_j$$

**一般多项式展开**:
$$f_w(x) = w_0 + \sum_i w_i x_i + \sum_{i,j} w_{ij} x_i x_j + \sum_{i,j,k} w_{ijk} x_i x_j x_k + \cdots = \phi(x)^\top w$$

其中**特征映射 (feature map)**:
$$\phi(x) = [1, x_1, \ldots, x_d, \ldots, x_i x_j, \ldots, x_i x_j x_k, \ldots]^\top$$

> **关键点**: $f_w(x)$ 关于 $w$ 仍是**线性函数**，只是关于 $x$ 变成了非线性的。因此仍称为"线性模型"，可以使用相同的求解方法。

### 11.3 矩阵化

对 $m$ 个数据点做基展开：
$$P(X) = [\phi(x_1)^\top; \ldots; \phi(x_m)^\top] \in \mathbb{R}^{m \times |\phi|}$$

**Ridge + Basis Expansion 的解**:
$$\hat{w} = (P^\top P + \lambda I)^{-1} P^\top y$$

### 11.4 注意事项

- 对于高维 $d$ 和高阶多项式，展开项数**指数级增长**。
- 实践中，高维问题下阶数超过 3 的多项式很少使用。
- 多项式回归容易**过拟合**，通常配合 Ridge 使用。

**应用示例（分类）**:
- 对 XOR 二分类，使用 2 阶多项式展开可以用线性方法实现非线性分类边界。

---

## 12. Lasso Regression（L1 正则化）

### 12.1 Laplacian 先验

将高斯先验替换为**拉普拉斯先验 (Laplacian prior)**:
$$p(w) = \text{Lap}(w \mid 0, b) = \frac{1}{2b} \exp\left(-\frac{\|w\|_1}{b}\right)$$

MAP 估计推导：

$$w_{\text{MAP}} = \arg\max_w \left[\sum_{i=1}^m \log p(y_i \mid x_i, w) + \log p(w)\right]$$

$$\equiv \arg\min_w \left[\sum_{i=1}^m (x_i^\top w - y_i)^2 + \alpha \|w\|_1\right]$$

> **Lasso 目标函数**:
> $$J_{\text{lasso}}(w) = \sum_{i=1}^m (f_w(x_i) - y_i)^2 + \alpha \|w\|_1$$

### 12.2 Lasso 的特性：稀疏解

> **关键性质**: Lasso 会产生**稀疏参数 (sparse parameters)**，即大量参数精确为 0。

**几何解释**（Ridge vs Lasso）:
- Ridge 的约束区域是**圆球**（所有方向惩罚相同），解趋向于小但非零的参数。
- Lasso 的约束区域是**菱形/多面体**（有角点），解容易停在角点处（某些参数恰好为零）。

这使得 Lasso 实现了**特征选择 (feature selection)**——自动将不重要的特征权重置零。

### 12.3 Ridge 与 Lasso 的对比

| 性质 | Ridge (L2) | Lasso (L1) |
|------|-----------|-----------|
| 先验分布 | Gaussian | Laplacian |
| 正则项 | $\lambda \|w\|_2^2$ | $\alpha \|w\|_1$ |
| 解的特点 | 小而非零 | 稀疏（部分精确为零） |
| 特征选择 | 否 | 是 |
| 可解析求解 | 是 | 否（需凸优化算法） |
| 约束形状 | 圆球 | 菱形 |

**Elastic Net**: 结合 L1 和 L2 正则化：
$$J_{\text{EN}}(w) = \sum_{i=1}^m (f_w(x_i) - y_i)^2 + \alpha_1 \|w\|_1 + \alpha_2 \|w\|_2^2$$

---

## 13. 鲁棒线性回归 (Robust Linear Regression)

### 13.1 离群点问题

普通最小二乘对**离群点 (outliers)** 非常敏感。当存在极端值时，$\ell_2$ 损失（平方误差）会被极大地放大，导致拟合结果严重偏移。

原因：$\ell_2$ 损失随残差**二次增长**，大残差的权重远大于小残差。

### 13.2 L1 损失（鲁棒回归）

使用 $\ell_1$ 损失替代 $\ell_2$ 损失：

$$J(w) = \sum_{i=1}^m |x_i^\top w - y_i|$$

**概率解释**: 假设噪声服从**拉普拉斯分布 (Laplacian distribution)**：
$$p(y \mid x, w, b) = \text{Lap}(y \mid w^\top x, b) \propto \exp\left(-\frac{|y - w^\top x|}{b}\right)$$

MLE 等价于最小化 $\ell_1$ 损失。

> **直觉**: 当残差较大时，$\ell_1$ 损失远小于 $\ell_2$ 损失，因此离群点的影响被大幅削弱。

### 13.3 求解挑战

$\ell_1$ 损失在 $y_i = w^\top x_i$ 处**不可微**，不能直接用梯度下降。

**方法一：转化为线性规划 (LP)**:
$$\min_{w, t} \sum_i t_i, \quad \text{s.t.} \quad -t_i \leq x_i^\top w - y_i \leq t_i$$

**方法二：迭代重加权最小二乘 (IRLS)**:

利用 $|a| = \min_{\mu > 0} \frac{1}{2}\left(\frac{a^2}{\mu} + \mu\right)$，将 $\ell_1$ 问题转化为加权 $\ell_2$ 问题，交替优化：
- 给定 $w$：令 $\mu_i = |x_i^\top w - y_i|$
- 给定 $\mu$：$w = \arg\min_w \sum_{i=1}^m \frac{(x_i^\top w - y_i)^2}{2\mu_i}$（加权最小二乘）

### 13.4 概率分布对应的回归方法

| $p(y \mid x, w)$ | $p(w)$（先验）| 回归方法 |
|------|------|------|
| Gaussian | Uniform（均匀） | 普通最小二乘 (OLS) |
| Gaussian | Gaussian | Ridge Regression |
| Gaussian | Laplace | Lasso Regression |
| Laplace | Uniform | Robust Regression ($\ell_1$) |
| Student-$t$ | Uniform | Robust Regression |

---

## 14. 各类线性回归总结对比

| 特性 | OLS | Ridge | Lasso | Robust |
|------|-----|-------|-------|--------|
| 损失函数 | $\ell_2$ | $\ell_2$ | $\ell_2$ | $\ell_1$ |
| 正则项 | 无 | $\lambda \|w\|_2^2$ | $\alpha \|w\|_1$ | 无 |
| 先验 | Uniform | Gaussian | Laplace | Uniform |
| 闭式解 | 有 | 有 | 无（需迭代） | 无（需迭代/LP） |
| 稀疏性 | 无 | 无 | 有 | 无 |
| 抗离群点 | 差 | 差 | 差 | 强 |

---

## 15. 模型评估指标

### 15.1 均方误差 (MSE)

$$\text{MSE} = \frac{1}{m} \sum_{i=1}^m (y_i - \hat{y}_i)^2$$

### 15.2 均方根误差 (RMSE)

$$\text{RMSE} = \sqrt{\text{MSE}} = \sqrt{\frac{1}{m} \sum_{i=1}^m (y_i - \hat{y}_i)^2}$$

与 $y$ 的量纲相同，更直观。

### 15.3 平均绝对误差 (MAE)

$$\text{MAE} = \frac{1}{m} \sum_{i=1}^m |y_i - \hat{y}_i|$$

对离群点不敏感（$\ell_1$ 基础）。

### 15.4 决定系数 R²

$$R^2 = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2}$$

- $R^2 = 1$：完美拟合
- $R^2 = 0$：模型与均值预测等效（没有解释能力）
- $R^2 < 0$：模型比均值预测还差

---

## 常见错误与考试注意点

1. **$w^\top x$ 还是 $x^\top w$?** 两者相等（标量），但矩阵化时注意 $Xw$（$X$ 是 $m \times (d+1)$ 矩阵）。
2. **正规方程不总成立**: 当 $X^\top X$ 奇异（特征共线或 $m < d$）时需用 Ridge 或伪逆。
3. **MLE = OLS 的前提**: 噪声必须是高斯分布；若噪声是拉普拉斯分布，MLE 对应的是 $\ell_1$ 损失（Robust 回归）。
4. **Ridge 不对偏置项惩罚**: 只对 $w_1, \ldots, w_d$ 加 L2 正则，$w_0$ 不正则化。
5. **多项式回归仍是"线性模型"**: 因为对参数 $w$ 仍是线性的——可以用相同的解析解方法。
6. **Lasso 无解析解**: L1 范数不可微，无法通过令梯度为零直接求解。
7. **梯度计算**: $\frac{\partial}{\partial w}\frac{1}{2}\|Xw - y\|^2 = X^\top(Xw - y)$，记得 $\frac{1}{2}$ 的因子化简作用。

---

## 16. 补充资料

### 教材参考

- **Murphy "Machine Learning: A Probabilistic Perspective"**  
  - Ch. 7: 线性回归（OLS、Ridge、Lasso、Elastic Net、贝叶斯线性回归）  
  - Ch. 8.3: 稀疏线性模型（Lasso 的详细推导）  

- **Bishop "Pattern Recognition and Machine Learning" (PRML)**  
  - Ch. 3: 线性回归模型（包含基函数、正则化、贝叶斯视角的完整推导）  
  - Ch. 3.1: 最小二乘  
  - Ch. 3.3: 贝叶斯线性回归  

- **Burkov "The Hundred-Page Machine Learning Book"**  
  - Ch. 4: 基础数学（梯度、矩阵运算）  
  - Ch. 5: 基本算法（线性回归、正则化）  

### 在线资源

- **矩阵求导参考**: [Matrix Calculus (Wikipedia)](https://en.wikipedia.org/wiki/Matrix_calculus)
- **L1 范数最小化转线性规划**: [StackExchange 推导](https://math.stackexchange.com/questions/1639716/how-can-l-1-norm-minimization-with)
- **Boyd "Convex Optimization"**: Appendix A（线性代数背景）

### 编程实践建议

- 使用 `numpy.linalg.lstsq` 实现 OLS（自动处理奇异矩阵）
- 使用 `sklearn.linear_model.Ridge`、`Lasso`、`ElasticNet`
- 可视化不同 $\lambda$ 值下 Ridge/Lasso 的系数路径（Regularization Path）
