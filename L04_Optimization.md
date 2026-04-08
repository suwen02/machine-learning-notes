# L04 Basic Optimization — DDA3020 Machine Learning

> **课程**: DDA3020 Machine Learning, CUHK-SZ  
> **讲师**: Baoyuan Wu  
> **日期**: January 14/19, 2026  
> **参考来源**: Boyd & Vandenberghe "Convex Optimization" (Stanford EE364a)

---

## 目录 (Table of Contents)

1. [凸集 (Convex Set)](#1-凸集-convex-set)
2. [凸函数 (Convex Function)](#2-凸函数-convex-function)
3. [凸优化问题 (Convex Optimization Problem)](#3-凸优化问题-convex-optimization-problem)
4. [无约束最小化：梯度下降法](#4-无约束最小化梯度下降法-unconstrained-minimization)
5. [有约束最小化：拉格朗日对偶与KKT条件](#5-有约束最小化拉格朗日对偶与kkt条件)
6. [优化与机器学习的关系](#6-优化与机器学习的关系)
7. [补充资料](#7-补充资料)

---

## 1. 凸集 (Convex Set)

### 1.1 仿射集 (Affine Set)

**定义**: 过 $x_1, x_2$ 两点的**仿射直线**上的所有点可以表示为：

$$x = \theta x_1 + (1-\theta) x_2, \quad \theta \in \mathbb{R}$$

> **关键定义**: 集合 $C$ 是**仿射集 (Affine set)**，当且仅当对集合中任意两个不同点，过这两点的直线完整地属于集合 $C$。

- **例子**: 线性方程组的解集 $\{x \mid Ax = b\}$ 是仿射集。
- 反之，每一个仿射集都可以表示为某个线性方程组的解集。

### 1.2 凸集 (Convex Set)

**定义**: 连接 $x_1, x_2$ 的**线段**上所有点：

$$x = \theta x_1 + (1-\theta) x_2, \quad 0 \leq \theta \leq 1$$

> **关键定义**: 集合 $C$ 是**凸集 (Convex set)**，当且仅当对任意 $x_1, x_2 \in C$ 和 $0 \leq \theta \leq 1$，有：
> $$\theta x_1 + (1-\theta) x_2 \in C$$

**直觉解释**: 凸集中任意两点的连线仍在集合内。与仿射集的区别在于仿射集要求整条直线（$\theta$ 可以是任意实数）都在集合内，而凸集只要求线段（$0 \leq \theta \leq 1$）在集合内。

- 仿射集是凸集的一种特例（仿射集一定是凸集）。

---

## 2. 凸函数 (Convex Function)

### 2.1 凸函数定义

> **定义**: $f: \mathbb{R}^n \to \mathbb{R}$ 是**凸函数 (convex function)**，当且仅当 $\text{dom}\, f$ 是凸集，且对所有 $x, y \in \text{dom}\, f$，$0 \leq \theta \leq 1$，有：
> $$f(\theta x + (1-\theta)y) \leq \theta f(x) + (1-\theta)f(y)$$

**几何直觉**: 函数图像上任意两点之间的**弦 (chord)** 位于函数图像的**上方**——即函数曲线是"向下凹的"。

- **凹函数 (concave function)**: $-f$ 是凸函数，则 $f$ 是凹函数。
- **严格凸函数 (strictly convex)**: $x \neq y$，$0 < \theta < 1$ 时不等式严格成立。

### 2.2 凸函数举例

**在 $\mathbb{R}$ 上的凸函数**:
- 仿射函数: $ax + b$（同时是凸函数和凹函数）
- 指数函数: $e^{ax}$，$\forall a \in \mathbb{R}$
- 幂函数: $x^\alpha$（$x > 0$），当 $\alpha \geq 1$ 或 $\alpha \leq 0$
- 绝对值的幂: $|x|^p$，$p \geq 1$
- 负熵: $x \log x$（$x > 0$）

**在 $\mathbb{R}$ 上的凹函数**:
- 对数函数: $\log x$（$x > 0$）
- 幂函数: $x^\alpha$（$0 \leq \alpha \leq 1$，$x > 0$）

**在 $\mathbb{R}^n$ 上**:
- 仿射函数 $f(x) = a^\top x + b$
- $\ell_p$ 范数: $\|x\|_p = \left(\sum_{i=1}^n |x_i|^p\right)^{1/p}$，$p \geq 1$

**在矩阵空间 $\mathbb{R}^{m \times n}$ 上**:
- 仿射函数: $f(X) = \text{tr}(A^\top X) + b$
- 谱范数 (spectral norm): $f(X) = \|X\|_2 = \sigma_{\max}(X) = \left(\lambda_{\max}(X^\top X)\right)^{1/2}$

### 2.3 一阶条件 (First-order Condition)

若 $f$ 可微（$\text{dom}\, f$ 是开集，一阶偏导数存在且连续），梯度向量为：
$$\nabla f(x) = \left[\frac{\partial f(x)}{\partial x_1}; \frac{\partial f(x)}{\partial x_2}; \ldots; \frac{\partial f(x)}{\partial x_n}\right]$$

> **一阶条件**: 可微函数 $f$（定义域为凸集）是凸函数，当且仅当对所有 $x, y \in \text{dom}\, f$：
> $$f(y) \geq f(x) + \nabla f(x)^\top (y - x)$$

**直觉**: 一阶泰勒展开是函数的**全局下估计 (global underestimator)**。若在 $x$ 处的切线（切超平面）总在函数曲线之下，则函数是凸的。

**考试要点**: 这个不等式是证明凸性的重要工具，也是梯度下降法的理论基础。

### 2.4 二阶条件 (Second-order Condition)

若 $f$ 二阶可微，**Hessian 矩阵**为：
$$\nabla^2 f(x)_{ij} = \frac{\partial^2 f(x)}{\partial x_i \partial x_j}, \quad i, j = 1, \ldots, n$$

> **二阶条件**: 二阶可微函数 $f$（定义域为凸集）是凸函数，当且仅当对所有 $x \in \text{dom}\, f$：
> $$\nabla^2 f(x) \succeq 0 \quad (\text{正半定, positive semi-definite})$$
>
> 若 $\nabla^2 f(x) \succ 0$（正定），则 $f$ 是严格凸函数。

**典型例子**:
- **二次函数**: $f(x) = \frac{1}{2} x^\top P x + q^\top x + r$
  - $\nabla f(x) = Px + q$，$\nabla^2 f(x) = P$
  - 凸当且仅当 $P \succeq 0$
- **最小二乘目标**: $f(x) = \|Ax - b\|_2^2$
  - $\nabla f(x) = 2A^\top(Ax - b)$，$\nabla^2 f(x) = 2A^\top A$
  - 对任意 $A$ 都是凸函数（因为 $A^\top A \succeq 0$ 恒成立）

### 2.5 Jensen 不等式 (Jensen's Inequality)

> **Jensen 不等式**: 若 $f$ 是凸函数，对任意随机变量 $z$：
> $$f(\mathbb{E}[z]) \leq \mathbb{E}[f(z)]$$

**重要性**: Jensen 不等式在信息论、统计学习中极为常用，例如证明 KL 散度的非负性：$D_{KL}(p \| q) \geq 0$。

---

## 3. 凸优化问题 (Convex Optimization Problem)

### 3.1 优化问题标准形式

$$\min_{x \in \mathbb{R}^n} f_0(x)$$
$$\text{subject to} \quad f_i(x) \leq 0, \quad i = 1, \ldots, m$$
$$\quad\quad\quad\quad\quad h_i(x) = 0, \quad i = 1, \ldots, p$$

**变量说明**:
- $x \in \mathbb{R}^n$：优化变量
- $f_0: \mathbb{R}^n \to \mathbb{R}$：**目标函数 (objective function)** 或代价函数
- $f_i: \mathbb{R}^n \to \mathbb{R}$：**不等式约束函数 (inequality constraints)**
- $h_i: \mathbb{R}^n \to \mathbb{R}$：**等式约束函数 (equality constraints)**

### 3.2 最优目标值

$$p^* = \inf\{f_0(x) \mid f_i(x) \leq 0, \, h_i(x) = 0\}$$

- 若问题**不可行 (infeasible)**（无满足约束的 $x$），则 $p^* = \infty$
- 若问题**无下界 (unbounded below)**，则 $p^* = -\infty$

### 3.3 可行点与最优点

- **可行点 (feasible point)**: $x \in \text{dom}\, f_0$ 且满足所有约束。
- **最优点 (optimal point)**: 可行且 $f_0(x) = p^*$。
- **局部最优点 (locally optimal point)**: 存在 $r > 0$，使得在 $\|z - x\|_2 \leq r$ 的邻域内，$x$ 是最优点。

### 3.4 凸优化问题

> **定义**: 凸优化问题标准形式：
> $$\min f_0(x), \quad \text{s.t.} \quad f_i(x) \leq 0, \quad Ax = b$$
> 其中 $f_0, f_1, \ldots, f_m$ 均为**凸函数**，等式约束为**仿射函数**。

**重要性质**: 凸优化问题的可行集 (feasible set) 是**凸集**。

### 3.5 局部最优 = 全局最优（凸优化的核心定理）

> **定理**: 凸优化问题的**任何局部最优点都是全局最优点**。

**证明思路**:
1. 设 $x$ 是局部最优但非全局最优，则存在可行点 $y$ 使 $f_0(y) < f_0(x)$。
2. 构造 $z = \theta y + (1-\theta) x$，其中 $\theta = r / (2\|y - x\|_2)$，使 $z$ 在 $x$ 的局部邻域内。
3. 由凸函数性质: $f_0(z) \leq \theta f_0(y) + (1-\theta)f_0(x) < f_0(x)$，与 $x$ 是局部最优矛盾。

**考试要点**: 这个定理是凸优化最重要的性质——它保证了局部搜索方法（如梯度下降）在凸问题上可以找到全局最优解。

---

## 4. 无约束最小化：梯度下降法 (Unconstrained Minimization)

### 4.1 无约束凸最小化问题

$$\min f(x)$$

其中 $f$ 是凸函数，二阶连续可微，最优值 $p^* = \inf_x f(x)$ 有限且可达。

**目标**: 产生一系列点 $x^{(k)} \in \text{dom}\, f$，$k = 0, 1, \ldots$，使得 $f(x^{(k)}) \to p^*$。

### 4.2 一般下降法 (General Descent Method)

**一步更新**:
$$x^{(k+1)} = x^{(k)} + t^{(k)} \Delta x^{(k)}, \quad \text{使得} \quad f(x^{(k+1)}) < f(x^{(k)})$$

- $\Delta x$：搜索方向 (search direction)
- $t$：步长 (step size)

由一阶条件可知，$f(x^+) \geq f(x) + t \nabla f(x)^\top \Delta x$，因此：

> 若 $f(x^+) < f(x)$，则 $\nabla f(x)^\top \Delta x < 0$，即 $\Delta x$ 是**下降方向 (descent direction)**。

### 4.3 线搜索方法 (Line Search)

**精确线搜索 (Exact line search)**:
$$t = \arg\min_{t > 0} f(x + t\Delta x)$$

**回溯线搜索 (Backtracking line search)**（参数 $\alpha \in (0, 1/2)$，$\beta \in (0, 1)$）:
- 从 $t = 1$ 开始，重复 $t := \beta t$，直到：
  $$f(x + t\Delta x) < f(x) + \alpha t \nabla f(x)^\top \Delta x$$

### 4.4 梯度下降法 (Gradient Descent Method)

取 $\Delta x = -\nabla f(x)$（负梯度方向），即**最速下降方向**。

> **梯度下降算法**:
> 1. 给定初始点 $x \in \text{dom}\, f$
> 2. 重复：
>    - $\Delta x := -\nabla f(x)$
>    - 通过线搜索选择步长 $t$
>    - 更新: $x := x + t\Delta x$
> 3. 直到停止条件满足: $\|\nabla f(x)\|_2 \leq \epsilon$

**直觉**: 在当前点，沿负梯度方向（函数下降最快的方向）移动一小步。

**Learning Rate（学习率）$\alpha$ 的重要性**:
- $\alpha$ 太大：可能跳过最优点，导致震荡甚至发散。
- $\alpha$ 太小：收敛速度极慢。
- 实践中：通常使用固定学习率、学习率衰减，或 Adam 等自适应方法。

**注意**: 虽然这里考虑的是凸最小化，但梯度下降及其变体（如 SGD）也可直接用于非凸优化（如深度神经网络训练）。

### 4.5 梯度下降的收敛性分析

对于二次问题 $\min_x f(x) = \frac{1}{2}(x_1^2 + \gamma x_2^2)$（$\gamma > 0$）：

使用精确线搜索，从 $x^{(0)} = (\gamma, 1)$ 出发：

$$x_1^{(k)} = \gamma \left(\frac{\gamma-1}{\gamma+1}\right)^k, \quad x_2^{(k)} = \left(-\frac{\gamma+1}{\gamma-1}\right)^k$$

**条件数 (condition number)**: $\kappa = \gamma / 1$。当 $\gamma \gg 1$ 或 $\gamma \ll 1$ 时（条件数很大），收敛极慢，出现"之字形 (zigzag)"路径。

**常见陷阱**: 对于高度非各向同性 (ill-conditioned) 的问题，纯梯度下降收敛很慢，需要预处理 (preconditioning) 或二阶方法。

### 4.6 牛顿法 (Newton's Method)

梯度下降使用一阶信息，**牛顿法**利用二阶信息（Hessian）：

$$\Delta x_{\text{nt}} = -(\nabla^2 f(x))^{-1} \nabla f(x)$$

更新规则：$x^{(k+1)} = x^{(k)} - (\nabla^2 f(x^{(k)}))^{-1} \nabla f(x^{(k)})$

**优点**: 对于强凸问题，牛顿法具有**二次收敛 (quadratic convergence)**——离最优点越近，收敛越快。

**缺点**: 每步需要计算和求逆 Hessian 矩阵，计算复杂度为 $O(n^3)$，在高维场景（如深度学习）中不可行。

### 4.7 随机梯度下降 (SGD) 和 Mini-batch SGD

在机器学习中，目标函数通常是**所有样本损失之和**：
$$J(w) = \frac{1}{m} \sum_{i=1}^m \ell(x_i, y_i; w)$$

**批量梯度下降 (Batch GD)**: 每次用全部 $m$ 个样本计算梯度，计算开销大。

**随机梯度下降 (SGD)**: 每次随机选 1 个样本：
$$w \leftarrow w - \alpha \nabla_w \ell(x_i, y_i; w)$$

**Mini-batch SGD**: 每次随机选 $B$（batch size）个样本，平衡计算效率与梯度估计质量。

| 方法 | 每步计算量 | 梯度噪声 | 收敛速度 |
|------|-----------|---------|---------|
| Batch GD | $O(m)$ | 无噪声 | 稳定但慢 |
| SGD | $O(1)$ | 高噪声 | 快但震荡 |
| Mini-batch SGD | $O(B)$ | 中等噪声 | 折中方案 |

**SGD 的直觉**: 对于大数据集，完整计算梯度代价太高；单个样本的梯度是真实梯度的有噪声估计，但足够指引下降方向。

---

## 5. 有约束最小化：拉格朗日对偶与KKT条件

### 5.1 拉格朗日函数 (Lagrangian Function)

考虑一般最小化问题：
$$\min_{x \in \mathbb{R}^n} f(x), \quad \text{s.t.} \quad h_i(x) \leq 0, \; i=1,\ldots,m; \quad \ell_j(x) = 0, \; j=1,\ldots,r$$

> **拉格朗日函数 (Lagrangian)**:
> $$L(x, u, v) = f(x) + \sum_{i=1}^m u_i h_i(x) + \sum_{j=1}^r v_j \ell_j(x)$$

其中 $u_i \geq 0$ 是**不等式约束的拉格朗日乘子 (Lagrange multipliers)**，$v_j$ 是**等式约束的拉格朗日乘子**。

**直觉**: 将约束"吸收"进目标函数，用乘子惩罚违约行为。

### 5.2 拉格朗日对偶函数 (Lagrange Dual Function)

$$g(u, v) = \min_{x \in \mathbb{R}^n} L(x, u, v)$$

**对偶问题 (Dual problem)**:
$$\max_{u \in \mathbb{R}^m, v \in \mathbb{R}^r} g(u, v), \quad \text{s.t.} \quad u \geq 0$$

**弱对偶性 (Weak duality)**: 对偶最优值 $d^* \leq p^*$（原问题最优值）。

**强对偶性 (Strong duality)**: 在满足一定条件（如 Slater 条件）时，$d^* = p^*$。

### 5.3 KKT 条件 (Karush-Kuhn-Tucker Conditions)

> **KKT 条件**: 对于上述一般问题，KKT 条件是：
>
> 1. **平稳性 (Stationarity)**:
>    $$0 \in \partial f(x) + \sum_{i=1}^m u_i \partial h_i(x) + \sum_{j=1}^r v_j \partial \ell_j(x)$$
>
> 2. **互补松弛性 (Complementary Slackness)**:
>    $$u_i \cdot h_i(x) = 0, \quad \forall i$$
>
> 3. **原问题可行性 (Primal Feasibility)**:
>    $$h_i(x) \leq 0, \quad \ell_j(x) = 0, \quad \forall i, j$$
>
> 4. **对偶可行性 (Dual Feasibility)**:
>    $$u_i \geq 0, \quad \forall i$$

**KKT 条件的意义**:
- 对于**凸优化问题**（在强对偶性成立时），KKT 条件是最优性的**充要条件**。
- 互补松弛性意味着：若约束 $h_i(x) < 0$（不紧），则对应的乘子 $u_i = 0$；若 $u_i > 0$，则约束 $h_i(x) = 0$（紧）。

**在机器学习中的应用**:
- **支持向量机 (SVM)**: 用 KKT 条件推导支持向量 (support vectors) 和对偶形式。
- **K-means, PCA**: 也需要 Lagrangian 方法。

---

## 6. 优化与机器学习的关系

> 优化是机器学习的核心技术基础。

**在本课程中的应用**:

| 机器学习模型 | 优化工具 |
|------------|---------|
| 线性回归 (Linear Regression) | 凸最小化，梯度下降 |
| 逻辑回归 (Logistic Regression) | 凸最小化，梯度下降 |
| 支持向量机 (SVM) | Lagrangian，KKT 条件 |
| K-means | Lagrangian |
| 高斯混合模型 (GMM) | KKT 条件，EM 算法 |
| 主成分分析 (PCA) | Lagrangian，KKT 条件 |
| 深度神经网络 | SGD，Adam 等 |

**给定一个机器学习模型，你应当能够判断**:
1. 目标函数是凸的还是非凸的？
2. 是否存在局部最优？全局最优是否唯一？
3. 选用哪种优化方法最合适？

---

## 常见错误与考试注意点

1. **混淆仿射集和凸集**: 仿射集要求整条直线在集合内（$\theta \in \mathbb{R}$），凸集只要求线段（$\theta \in [0,1]$）。
2. **凸函数 vs 凸集**: 一个函数是凸函数，当且仅当其 **epigraph**（图像上方的区域）是凸集。不要混淆。
3. **二阶条件的适用条件**: 二阶条件要求函数二次可微；对不可微函数，需用次梯度 (subgradient) 方法。
4. **KKT 是必要条件还是充要条件**: 对一般非凸问题，KKT 只是必要条件；对凸问题（满足正则条件），是充要条件。
5. **互补松弛性**: 注意 $u_i h_i(x) = 0$ 的含义：乘子和约束至少一个为零，不能同时严格大于零。
6. **梯度下降的停止条件**: 通常是 $\|\nabla f(x)\|_2 \leq \epsilon$，而非损失值本身。

---

## 7. 补充资料

### 教材参考

- **Boyd & Vandenberghe "Convex Optimization"** (免费在线): 本节内容的主要来源。详见 [Stanford EE364a](https://web.stanford.edu/class/ee364a/lectures.html)（Lecture 01-06）。
- **Murphy "Machine Learning: A Probabilistic Perspective"** Ch. 7（线性回归优化）、Ch. 8（逻辑回归优化）。
- **Bishop "Pattern Recognition and Machine Learning" (PRML)** Ch. 1.2（最优化基础）、Appendix E（拉格朗日乘子法）。
- **Burkov "The Hundred-Page ML Book"** Ch. 4（基础数学，包含优化简介）。

### 扩展阅读

- **Gradient Descent Convergence**: 对于 $L$-Lipschitz 光滑凸函数，梯度下降的收敛速率为 $O(1/k)$；对强凸函数为线性收敛 $O(\rho^k)$，$\rho < 1$。
- **Infimum and Supremum**: [Wikipedia](https://en.wikipedia.org/wiki/Infimum_and_supremum)
- **Matrix Calculus**: [Wikipedia Matrix Calculus](https://en.wikipedia.org/wiki/Matrix_calculus)（分子布局 vs 分母布局要统一）
- **Adam, RMSProp 等自适应优化器**: 在深度学习中广泛使用，是 SGD 的扩展。
