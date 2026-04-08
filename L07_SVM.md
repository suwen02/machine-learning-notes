# Lecture 07: Support Vector Machine (SVM)
**DDA3020 Machine Learning — CUHK-SZ**  
**讲师**: Baoyuan Wu | **日期**: March 2/4/9, 2026

---

## 目录 (Table of Contents)

1. [直觉与动机 (Motivation)](#1-直觉与动机)
2. [推导一：最大间隔 (Derivation I: Large Margin)](#2-推导一最大间隔)
3. [推导二：Hinge Loss](#3-推导二hinge-loss)
4. [Lagrange对偶与KKT条件](#4-lagrange对偶与kkt条件)
5. [用Lagrange对偶求解SVM](#5-用lagrange对偶求解svm)
6. [软间隔SVM：Slack Variables](#6-软间隔svmslack-variables)
7. [核方法 (Kernel Methods)](#7-核方法)
8. [其他：多分类、与Logistic Regression比较](#8-其他多分类与logistic-regression比较)
9. [SVM优缺点总结](#9-svm优缺点总结)
10. [补充资料](#10-补充资料)

---

## 1. 直觉与动机

### 1.1 二元分类问题回顾

给定训练集 $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^m$，其中 $\mathbf{x}_i \in \mathbb{R}^n$，$y_i \in \{-1, +1\}$。

令 $\mathbf{w} = [w_0, w_1, \ldots, w_n]^\top$ 且将 $\mathbf{x}$ 扩增为 $[1, x_1, \ldots, x_n]^\top$，则假设函数为：

$$y = \text{Sgn}(f_\mathbf{w}(\mathbf{x})) = \text{Sgn}(\mathbf{w}^\top \mathbf{x})$$

**分类条件**：
- 若 $y_i = +1$，则要求 $\mathbf{w}^\top \mathbf{x}_i > 0$
- 若 $y_i = -1$，则要求 $\mathbf{w}^\top \mathbf{x}_i < 0$

### 1.2 为什么需要SVM？

**问题**：存在多条决策边界均能完美分隔数据，应该选哪一条？

- 标准 Logistic Regression 的目标函数（cross entropy loss）是 **convex** 但非 strictly convex，因此最优解不唯一
- 带 $\ell_2$ 正则化的 Logistic Regression 有唯一最优解，但需要调参 $\lambda$

**直觉**：选择"中间"那条决策边界——它距两类数据都最远，泛化能力最强。

### 1.3 间隔 (Margin) 的概念

> **关键定义**：**Margin（间隔）** 是决策边界到正类和负类中最近数据点的距离之和。

SVM 的核心思想：**选择具有最大间隔的决策边界**，故 SVM 也称为 **large margin classifier（最大间隔分类器）**。

**直觉意义**：间隔越大，模型对新样本的分类越鲁棒，过拟合风险越小。

---

## 2. 推导一：最大间隔

### 2.1 基本线性代数引理

**引理 1**：$\mathbf{w}$ 与超平面 $f_{\mathbf{w},b}(\mathbf{x}) = \mathbf{w}^\top \mathbf{x} + b = 0$ 正交。

**证明**：
1. 取超平面上任意两点 $\mathbf{x}_1, \mathbf{x}_2$
2. $\mathbf{w}^\top \mathbf{x}_1 + b = 0$ 且 $\mathbf{w}^\top \mathbf{x}_2 + b = 0$
3. 相减得 $\mathbf{w}^\top(\mathbf{x}_1 - \mathbf{x}_2) = 0$，即 $\mathbf{w} \perp (\mathbf{x}_1 - \mathbf{x}_2)$。$\square$

**命题 3（点到超平面距离）**：点 $\mathbf{x}$ 到超平面 $\mathbf{w}^\top \mathbf{x} + b = 0$ 的距离为

> $$d = \frac{|f_{\mathbf{w},b}(\mathbf{x})|}{\|\mathbf{w}\|} = \frac{|\mathbf{w}^\top \mathbf{x} + b|}{\|\mathbf{w}\|}$$

**证明**：设 $\mathbf{x} = \mathbf{x}_\perp + r \frac{\mathbf{w}}{\|\mathbf{w}\|}$，$|r|$ 即为距离。两边乘以 $\mathbf{w}^\top$ 并加 $b$：
- 左边：$\mathbf{w}^\top \mathbf{x} + b = f_{\mathbf{w},b}(\mathbf{x})$
- 右边：$\mathbf{w}^\top \mathbf{x}_\perp + r\|\mathbf{w}\| + b = 0 + r\|\mathbf{w}\|$

因此 $r = \frac{f_{\mathbf{w},b}(\mathbf{x})}{\|\mathbf{w}\|}$，距离 $= |r| = \frac{|f_{\mathbf{w},b}(\mathbf{x})|}{\|\mathbf{w}\|}$。$\square$

### 2.2 几何间隔 (Geometric Margin)

对所有训练数据，间隔定义为：

$$\gamma = \min_i \frac{|f_{\mathbf{w},b}(\mathbf{x}_i)|}{\|\mathbf{w}\|}$$

由于我们要求分类正确，且 $y_i \in \{+1, -1\}$，上式等价于：

$$\gamma = \min_i \frac{y_i f_{\mathbf{w},b}(\mathbf{x}_i)}{\|\mathbf{w}\|}$$

若 $f_{\mathbf{w},b}$ 在某个 $\mathbf{x}_i$ 上分类错误，则间隔为负。

### 2.3 优化问题的Formulation（Hard Margin SVM）

**目标**：最大化间隔：

$$\max_{\mathbf{w},b} \gamma = \max_{\mathbf{w},b} \min_i \frac{y_i(\mathbf{w}^\top \mathbf{x}_i + b)}{\|\mathbf{w}\|}$$

**关键观察**：当 $(\mathbf{w}, b)$ 被缩放因子 $c$ 缩放时，间隔不变：

$$\frac{y_i(c\mathbf{w}^\top \mathbf{x}_i + cb)}{\|c\mathbf{w}\|} = \frac{y_i(\mathbf{w}^\top \mathbf{x}_i + b)}{\|\mathbf{w}\|}$$

**固定缩放尺度**：令最近点满足：

$$y_{i^*}(\mathbf{w}^\top \mathbf{x}_{i^*} + b) = 1$$

则对所有数据有 $y_i(\mathbf{w}^\top \mathbf{x}_i + b) \geq 1$，且间隔为 $\frac{1}{\|\mathbf{w}\|}$。

最大化间隔等价于最小化 $\|\mathbf{w}\|$，得到：

> **Hard Margin SVM 优化问题**：
> $$\min_{\mathbf{w},b} \frac{1}{2}\|\mathbf{w}\|^2$$
> $$\text{s.t.} \quad y_i(\mathbf{w}^\top \mathbf{x}_i + b) \geq 1, \quad \forall i$$

**注**：用 $\frac{1}{2}\|\mathbf{w}\|^2$ 而非 $\|\mathbf{w}\|$ 是为了数学推导方便（去掉绝对值，令导数简洁）。

**支持向量 (Support Vectors)**：满足等号 $y_i(\mathbf{w}^\top \mathbf{x}_i + b) = 1$ 的数据点，即距决策边界最近的点。两侧支持向量之间的距离为 $\frac{2}{\|\mathbf{w}\|}$（即 2×间隔）。

---

## 3. 推导二：Hinge Loss

### 3.1 Logistic Regression 与 SVM 的联系

Logistic Regression 的目标函数（单样本）：

$$J(\mathbf{w}) = -\delta_{y=1}\log(f_{\mathbf{w},b}(\mathbf{x})) - \delta_{y=-1}\log(1 - f_{\mathbf{w},b}(\mathbf{x}))$$

SVM 将 Logistic loss 替换为代价函数 $\text{cost}_1$ 和 $\text{cost}_{-1}$：

$$\min_{\mathbf{w}} \; C \sum_{i=1}^m \left[\delta_{y_i=1} \cdot \text{cost}_1(\mathbf{w}^\top \mathbf{x}_i + b) + \delta_{y_i=-1} \cdot \text{cost}_{-1}(\mathbf{w}^\top \mathbf{x}_i + b)\right] + \frac{1}{2}\sum_{j=1}^n w_j^2$$

其中 $C = \frac{1}{\lambda}$ 是正则化超参数的倒数。

### 3.2 Hinge Loss

**SVM 的代价函数要求**：
- 若 $y_i = +1$，当 $\mathbf{w}^\top \mathbf{x}_i + b \geq 1$ 时损失为 0
- 若 $y_i = -1$，当 $\mathbf{w}^\top \mathbf{x}_i + b \leq -1$ 时损失为 0

因此，**Hinge Loss（合页损失）**为：

> $$\ell_{\text{hinge}}(\mathbf{x}_i, y_i) = \max\left(0, 1 - y_i(\mathbf{w}^\top \mathbf{x}_i + b)\right)$$

**直觉**：Hinge Loss 对正确分类且距边界足够远的样本损失为 0；对于错误分类或距边界太近的样本施加线性惩罚。

**与 Logistic Loss 的对比**：

| 特性 | Logistic Loss | Hinge Loss |
|------|---------------|------------|
| 类型 | 光滑 (smooth) | 非光滑，在 $y_i f=1$ 处不可微 |
| 对正确分类的行为 | 渐近减小（从不为 0） | 超过间隔时精确为 0 |
| 稀疏性 | 所有样本均参与 | 只有支持向量参与（稀疏解） |
| 概率输出 | 可给出概率 | 无直接概率解释 |

虽然 Hinge Loss 非光滑，但可以通过引入约束将目标函数转化为等价的光滑带约束优化问题（Hard Margin SVM 形式）。

---

## 4. Lagrange对偶与KKT条件

### 4.1 一般最小化问题

考虑一般约束优化问题：

$$\min_{\mathbf{x} \in \mathbb{R}^n} f(\mathbf{x})$$
$$\text{s.t.} \quad h_i(\mathbf{x}) \leq 0, \; i = 1,\ldots,m$$
$$\quad\quad\quad\; \ell_j(\mathbf{x}) = 0, \; j = 1,\ldots,r$$

### 4.2 Lagrange函数 (Lagrangian)

$$L(\mathbf{x}, \mathbf{u}, \mathbf{v}) = f(\mathbf{x}) + \sum_{i=1}^m u_i h_i(\mathbf{x}) + \sum_{j=1}^r v_j \ell_j(\mathbf{x})$$

### 4.3 对偶函数与对偶问题

**Lagrange 对偶函数**：

$$g(\mathbf{u}, \mathbf{v}) = \min_{\mathbf{x} \in \mathbb{R}^n} L(\mathbf{x}, \mathbf{u}, \mathbf{v})$$

**对偶问题 (Dual Problem)**：

$$\max_{\mathbf{u} \in \mathbb{R}^m, \mathbf{v} \in \mathbb{R}^r} g(\mathbf{u}, \mathbf{v}) \quad \text{s.t.} \; \mathbf{u} \geq \mathbf{0}$$

> **弱对偶性 (Weak Duality)**：对偶最优值 $\leq$ 原问题最优值。  
> **强对偶性 (Strong Duality)**：若满足约束规范条件（如 Slater 条件），两者相等。

### 4.4 KKT条件

最优解 $\mathbf{x}^*$, $\mathbf{u}^*$, $\mathbf{v}^*$ 需满足 **Karush-Kuhn-Tucker (KKT) 条件**：

| 条件 | 公式 |
|------|------|
| **Stationarity（驻点）** | $0 \in \partial f(\mathbf{x}) + \sum_i u_i \partial h_i(\mathbf{x}) + \sum_j v_j \partial \ell_j(\mathbf{x})$ |
| **Complementary Slackness（互补松弛）** | $u_i \cdot h_i(\mathbf{x}) = 0, \; \forall i$ |
| **Primal Feasibility（原可行性）** | $h_i(\mathbf{x}) \leq 0, \; \ell_j(\mathbf{x}) = 0, \; \forall i, j$ |
| **Dual Feasibility（对偶可行性）** | $u_i \geq 0, \; \forall i$ |

**互补松弛的直觉**：对于不等式约束 $h_i(\mathbf{x}) \leq 0$，要么约束 **严格满足**（$h_i < 0$，则 $u_i = 0$，该约束不起作用），要么约束 **刚好取等**（$h_i = 0$，则 $u_i \geq 0$，该约束起作用）。

**参考**：S. Boyd and L. Vandenberghe (2004), *Convex Optimization*, Cambridge University Press, Chapter 5.

---

## 5. 用Lagrange对偶求解SVM

### 5.1 构建Lagrange函数

SVM 原问题：

$$\min_{\mathbf{w},b} \frac{1}{2}\|\mathbf{w}\|^2 \quad \text{s.t.} \; 1 - y_i(\mathbf{w}^\top \mathbf{x}_i + b) \leq 0, \; \forall i$$

Lagrange 函数（引入乘子 $\alpha_i \geq 0$）：

$$L(\mathbf{w}, b, \boldsymbol{\alpha}) = \frac{1}{2}\|\mathbf{w}\|^2 + \sum_{i=1}^m \alpha_i \left(1 - y_i(\mathbf{w}^\top \mathbf{x}_i + b)\right)$$

### 5.2 KKT驻点条件

对 $L$ 关于 $\mathbf{w}$ 和 $b$ 求偏导并令其为零：

$$\frac{\partial L}{\partial \mathbf{w}} = 0 \implies \mathbf{w} = \sum_{i=1}^m \alpha_i y_i \mathbf{x}_i \tag{KKT-1}$$

$$\frac{\partial L}{\partial b} = 0 \implies \sum_{i=1}^m \alpha_i y_i = 0 \tag{KKT-2}$$

> **重要结论**：$\mathbf{w}$ 是训练样本的线性组合！$\mathbf{w} = \sum_i \alpha_i y_i \mathbf{x}_i$

### 5.3 推导对偶问题

将 KKT 驻点条件代入 Lagrange 函数（逐步推导）：

$$L(\mathbf{w}, b, \boldsymbol{\alpha}) = \frac{1}{2}\|\mathbf{w}\|^2 + \sum_i \alpha_i - \sum_{i,j} \alpha_i \alpha_j y_i y_j \mathbf{x}_i^\top \mathbf{x}_j - b\underbrace{\sum_i \alpha_i y_i}_{=0}$$

利用 $\mathbf{w} = \sum_i \alpha_i y_i \mathbf{x}_i$ 得 $\frac{1}{2}\|\mathbf{w}\|^2 = \frac{1}{2}\sum_{i,j}\alpha_i\alpha_j y_i y_j \mathbf{x}_i^\top \mathbf{x}_j$，因此：

$$L = \sum_i \alpha_i - \frac{1}{2}\sum_{i,j} \alpha_i \alpha_j y_i y_j \mathbf{x}_i^\top \mathbf{x}_j$$

> **SVM 对偶问题 (Dual Problem)**：
> $$\max_{\boldsymbol{\alpha}} \sum_{i=1}^m \alpha_i - \frac{1}{2}\sum_{i,j} \alpha_i \alpha_j y_i y_j \mathbf{x}_i^\top \mathbf{x}_j$$
> $$\text{s.t.} \quad \sum_{i=1}^m \alpha_i y_i = 0, \quad \alpha_i \geq 0, \; \forall i$$

**优点**：
1. 对偶问题揭示了问题的结构
2. 目标函数只涉及样本之间的**内积** $\mathbf{x}_i^\top \mathbf{x}_j$，为引入核函数奠定基础

### 5.4 支持向量 (Support Vectors)

由 KKT **互补松弛条件**：$\alpha_i(1 - y_i(\mathbf{w}^\top \mathbf{x}_i + b)) = 0$

- 若 $1 - y_i(\mathbf{w}^\top \mathbf{x}_i + b) < 0$（样本在间隔外），则 $\alpha_i = 0$，该样本**不参与**构造分类器
- 若 $1 - y_i(\mathbf{w}^\top \mathbf{x}_i + b) = 0$（样本恰好在间隔边界上），则 $\alpha_i \geq 0$

**支持集** $\mathcal{S} = \{i | \alpha_i > 0\}$，即支持向量的下标集合。

> **支持向量的意义**：只有支持向量决定了分类器 $\mathbf{w}$；其余样本即使移除，结果不变。这就是"support vector machine"得名的原因。

### 5.5 求解偏置参数 $b$

对任意支持向量 $\mathbf{x}_j, j \in \mathcal{S}$，有 $y_j(\mathbf{w}^\top \mathbf{x}_j + b) = 1$，即：

$$\sum_{i=1}^m \alpha_i y_i \mathbf{x}_i^\top \mathbf{x}_j + b = y_j$$

对所有支持向量取平均（提高数值稳定性）：

> $$b = \frac{1}{|\mathcal{S}|} \sum_{j \in \mathcal{S}} \left(y_j - \sum_{i=1}^m \alpha_i y_i \mathbf{x}_i^\top \mathbf{x}_j\right)$$

### 5.6 预测新数据

给定最优解 $\{\boldsymbol{\alpha}, \mathbf{w}, b\}$，对新数据 $\mathbf{x}$ 的预测为：

$$\mathbf{w}^\top \mathbf{x} + b = \sum_{i=1}^m \alpha_i y_i \mathbf{x}_i^\top \mathbf{x} + b$$

- 若 $\mathbf{w}^\top \mathbf{x} + b > 0$，预测为 $+1$；否则预测为 $-1$

**注意**：预测依赖于训练样本与新样本的内积——这为核方法的引入提供了直接动机。

---

## 6. 软间隔SVM：Slack Variables

### 6.1 问题动机

Hard Margin SVM 假设数据线性可分，但实际数据往往**有噪声**或**类间重叠**（non-separable），导致约束 $y_i(\mathbf{w}^\top \mathbf{x}_i + b) \geq 1$ 无法同时满足。

### 6.2 引入松弛变量 (Slack Variables)

引入 $\xi_i \geq 0$，允许部分样本违反间隔约束：

$$y_i(\mathbf{w}^\top \mathbf{x}_i + b) \geq 1 - \xi_i, \quad \forall i$$

- $\xi_i = 0$：样本在间隔外（正确分类，不违反约束）
- $0 < \xi_i \leq 1$：样本在间隔内但在正确侧（正确分类）
- $\xi_i > 1$：样本越过决策边界（分类错误）

> **Soft Margin SVM (C-SVM) 优化问题**：
> $$\min_{\mathbf{w},b,\boldsymbol{\xi}} \frac{1}{2}\|\mathbf{w}\|^2 + C\sum_{i=1}^m \xi_i$$
> $$\text{s.t.} \quad y_i(\mathbf{w}^\top \mathbf{x}_i + b) \geq 1 - \xi_i, \quad \xi_i \geq 0, \quad \forall i$$

**C 参数的意义**：
- $C$ 大：对间隔违反惩罚大，倾向于 hard margin（小间隔，可能过拟合）
- $C$ 小：对间隔违反容忍度高，允许更大间隔（大间隔，可能欠拟合）

### 6.3 KKT条件与对偶问题推导

Lagrange 函数（引入乘子 $\alpha_i \geq 0$, $\mu_i \geq 0$）：

$$L(\mathbf{w},b,\boldsymbol{\xi},\boldsymbol{\alpha},\boldsymbol{\mu}) = \frac{1}{2}\|\mathbf{w}\|^2 + C\sum_i \xi_i + \sum_i \left[\alpha_i(1-\xi_i - y_i(\mathbf{w}^\top \mathbf{x}_i + b)) + \mu_i(-\xi_i)\right]$$

**KKT 驻点条件**：

$$\frac{\partial L}{\partial \mathbf{w}} = 0 \implies \mathbf{w} = \sum_i \alpha_i y_i \mathbf{x}_i \tag{相同}$$

$$\frac{\partial L}{\partial b} = 0 \implies \sum_i \alpha_i y_i = 0 \tag{相同}$$

$$\frac{\partial L}{\partial \xi_i} = 0 \implies \alpha_i = C - \mu_i \implies 0 \leq \alpha_i \leq C \tag{新约束}$$

> **Soft Margin SVM 对偶问题**：
> $$\max_{\boldsymbol{\alpha}} \sum_{i=1}^m \alpha_i - \frac{1}{2}\sum_{i,j} \alpha_i \alpha_j y_i y_j \mathbf{x}_i^\top \mathbf{x}_j$$
> $$\text{s.t.} \quad \sum_{i=1}^m \alpha_i y_i = 0, \quad 0 \leq \alpha_i \leq C, \quad \forall i$$

**与 Hard Margin 对比**：对偶问题**形式完全相同**，仅将约束 $\alpha_i \geq 0$ 改为 $0 \leq \alpha_i \leq C$。

### 6.4 解的解释

| $\alpha_i$ 的值 | 含义 |
|----------------|------|
| $\alpha_i = 0$ | 样本在间隔外，不参与构造分类器 |
| $0 < \alpha_i < C$ | $\xi_i = 0$，样本在间隔边界上（支持向量） |
| $\alpha_i = C$ | $\xi_i \geq 0$，样本在间隔内部或越界 |

**求偏置 $b$**：令 $\mathcal{M} = \{i | 0 < \alpha_i < C\}$（精确在间隔边界上的支持向量，此时 $\xi_i = 0$）：

$$b = \frac{1}{|\mathcal{M}|}\sum_{j \in \mathcal{M}}\left(y_j - \sum_{i=1}^m \alpha_i y_i \mathbf{x}_i^\top \mathbf{x}_j\right)$$

---

## 7. 核方法

### 7.1 为什么需要核？

Hard/Soft Margin SVM 只能处理**线性可分**（或近似线性可分）的数据。对于非线性可分数据（如 XOR 问题、同心圆问题），可以通过**特征映射**将数据变换到高维空间：

$$f_{\mathbf{w},b}(\mathbf{x}) = \mathbf{w}^\top \phi(\mathbf{x}) + b$$

**问题**：$\phi(\mathbf{x})$ 维度可能极高（甚至无穷维），计算代价巨大。

### 7.2 核技巧 (Kernel Trick)

**关键洞察**：SVM 的对偶问题和预测仅涉及样本之间的内积 $\phi(\mathbf{x}_i)^\top \phi(\mathbf{x}_j)$，而非 $\phi(\mathbf{x})$ 本身！

定义**核函数 (Kernel Function)**：

> $$k(\mathbf{x}_i, \mathbf{x}_j) = \phi(\mathbf{x}_i)^\top \phi(\mathbf{x}_j)$$

只需能高效计算 $k(\mathbf{x}_i, \mathbf{x}_j)$，无需显式计算 $\phi(\mathbf{x})$。

**带核的 SVM 对偶问题**：

$$\max_{\boldsymbol{\alpha}} \sum_i \alpha_i - \frac{1}{2}\sum_{i,j} \alpha_i \alpha_j y_i y_j k(\mathbf{x}_i, \mathbf{x}_j)$$
$$\text{s.t.} \quad \sum_i \alpha_i y_i = 0, \quad \alpha_i \geq 0, \quad \forall i$$

**带核的预测**：

$$\mathbf{w}^\top \phi(\mathbf{x}) + b = \sum_i \alpha_i y_i k(\mathbf{x}_i, \mathbf{x}) + b$$

由于 $\boldsymbol{\alpha}$ 稀疏（只有支持向量有 $\alpha_i > 0$），此分类器也称为**稀疏核分类器**。

### 7.3 常用核函数

| 核函数 | 公式 | 特点 |
|--------|------|------|
| **Linear（线性核）** | $k(\mathbf{x}_i, \mathbf{x}_j) = \mathbf{x}_i^\top \mathbf{x}_j$ | 等价于标准线性 SVM |
| **Polynomial（多项式核）** | $k(\mathbf{x}_i, \mathbf{x}_j) = \left(1 + \frac{\mathbf{x}_i^\top \mathbf{x}_j}{\sigma^2}\right)^p$ | 对应 $p$ 阶多项式特征映射 |
| **RBF / Gaussian（高斯核）** | $k(\mathbf{x}_i, \mathbf{x}_j) = \exp\left(-\frac{\|\mathbf{x}_i - \mathbf{x}_j\|^2}{2\sigma^2}\right)$ | 最常用，对应无穷维特征空间 |
| **Sigmoid** | $k(\mathbf{x}_i, \mathbf{x}_j) = \frac{1}{1 + \exp\left(-\frac{\mathbf{x}_i^\top \mathbf{x}_j + b}{\sigma^2}\right)}$ | 与神经网络有关联 |

**RBF 核的直觉**：两个样本越"相似"（距离越近），核值越大（接近1）；越"不相似"（距离越远），核值越小（接近0）。$\sigma$ 控制相似度的"带宽"。

**多项式核的直觉**：$k(\mathbf{x}_i, \mathbf{x}_j) = \left(1 + \frac{\mathbf{x}_i^\top \mathbf{x}_j}{\sigma^2}\right)^p$ 对应于所有次数最高为 $p$ 的多项式特征之间的内积，但计算只需 $O(n)$，而显式计算需要 $O(n^p)$。

### 7.4 Mercer条件

什么样的函数 $k(\cdot, \cdot)$ 是合法的核函数？

> **Mercer 定理**：函数 $k(\mathbf{x}, \mathbf{z})$ 是有效核函数，当且仅当对任意有限样本集 $\{\mathbf{x}_1, \ldots, \mathbf{x}_m\}$，对应的 **Gram 矩阵** $K$（其中 $K_{ij} = k(\mathbf{x}_i, \mathbf{x}_j)$）是**半正定 (positive semi-definite)** 的。

- 线性核、多项式核（$p$ 为正整数）、RBF 核均满足 Mercer 条件
- Sigmoid 核在某些参数下不满足 Mercer 条件，但实际中仍被使用

**实践提示**：使用 RBF 核时，务必先进行**特征缩放 (feature scaling)**，否则不同量纲的特征会导致距离计算失真。

---

## 8. 其他：多分类与Logistic Regression比较

### 8.1 SVM多分类策略

SVM 原生仅支持二元分类，扩展到 $K$ 类有两种主要策略：

#### One-vs-All (OvA) / One-vs-Rest (OvR)

**训练**：训练 $K$ 个 SVM，第 $k$ 个 SVM 将第 $k$ 类与其余所有类分开，得到参数 $(\mathbf{w}^{(k)}, b^{(k)})$。

**预测**：选择置信度最高的分类器：

> $$\hat{y} = \arg\max_{k \in \{1,\ldots,K\}} \left[(\mathbf{w}^{(k)})^\top \mathbf{x} + b^{(k)}\right]$$

#### One-vs-One (OvO)

**训练**：训练 $\binom{K}{2}$ 个二元 SVM，每个处理一对类别。

**预测**：投票（每个 SVM 对测试样本投票，选票最多的类别获胜）。

| 策略 | 训练时间 | 预测时间 | 内存 |
|------|---------|---------|------|
| OvA | $O(K)$ | $O(K)$ | $O(K)$ |
| OvO | $O(K^2)$ | $O(K^2)$ | $O(K^2)$ |
| OvO 优点 | 每次只需在两类小数据上训练，更快 | 更稳定 | — |

### 8.2 SVM vs. Logistic Regression 使用建议

令 $n$ 为特征数量，$m$ 为训练样本数量：

| 场景 | 推荐算法 | 原因 |
|------|---------|------|
| $n$ 大，$m$ 中等或小 | LR 或 SVM（无核） | 数据可能线性可分 |
| $n$ 小，$m$ 中等 | SVM + Gaussian kernel | 非线性，核方法有效 |
| $n$ 小，$m$ 大 | 增加特征，再用 LR 或 SVM（无核） | 核方法在大样本下计算慢 |

**Loss 函数对比**：

$$\text{SVM: Hinge loss} = \left(1 - (\mathbf{w}^\top \mathbf{x}_i + b)y_i\right)_+$$

$$\text{LR: Log loss} = \log\left(1 + e^{-(\mathbf{w}^\top \mathbf{x}_i + b)y_i}\right)$$

---

## 9. SVM优缺点总结

### 优点

1. **有效处理高维数据**：由于对偶问题只依赖内积，在维度远大于样本数时仍然有效
2. **全局最优解**：SVM 的优化问题是凸的，保证找到全局最优
3. **核方法的灵活性**：通过选择不同核函数，能处理各种非线性模式
4. **鲁棒性**：最终解只依赖支持向量，不受非支持向量样本的影响
5. **泛化能力强**：最大间隔原则有理论保证（VC 维、结构风险最小化）

### 缺点

1. **大规模数据效率低**：训练复杂度为 $O(m^2)$ 至 $O(m^3)$，对大数据集不友好
2. **多类分类需要技巧**：不直接支持多分类，需使用 OvA 或 OvO
3. **核函数和 $C$ 的选择**：需要交叉验证调参，计算代价高
4. **不直接输出概率**：需要 Platt Scaling 等额外处理
5. **特征缩放敏感**：使用 RBF 核前必须归一化特征

### SVM 其他变体

- **Semi-supervised SVM**：利用未标注数据
- **Structured SVM**：处理结构化输出（如序列标注）
- **Latent Variable SVM**：处理隐变量
- **SVR (Support Vector Regression)**：用于回归任务

---

## 10. 补充资料

### 教材参考

- **Bishop "PRML"**：Chapter 7.1 — 线性 SVM；Chapter 7.2 — Kernel SVM（*Pattern Recognition and Machine Learning*, Springer, 2006）
- **Murphy "MLAPP"**：Chapter 14.5 — SVMs（*Machine Learning: A Probabilistic Perspective*, MIT Press, 2012）
- **Burkov "Hundred-Page ML Book"**：Chapter 3.4 — SVM（极简版本，适合快速回顾）

### 在线资源

- **Andrew Ng's CS229 Notes on SVM**: https://see.stanford.edu/materials/aimlcs229/cs229-notes3.pdf
- **KKT Conditions (CMU)**: https://www.stat.cmu.edu/~ryantibs/convexopt-S15/scribes/12-kkt-pdf
- **SVM Tutorial (Nian)**: https://nianlonggu.com/2019/06/07/tutorial-on-SVM/
- **凸优化教材**: S. Boyd & L. Vandenberghe, *Convex Optimization*, Chapter 5

### 实现

- **LibSVM**: 最广泛使用的 SVM 实现，支持多核函数
- **scikit-learn SVC**: `sklearn.svm.SVC`，封装 LibSVM，参数 `C`, `kernel`, `gamma`

### 公式速查

| 公式 | 内容 |
|------|------|
| $\min_{\mathbf{w},b} \frac{1}{2}\|\mathbf{w}\|^2$，s.t. $y_i(\mathbf{w}^\top \mathbf{x}_i+b)\geq 1$ | Hard Margin SVM |
| $\min_{\mathbf{w},b,\xi} \frac{1}{2}\|\mathbf{w}\|^2 + C\sum_i\xi_i$ | Soft Margin SVM |
| $\mathbf{w} = \sum_i \alpha_i y_i \mathbf{x}_i$ | 原问题解 |
| $k(\mathbf{x}_i,\mathbf{x}_j) = \exp(-\|\mathbf{x}_i-\mathbf{x}_j\|^2 / 2\sigma^2)$ | RBF 核 |
| $\hat{y} = \text{Sgn}\left(\sum_i \alpha_i y_i k(\mathbf{x}_i, \mathbf{x}) + b\right)$ | 核 SVM 预测 |
