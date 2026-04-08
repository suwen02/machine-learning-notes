# DDA3020 机器学习 — Lecture 3: Linear Algebra（线性代数）

> **课程**: DDA3020 Machine Learning, CUHK-SZ  
> **讲师**: Baoyuan Wu  
> **日期**: 2026年1月12/14日

---

## 目录 (Table of Contents)

1. [标量、向量与矩阵](#1-标量向量与矩阵)
2. [向量和矩阵的基本运算](#2-向量和矩阵的基本运算)
3. [向量范数 (Vector Norms)](#3-向量范数-vector-norms)
4. [矩阵范数 (Matrix Norms)](#4-矩阵范数-matrix-norms)
5. [矩阵的逆 (Matrix Inverse)](#5-矩阵的逆-matrix-inverse)
6. [线性相关与线性无关](#6-线性相关与线性无关-linear-dependence--independence)
7. [线性方程组 (Systems of Linear Equations)](#7-线性方程组-systems-of-linear-equations)
8. [特征值与特征向量（补充）](#8-特征值与特征向量-eigenvalues--eigenvectors-补充)
9. [矩阵分解：Eigendecomposition 与 SVD（补充）](#9-矩阵分解eigendecomposition-与-svd-补充)
10. [正定矩阵（补充）](#10-正定矩阵-positive-definite-matrices-补充)
11. [向量与矩阵微积分（补充）](#11-向量与矩阵微积分-vectormatrix-calculus-补充)
12. [线性代数与 ML 的关联](#12-线性代数与-ml-的关联)
13. [补充资料](#13-补充资料-supplementary-resources)

---

## 1. 标量、向量与矩阵

### 1.1 标量 (Scalar)

**标量**是一个简单的数值，如 $15$ 或 $-3.2$。

- 用斜体小写字母表示，如 $x$ 或 $a$
- 本课程聚焦于实数（Real Numbers）

**符号约定**:

$$\sum_{i=1}^m x_i = x_1 + x_2 + \ldots + x_m \qquad \prod_{i=1}^m x_i = x_1 \cdot x_2 \cdots x_m$$

---

### 1.2 向量 (Vector)

**向量**是标量值的有序列表（属性列表）。

- 用**加粗小写字母**表示，如 $\mathbf{x}$ 或 $\mathbf{w}$
- 通常写成**列向量**形式（column-wise）：

$$\mathbf{a} = \begin{bmatrix} 2 \\ 3 \end{bmatrix}, \quad \mathbf{b} = \begin{bmatrix} -2 \\ 5 \end{bmatrix}, \quad \mathbf{c} = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$$

- 向量既可以可视化为**有方向的箭头**（指向某方向），也可以视为**多维空间中的点**

**向量元素的表示**:

$$\mathbf{a} = \begin{bmatrix} a^{(1)} \\ a^{(2)} \end{bmatrix} = \begin{bmatrix} a_1 \\ a_2 \end{bmatrix} = \begin{bmatrix} 2 \\ 3 \end{bmatrix}$$

> **注意**: $x^{(j)}$ 表示向量 $\mathbf{x}$ 的第 $j$ 个分量，**不是幂次**。若要表示平方，写作 $(x^{(j)})^2$。

---

### 1.3 矩阵 (Matrix)

**矩阵**是数字排列成的矩形阵列（行与列）。

- 用**加粗大写字母**表示，如 $\mathbf{X}$ 或 $\mathbf{W}$
- 一个 $2 \times 3$ 矩阵（2行3列）的例子：

$$\mathbf{X} = \begin{bmatrix} 2 & -3 & -6 \\ 4 & 21 & -1 \end{bmatrix}$$

**扩展索引表示**: 神经网络中可用多重索引，如 $x_{l,u}^{(j)}$ 表示第 $l$ 层第 $u$ 个单元的第 $j$ 个输入特征。

> **ML 中的意义**: 在 ML 中，数据集通常表示为矩阵 $\mathbf{X} \in \mathbb{R}^{N \times D}$，其中 $N$ 为样本数，$D$ 为特征维度。每一行是一个样本的特征向量。

---

## 2. 向量和矩阵的基本运算

### 2.1 向量运算

**向量加法与减法**（逐元素运算）:

$$\mathbf{x} + \mathbf{y} = \begin{bmatrix} x_1 \\ x_2 \end{bmatrix} + \begin{bmatrix} y_1 \\ y_2 \end{bmatrix} = \begin{bmatrix} x_1 + y_1 \\ x_2 + y_2 \end{bmatrix}$$

**标量乘法 (Scalar Multiplication)**:

$$a\mathbf{x} = a \begin{bmatrix} x_1 \\ x_2 \end{bmatrix} = \begin{bmatrix} ax_1 \\ ax_2 \end{bmatrix}, \qquad \frac{1}{a}\mathbf{x} = \begin{bmatrix} x_1/a \\ x_2/a \end{bmatrix}$$

---

### 2.2 转置 (Transpose)

**向量转置**（列向量变行向量）:

$$\mathbf{x} = \begin{bmatrix} x_1 \\ x_2 \end{bmatrix}, \quad \mathbf{x}^\top = \begin{bmatrix} x_1 & x_2 \end{bmatrix}$$

**矩阵转置**（行列互换）:

$$\mathbf{X} = \begin{bmatrix} x_{1,1} & x_{1,2} & x_{1,3} \\ x_{2,1} & x_{2,2} & x_{2,3} \\ x_{3,1} & x_{3,2} & x_{3,3} \end{bmatrix}, \quad \mathbf{X}^\top = \begin{bmatrix} x_{1,1} & x_{2,1} & x_{3,1} \\ x_{1,2} & x_{2,2} & x_{3,2} \\ x_{1,3} & x_{2,3} & x_{3,3} \end{bmatrix}$$

即 $(\mathbf{X}^\top)_{ij} = \mathbf{X}_{ji}$。

---

### 2.3 内积 / 点积 (Dot Product / Inner Product)

> $$\mathbf{x} \cdot \mathbf{y} = \mathbf{x}^\top \mathbf{y} = \begin{bmatrix} x_1 & x_2 \end{bmatrix} \begin{bmatrix} y_1 \\ y_2 \end{bmatrix} = x_1 y_1 + x_2 y_2$$

**几何意义**: $\mathbf{x}^\top \mathbf{y} = \|\mathbf{x}\|_2 \|\mathbf{y}\|_2 \cos\theta$，其中 $\theta$ 为两向量夹角。

---

### 2.4 矩阵的迹 (Trace)

对于**方阵** $\mathbf{X} \in \mathbb{R}^{n \times n}$，迹定义为对角线元素之和：

> $$\text{tr}(\mathbf{X}) = \sum_{i=1}^n x_{ii}$$

---

### 2.5 矩阵-向量乘积 (Matrix-Vector Product)

$\mathbf{X}\mathbf{w}$，其中 $\mathbf{X} \in \mathbb{R}^{m \times d}$，$\mathbf{w} \in \mathbb{R}^d$，结果为 $\mathbb{R}^m$ 中的向量：

$$\mathbf{X}\mathbf{w} = \begin{bmatrix} \sum_{j} x_{1,j} w_j \\ \sum_{j} x_{2,j} w_j \\ \vdots \\ \sum_{j} x_{m,j} w_j \end{bmatrix}$$

> **ML 直觉**: 线性模型的预测 $\hat{\mathbf{y}} = \mathbf{X}\mathbf{w}$ 即为矩阵-向量乘积，$\mathbf{X}$ 是设计矩阵，$\mathbf{w}$ 是权重向量。

---

### 2.6 矩阵-矩阵乘积 (Matrix-Matrix Product)

$\mathbf{X}\mathbf{W}$，其中 $\mathbf{X} \in \mathbb{R}^{m \times d}$，$\mathbf{W} \in \mathbb{R}^{d \times h}$，结果为 $\mathbb{R}^{m \times h}$：

$$(\mathbf{X}\mathbf{W})_{i,k} = \sum_{j=1}^d x_{i,j} w_{j,k}$$

> **注意**: 矩阵乘法**不满足交换律**，即一般情况下 $\mathbf{X}\mathbf{W} \neq \mathbf{W}\mathbf{X}$。但满足结合律：$(\mathbf{A}\mathbf{B})\mathbf{C} = \mathbf{A}(\mathbf{B}\mathbf{C})$。

---

## 3. 向量范数 (Vector Norms)

**范数（Norm）** $\|\cdot\|$ 是将向量映射到非负实数的函数，衡量向量的"大小"或"长度"。

### 3.1 范数的三个公理

> 1. **正定性 (Positivity)**: $\|\mathbf{x}\| = 0 \iff \mathbf{x} = \mathbf{0}$
> 2. **齐次性 (Homogeneity)**: $\|\lambda \mathbf{x}\| = |\lambda| \|\mathbf{x}\|$
> 3. **三角不等式 (Triangle Inequality)**: $\|\mathbf{x} + \mathbf{y}\| \leq \|\mathbf{x}\| + \|\mathbf{y}\|$

### 3.2 常见向量范数

| 范数 | 公式 | 特点 |
|------|------|------|
| **$\ell_2$-范数**（Euclidean Norm）| $\|\mathbf{x}\|_2 = \sqrt{\sum_{i=1}^d x_i^2}$ | 欧氏距离，最常用 |
| **$\ell_1$-范数**（Manhattan Norm）| $\|\mathbf{x}\|_1 = \sum_{i=1}^d |x_i|$ | 促进稀疏性（Sparsity） |
| **$\ell_p$-范数**（$p \geq 1$）| $\|\mathbf{x}\|_p = \left(\sum_{i=1}^d |x_i|^p\right)^{1/p}$ | 一般化形式 |
| **$\ell_0$-范数**（Sparsity）| $\|\mathbf{x}\|_0 = $ 非零元素个数 | 计算 NP-hard，常用 $\ell_1$ 近似 |

> **ML 应用**:
> - **L2 正则化 (Ridge)**: 损失函数添加 $\lambda \|\mathbf{w}\|_2^2$，防止权重过大，对应 Gaussian 先验（MAP 估计）
> - **L1 正则化 (Lasso)**: 损失函数添加 $\lambda \|\mathbf{w}\|_1$，促进权重稀疏，对应 Laplace 先验

---

## 4. 矩阵范数 (Matrix Norms)

矩阵范数除满足向量范数三公理外，还可满足**次乘性 (Sub-multiplicative Property)**:

$$\|\mathbf{X}\mathbf{Y}\| \leq \|\mathbf{X}\| \|\mathbf{Y}\|$$

### 4.1 Frobenius 范数

> $$\|\mathbf{X}\|_F = \sqrt{\sum_{i=1}^m \sum_{j=1}^n x_{ij}^2}$$

即将矩阵所有元素的平方和求根，相当于将矩阵"拉直"后的 $\ell_2$ 范数。

### 4.2 谱范数 (Spectral Norm)

> $$\|\mathbf{X}\|_2 = \sigma_{\max}(\mathbf{X})$$

即矩阵最大奇异值（Largest Singular Value）。

谱范数与 SVD 密切相关：若 $\mathbf{X} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$，则 $\boldsymbol{\Sigma} = \text{diag}(\sigma_1, \sigma_2, \ldots)$。

---

## 5. 矩阵的逆 (Matrix Inverse)

### 5.1 定义

一个 $d \times d$ 方阵 $\mathbf{A}$ 称为**可逆矩阵**（非奇异矩阵，invertible / nonsingular），若存在 $d \times d$ 方阵 $\mathbf{B}$ 使得：

> $$\mathbf{A}\mathbf{B} = \mathbf{B}\mathbf{A} = \mathbf{I}$$

则称 $\mathbf{B} = \mathbf{A}^{-1}$ 为 $\mathbf{A}$ 的**逆矩阵**，其中 $\mathbf{I}$ 为单位矩阵（对角线全为1，其余为0）。

### 5.2 计算方法

#### 行列式 (Determinant) 与伴随矩阵 (Adjugate)

$$\mathbf{A}^{-1} = \frac{1}{\det(\mathbf{A})} \text{adj}(\mathbf{A})$$

**二阶矩阵行列式**:

$$\det(\mathbf{A}) = |\mathbf{A}| = \begin{vmatrix} a & b \\ c & d \end{vmatrix} = ad - bc$$

**余子式与代数余子式**:
- $M_{i,j}$（Minor）: 删去第 $i$ 行第 $j$ 列后的子矩阵的行列式
- $C_{i,j}$（Cofactor）: $C_{i,j} = M_{i,j} \times (-1)^{i+j}$
- $\text{adj}(\mathbf{A}) = \mathbf{C}^\top$（代数余子式矩阵的转置）

**二阶矩阵逆的例子**:

$$\mathbf{A} = \begin{bmatrix} a & b \\ c & d \end{bmatrix} \Rightarrow \mathbf{A}^{-1} = \frac{1}{ad-bc} \begin{bmatrix} d & -b \\ -c & a \end{bmatrix}$$

#### 通过 SVD 计算逆矩阵

若 $\mathbf{A} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$，则：

> $$\mathbf{A}^{-1} = \mathbf{V}\boldsymbol{\Sigma}^{-1}\mathbf{U}^\top, \qquad \boldsymbol{\Sigma}^{-1} = \text{diag}(\sigma_1^{-1}, \sigma_2^{-1}, \ldots)$$

**通过 SVD 计算行列式**:

> $$\det(\mathbf{A}) = \prod_i \sigma_i$$

> **直觉解释**: 当 $\det(\mathbf{A}) = 0$（即某个奇异值为0）时，矩阵不可逆（奇异矩阵，singular matrix）。这在线性方程组中意味着方程组无唯一解。

---

## 6. 线性相关与线性无关 (Linear Dependence & Independence)

### 6.1 定义

设 $d$ 维向量集合 $\{\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_m\}$（$m \geq 1$）：

**线性相关 (Linearly Dependent)**:

> 若存在**不全为零**的系数 $\beta_1, \ldots, \beta_m$，使得
> $$\beta_1 \mathbf{x}_1 + \beta_2 \mathbf{x}_2 + \ldots + \beta_m \mathbf{x}_m = \mathbf{0}$$
> 则称这组向量线性相关。

**线性无关 (Linearly Independent)**:

> 若上式**仅当** $\beta_1 = \beta_2 = \ldots = \beta_m = 0$ 时成立，则称这组向量线性无关。

### 6.2 几何直觉

- **二维空间**: 两个向量线性无关 ⟺ 它们不共线（方向不同）
- **三维空间**: 三个向量线性无关 ⟺ 它们不共面

### 6.3 与矩阵可逆性的关系

> **定理**: 若矩阵 $\mathbf{X}$ 的所有行（或列）线性无关，则 $\mathbf{X}$ 可逆（当 $\mathbf{X}$ 为方阵时）。

> **ML 含义**: 特征之间的线性相关（multicollinearity/多重共线性）会导致矩阵 $\mathbf{X}^\top\mathbf{X}$ 接近奇异，使线性回归的参数估计不稳定。正则化（如 Ridge Regression）正是解决此问题的手段之一。

---

## 7. 线性方程组 (Systems of Linear Equations)

### 7.1 矩阵形式

考虑 $m$ 个线性方程、$d$ 个未知数 $w_1, \ldots, w_d$ 的方程组：

$$x_{1,1}w_1 + x_{1,2}w_2 + \ldots + x_{1,d}w_d = y_1$$
$$x_{2,1}w_1 + x_{2,2}w_2 + \ldots + x_{2,d}w_d = y_2$$
$$\vdots$$
$$x_{m,1}w_1 + x_{m,2}w_2 + \ldots + x_{m,d}w_d = y_m$$

用矩阵-向量形式紧凑表示：

> $$\mathbf{X}\mathbf{w} = \mathbf{y}$$

其中 $\mathbf{X} \in \mathbb{R}^{m \times d}$，$\mathbf{w} \in \mathbb{R}^d$，$\mathbf{y} \in \mathbb{R}^m$。

---

### 7.2 三种情形

#### 情形 (i)：方程组（Even-determined System），$m = d$

$\mathbf{X}$ 为方阵。若 $\mathbf{X}$ 可逆（行或列线性无关），则存在唯一解：

> $$\mathbf{w} = \mathbf{X}^{-1}\mathbf{y}$$

**例题**（2个方程，2个未知数）:

$$w_1 + w_2 = 4, \quad w_1 - 2w_2 = 1$$

$$\mathbf{w} = \begin{bmatrix} 1 & 1 \\ 1 & -2 \end{bmatrix}^{-1} \begin{bmatrix} 4 \\ 1 \end{bmatrix} = \begin{bmatrix} 3 \\ 1 \end{bmatrix}$$

---

#### 情形 (ii)：超定方程组（Over-determined System），$m > d$

方程数 > 未知数，**无精确解**（$\mathbf{X}$ 非方阵，不可逆）。

但可求**近似最优解（Least Squares Solution）**，利用**左逆 (Left-inverse)**：

$$\mathbf{X}^\dagger = (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top \quad \text{（当 }\mathbf{X}^\top\mathbf{X}\text{ 可逆时）}$$

> $$\mathbf{w} = \mathbf{X}^\dagger \mathbf{y} = (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top \mathbf{y}$$

满足 $\mathbf{X}^\dagger \mathbf{X} = \mathbf{I}$（左逆）。

**ML 重要联系**: 线性回归（Linear Regression）的解析解正是此形式！

$$\hat{\mathbf{w}} = (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top \mathbf{y}$$

这称为**普通最小二乘法 (OLS, Ordinary Least Squares)** 的闭合解（closed-form solution）。

**例题**（3个方程，2个未知数）:

$$w_1 + w_2 = 1, \quad w_1 - w_2 = 0, \quad w_1 = 2$$

$$\mathbf{w} = (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y} = \begin{bmatrix} 1 \\ 0.5 \end{bmatrix} \quad \text{（近似解）}$$

---

#### 情形 (iii)：欠定方程组（Under-determined System），$m < d$

方程数 < 未知数，存在**无穷多个解**。

利用**右逆 (Right-inverse)**：

$$\mathbf{X}^\dagger = \mathbf{X}^\top(\mathbf{X}\mathbf{X}^\top)^{-1} \quad \text{（当 }\mathbf{X}\mathbf{X}^\top\text{ 可逆时）}$$

**推导**:

设 $\mathbf{w} = \mathbf{X}^\top \mathbf{a}$（约束搜索方向），则：

$$\mathbf{X}\mathbf{X}^\top \mathbf{a} = \mathbf{y} \Rightarrow \mathbf{a} = (\mathbf{X}\mathbf{X}^\top)^{-1}\mathbf{y}$$

$$\mathbf{w} = \mathbf{X}^\top\mathbf{a} = \underbrace{\mathbf{X}^\top(\mathbf{X}\mathbf{X}^\top)^{-1}}_{\mathbf{X}^\dagger} \mathbf{y}$$

得到一个**最小范数解 (Minimum-norm Solution)**（满足 $\mathbf{X}\mathbf{w} = \mathbf{y}$ 的 $\ell_2$ 范数最小解）。

**例题**（2个方程，3个未知数）:

$$w_1 + 2w_2 + 3w_3 = 2, \quad w_1 - 2w_2 + 3w_3 = 1$$

$$\mathbf{w} = \mathbf{X}^\top(\mathbf{X}\mathbf{X}^\top)^{-1}\mathbf{y} = \begin{bmatrix} 0.15 \\ 0.25 \\ 0.45 \end{bmatrix} \quad \text{（约束解）}$$

**特殊情况（无解）**: 若 $\mathbf{X}\mathbf{X}^\top$ 和 $\mathbf{X}^\top\mathbf{X}$ 均不可逆，则方程组无解。

---

### 7.3 伪逆 (Moore-Penrose Pseudoinverse)

统一上述三种情形，定义**伪逆（Moore-Penrose Pseudoinverse）** $\mathbf{X}^\dagger$：

- Over-determined: $\mathbf{X}^\dagger = (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top$（最小二乘解）
- Under-determined: $\mathbf{X}^\dagger = \mathbf{X}^\top(\mathbf{X}\mathbf{X}^\top)^{-1}$（最小范数解）
- 通过 SVD 可统一计算：若 $\mathbf{X} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$，则 $\mathbf{X}^\dagger = \mathbf{V}\boldsymbol{\Sigma}^\dagger\mathbf{U}^\top$

---

## 8. 特征值与特征向量 (Eigenvalues & Eigenvectors) [补充]

### 8.1 定义

对于 $n \times n$ 方阵 $\mathbf{A}$，若存在非零向量 $\mathbf{v}$ 和标量 $\lambda$，使得：

> $$\mathbf{A}\mathbf{v} = \lambda\mathbf{v}$$

则称 $\lambda$ 为**特征值 (eigenvalue)**，$\mathbf{v}$ 为对应的**特征向量 (eigenvector)**。

**几何解释**: 矩阵 $\mathbf{A}$ 作用于特征向量 $\mathbf{v}$ 时，只改变其**长度**（缩放 $\lambda$ 倍），不改变其**方向**（或反向，当 $\lambda < 0$）。

### 8.2 特征多项式

特征值由**特征方程**求得：

$$\det(\mathbf{A} - \lambda\mathbf{I}) = 0$$

### 8.3 性质

- $n \times n$ 矩阵有 $n$ 个特征值（含重复，计入复数域）
- $\text{tr}(\mathbf{A}) = \sum_i \lambda_i$（迹等于特征值之和）
- $\det(\mathbf{A}) = \prod_i \lambda_i$（行列式等于特征值之积）
- 实对称矩阵的特征值均为实数，特征向量两两正交

---

## 9. 矩阵分解：Eigendecomposition 与 SVD [补充]

### 9.1 特征分解 (Eigendecomposition)

若 $\mathbf{A}$ 是 $n \times n$ **可对角化**矩阵（有 $n$ 个线性无关特征向量），则：

> $$\mathbf{A} = \mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^{-1}$$

其中 $\mathbf{Q}$ 的列为特征向量，$\boldsymbol{\Lambda} = \text{diag}(\lambda_1, \ldots, \lambda_n)$。

**实对称矩阵的特征分解**（$\mathbf{A} = \mathbf{A}^\top$）:

$$\mathbf{A} = \mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^\top, \quad \mathbf{Q}^\top\mathbf{Q} = \mathbf{I} \text{（正交矩阵）}$$

> **ML 应用**: PCA（主成分分析，W13讲授）利用协方差矩阵的特征分解找到数据方差最大的方向（主成分）。

---

### 9.2 奇异值分解 (SVD, Singular Value Decomposition)

SVD 是比特征分解更通用的矩阵分解，适用于任意 $m \times d$ 矩阵 $\mathbf{X}$：

> $$\mathbf{X} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$$

其中：
- $\mathbf{U} \in \mathbb{R}^{m \times m}$：**左奇异向量矩阵**，列为 $\mathbf{X}\mathbf{X}^\top$ 的特征向量，$\mathbf{U}^\top\mathbf{U} = \mathbf{I}$
- $\boldsymbol{\Sigma} \in \mathbb{R}^{m \times d}$：对角矩阵，对角线元素 $\sigma_1 \geq \sigma_2 \geq \ldots \geq 0$ 为**奇异值**
- $\mathbf{V} \in \mathbb{R}^{d \times d}$：**右奇异向量矩阵**，列为 $\mathbf{X}^\top\mathbf{X}$ 的特征向量，$\mathbf{V}^\top\mathbf{V} = \mathbf{I}$

**奇异值与特征值的关系**:
- $\sigma_i^2$ 是 $\mathbf{X}^\top\mathbf{X}$（或 $\mathbf{X}\mathbf{X}^\top$）的特征值

**SVD 的应用**:

| 应用 | 公式 |
|------|------|
| 矩阵求逆 | $\mathbf{A}^{-1} = \mathbf{V}\boldsymbol{\Sigma}^{-1}\mathbf{U}^\top$ |
| 行列式 | $\det(\mathbf{A}) = \prod_i \sigma_i$ |
| 谱范数 | $\|\mathbf{X}\|_2 = \sigma_{\max}$ |
| 最小二乘解 | $\mathbf{w} = \mathbf{V}\boldsymbol{\Sigma}^\dagger\mathbf{U}^\top\mathbf{y}$ |
| 低秩近似 (PCA) | 截断奇异值分解，保留最大 $k$ 个奇异值 |

> **直觉解释**: SVD 将矩阵的变换分解为三步：(1) $\mathbf{V}^\top$ 旋转，(2) $\boldsymbol{\Sigma}$ 在各轴方向缩放，(3) $\mathbf{U}$ 再次旋转。奇异值 $\sigma_i$ 衡量矩阵在对应方向上的"伸缩强度"。

---

## 10. 正定矩阵 (Positive Definite Matrices) [补充]

### 10.1 定义

一个实对称矩阵 $\mathbf{A} = \mathbf{A}^\top$ 称为**正定矩阵 (Positive Definite, PD)**，若对所有非零向量 $\mathbf{x}$：

> $$\mathbf{x}^\top \mathbf{A} \mathbf{x} > 0$$

若条件放宽为 $\geq 0$，则称为**半正定矩阵 (Positive Semi-definite, PSD)**。

### 10.2 等价条件

以下条件与正定性等价（对实对称矩阵）：
1. 所有特征值 $\lambda_i > 0$
2. 所有主子式（leading principal minors）行列式 $> 0$（Sylvester 准则）
3. 存在满秩矩阵 $\mathbf{L}$ 使得 $\mathbf{A} = \mathbf{L}\mathbf{L}^\top$（Cholesky 分解）

### 10.3 ML 中的重要性

- **协方差矩阵**（Covariance Matrix）必然是半正定矩阵
- **Hessian 矩阵**（目标函数的二阶导矩阵）为正定时，对应点为严格局部最小值
- **核矩阵 (Kernel Matrix)**（SVM, W5 中讲授）必须是半正定矩阵

---

## 11. 向量与矩阵微积分 (Vector/Matrix Calculus) [补充]

### 11.1 梯度 (Gradient)

设 $f: \mathbb{R}^n \to \mathbb{R}$ 是一个标量函数，则 $f$ 对向量 $\mathbf{x} = [x_1, \ldots, x_n]^\top$ 的**梯度**定义为：

> $$\nabla_\mathbf{x} f = \frac{\partial f}{\partial \mathbf{x}} = \begin{bmatrix} \frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2} \\ \vdots \\ \frac{\partial f}{\partial x_n} \end{bmatrix}$$

梯度是一个与 $\mathbf{x}$ 同维度的列向量，指向函数增长最快的方向。

### 11.2 常用求导规则

| 表达式 $f(\mathbf{x})$ | 梯度 $\nabla_\mathbf{x} f$ |
|------------------------|----------------------------|
| $\mathbf{a}^\top \mathbf{x}$ | $\mathbf{a}$ |
| $\mathbf{x}^\top \mathbf{a}$ | $\mathbf{a}$ |
| $\mathbf{x}^\top \mathbf{x}$ | $2\mathbf{x}$ |
| $\mathbf{x}^\top \mathbf{A} \mathbf{x}$（$\mathbf{A}$ 对称）| $2\mathbf{A}\mathbf{x}$ |
| $\|\mathbf{x}\|_2^2 = \mathbf{x}^\top\mathbf{x}$ | $2\mathbf{x}$ |
| $(\mathbf{A}\mathbf{x} - \mathbf{b})^\top(\mathbf{A}\mathbf{x} - \mathbf{b})$ | $2\mathbf{A}^\top(\mathbf{A}\mathbf{x} - \mathbf{b})$ |

### 11.3 线性回归的闭合解推导

线性回归的目标函数（均方误差）：

$$J(\mathbf{w}) = \frac{1}{2}\|\mathbf{X}\mathbf{w} - \mathbf{y}\|_2^2 = \frac{1}{2}(\mathbf{X}\mathbf{w} - \mathbf{y})^\top(\mathbf{X}\mathbf{w} - \mathbf{y})$$

对 $\mathbf{w}$ 求梯度并令其为零：

$$\nabla_\mathbf{w} J = \mathbf{X}^\top(\mathbf{X}\mathbf{w} - \mathbf{y}) = \mathbf{X}^\top\mathbf{X}\mathbf{w} - \mathbf{X}^\top\mathbf{y} = \mathbf{0}$$

> $$\hat{\mathbf{w}} = (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y} \quad \text{（Normal Equation，法方程）}$$

这正是情形 (ii) 中超定方程组的最小二乘解！

### 11.4 Hessian 矩阵

设 $f: \mathbb{R}^n \to \mathbb{R}$，其**Hessian 矩阵**为二阶偏导矩阵：

$$\mathbf{H} = \nabla^2_\mathbf{x} f = \begin{bmatrix} \frac{\partial^2 f}{\partial x_1^2} & \frac{\partial^2 f}{\partial x_1 \partial x_2} & \cdots \\ \frac{\partial^2 f}{\partial x_2 \partial x_1} & \frac{\partial^2 f}{\partial x_2^2} & \cdots \\ \vdots & & \ddots \end{bmatrix}$$

> - 若 $\mathbf{H}$ 正定，则驻点为局部最小值
> - 若 $\mathbf{H}$ 负定，则驻点为局部最大值
> - 若 $\mathbf{H}$ 不定，则驻点为鞍点

---

## 12. 线性代数与 ML 的关联

| 线性代数概念 | ML 中的应用 |
|-------------|------------|
| **矩阵-向量乘积** | 线性模型预测 $\hat{\mathbf{y}} = \mathbf{X}\mathbf{w}$、神经网络每层的前向传播 |
| **矩阵逆** | 线性回归 Normal Equation $(\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y}$ |
| **行列式** | 多元高斯分布的归一化常数 |
| **特征值/特征向量** | PCA 找主成分方向、协方差矩阵分析 |
| **SVD** | PCA 的数值实现、矩阵补全（Matrix Completion）、推荐系统 |
| **正定矩阵** | 协方差矩阵、核函数矩阵、优化凸性判断 |
| **范数** | L1/L2 正则化、梯度下降步长控制 |
| **线性无关** | 特征冗余检测（多重共线性）|
| **梯度（向量微积分）** | 梯度下降法优化损失函数 |

---

## 13. 补充资料 (Supplementary Resources)

### 教材章节对照

| 内容 | Murphy (MLPP) | Bishop (PRML) | Burkov (100-page ML) | Boyd & Vandenberghe |
|------|---------------|----------------|----------------------|---------------------|
| 向量与矩阵基础 | App. B | App. C | — | Ch. 1 |
| 范数 | App. B | App. C | — | Ch. 3 |
| 矩阵逆与行列式 | App. B | App. C | — | Ch. 11 |
| 特征值/特征向量 | App. B | App. C | — | Ch. 5 |
| SVD | App. B.5 | App. C | — | Ch. 7 |
| 矩阵微积分 | App. B | App. C | — | Ch. 8 |
| PCA（特征分解应用）| Ch. 12 | Ch. 12 | Ch. 9 | — |

**参考书目（本讲座指定）**:
- [Book1] Stephen Boyd and Lieven Vandenberghe, *Introduction to Applied Linear Algebra*, Cambridge University Press, 2018（在线免费：https://web.stanford.edu/~boyd/vmls/）
- [Book2] Andreas C. Müller and Sarah Guido, *Introduction to Machine Learning with Python*, O'Reilly, 2017

### 关键公式速查

> **矩阵-向量乘积（线性模型）**:
> $$\hat{\mathbf{y}} = \mathbf{X}\mathbf{w}$$

> **线性回归 Normal Equation（超定方程组的最小二乘解）**:
> $$\hat{\mathbf{w}} = (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y}$$

> **SVD 分解**:
> $$\mathbf{X} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top, \quad \mathbf{X}^{-1} = \mathbf{V}\boldsymbol{\Sigma}^{-1}\mathbf{U}^\top, \quad \det(\mathbf{X}) = \prod_i \sigma_i$$

> **向量范数**:
> $$\|\mathbf{x}\|_2 = \sqrt{\sum_i x_i^2}, \quad \|\mathbf{x}\|_1 = \sum_i |x_i|, \quad \|\mathbf{X}\|_F = \sqrt{\sum_{i,j} x_{ij}^2}$$

> **梯度（均方误差对 $\mathbf{w}$ 的导数）**:
> $$\nabla_\mathbf{w} \|\mathbf{X}\mathbf{w} - \mathbf{y}\|^2 = 2\mathbf{X}^\top(\mathbf{X}\mathbf{w} - \mathbf{y})$$

> **正定矩阵判别**:
> $$\mathbf{A} \succ 0 \iff \forall \mathbf{x} \neq \mathbf{0}: \mathbf{x}^\top\mathbf{A}\mathbf{x} > 0 \iff \text{所有特征值 } \lambda_i > 0$$

### 补充学习资源

- **Gilbert Strang 线性代数课程（MIT OCW）**: https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/
- **3Blue1Brown「线性代数的本质」系列（可视化理解）**: https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab
- **Boyd & Vandenberghe VMLS 在线教材**: https://web.stanford.edu/~boyd/vmls/
