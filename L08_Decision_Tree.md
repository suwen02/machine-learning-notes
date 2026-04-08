# Lecture 08: Tree-based Methods（决策树与集成方法）
**DDA3020 Machine Learning — CUHK-SZ**  
**讲师**: Baoyuan Wu | **日期**: March 11/16, 2026

---

## 目录 (Table of Contents)

1. [动机：参数模型的局限性](#1-动机参数模型的局限性)
2. [K近邻 (KNN) 简介](#2-k近邻-knn-简介)
3. [决策树基本概念](#3-决策树基本概念)
4. [分类树：属性选择准则](#4-分类树属性选择准则)
5. [分类树算法比较：ID3、C4.5、CART](#5-分类树算法比较id3c45cart)
6. [回归树 (Regression Tree)](#6-回归树-regression-tree)
7. [过拟合与剪枝 (Pruning)](#7-过拟合与剪枝-pruning)
8. [连续特征与缺失值处理](#8-连续特征与缺失值处理)
9. [决策树优缺点总结](#9-决策树优缺点总结)
10. [集成方法 (Ensemble Methods)](#10-集成方法-ensemble-methods)
11. [Bagging](#11-bagging)
12. [Random Forest（随机森林）](#12-random-forest随机森林)
13. [Boosting（提升方法）](#13-boosting提升方法)
14. [补充资料](#14-补充资料)

---

## 1. 动机：参数模型的局限性

### 1.1 参数模型回顾

我们已经学习了三种监督学习的参数模型：线性回归、逻辑回归、SVM。它们的共同点（参数模型）：

- **训练**：在整个输入空间上定义一个假设函数，从所有训练数据中学习固定数量的参数
- **测试**：对任意测试输入使用相同的模型和相同的参数集

**参数模型的局限性**：
1. 模型结构需要人工指定（可能与真实关系相差甚远）
2. 决策过程难以解释（"黑盒"）

### 1.2 非参数模型的优势

**非参数模型 (Nonparametric Models)**：不假设输入与输出之间的固定关系形式，让数据"自己说话"。

典型代表：K近邻（KNN）、决策树（DT）。

**决策树的特殊优势**：
- 层次化决策过程，**可解释性 (Interpretability)** 极强
- 非参数方法，无需强分布假设
- 支持混合特征类型（数值型、类别型均可）

---

## 2. K近邻 (KNN) 简介

KNN 是最简单的非参数模型，此处简要介绍作为对比。

| 特性 | 描述 |
|------|------|
| **训练** | 惰性学习（Lazy Learning），直接存储所有训练数据，无显式训练 |
| **预测** | 计算测试点到所有训练点的距离，找 $K$ 个最近邻，投票（分类）或取均值（回归） |
| **距离度量** | Euclidean: $\sqrt{\sum(x_i-x_j)^2}$；Manhattan: $\sum|x_i-x_j|$；Cosine: $\frac{\mathbf{x}_i\cdot\mathbf{x}_j}{\|\mathbf{x}_i\|\|\mathbf{x}_j\|}$ |

**K 值选择**：
- $K$ 小：高方差（过拟合），决策边界复杂
- $K$ 大：高偏差（欠拟合），决策边界过于平滑

---

## 3. 决策树基本概念

### 3.1 定义

> **决策树 (Decision Tree)**：一种层次化的监督学习模型，通过一系列递归分割在局部区域进行预测。每个内部节点基于单个输入属性进行测试，每条分支代表一个可能的测试结果，每个叶节点对应一个输出（类别或数值）。

### 3.2 分类与结构

**按输出类型**：
- **分类树 (Classification Tree)**：输出为类别标签
- **回归树 (Regression Tree)**：输出为连续数值

**按分叉数**：
- **二叉树 (Binary Tree)**：每个节点最多两个子节点（CART 使用）
- **多路树 (Multi-way Tree)**：每个节点可有多个子节点（ID3 使用）

### 3.3 基本术语

| 术语 | 定义 |
|------|------|
| **根节点 (Root Node)** | 树的顶部节点，包含所有训练样本 |
| **内部节点 (Internal Node)** | 含有属性测试条件的节点 |
| **叶节点 (Leaf Node)** | 树的末端，给出预测结果 |
| **深度 (Depth)** | 从根节点到最远叶节点的路径长度 |
| **大小 (Size)** | 树中节点的总数 |
| **父/子节点** | A 是 B、C 的父节点；B、C 是 A 的子节点 |

### 3.4 决策树学习算法（总流程）

决策树的构建是一个**贪心（Greedy）**、**自顶向下（Top-down）**、**递归分治（Divide-and-Conquer）**的算法：

```
1. 从根节点开始，包含所有训练样本
2. 选择最佳属性进行划分（基于不纯度减少）
3. 将样本分发到各子节点
4. 对每个子节点递归重复上述步骤
5. 停止条件：
   - 节点中所有样本属于同一类（纯节点）
   - 达到最大深度
   - 节点样本数低于阈值
   - 划分带来的信息增益低于阈值
```

**注**：找最小决策树是 NP-complete 问题（Quinlan 1986），实际使用基于启发式的贪心算法。

---

## 4. 分类树：属性选择准则

### 4.1 不纯度 (Impurity)

设节点 $S$ 包含 $|S|$ 个训练样本，其中属于第 $C_i$ 类的样本集合为 $S_i$，类概率估计为：

$$p_i = \frac{|S_i|}{|S|}$$

**纯节点 (Pure Node)**：存在某个 $i$ 使得 $p_i = 1$（所有样本属于同一类），无需继续划分。

### 4.2 熵 (Entropy) — ID3/C4.5 的基础

**信息熵**来自信息论，度量一个节点的不确定性：

> $$\text{Info}(S) = -\sum_{i=1}^K p_i \log_2 p_i$$

**性质**：
- 纯节点（所有样本同类）：$\text{Info}(S) = 0$（最小，信息量为 0）
- 最不纯节点（各类均匀分布）：$\text{Info}(S) = \log_2 K$（最大）
- 二分类时：$-p\log_2 p - (1-p)\log_2(1-p)$，$p=0.5$ 时取最大值 1

### 4.3 信息增益 (Information Gain) — ID3 算法

设用属性 $A \in \{a_1, a_2, \ldots, a_V\}$ 将 $S$ 划分为 $V$ 个子集 $D_1, D_2, \ldots, D_V$。

**划分后的条件熵**（加权平均子节点熵）：

$$\text{Info}(S|A) = \sum_{v=1}^V \frac{|D_v|}{|S|} \times \text{Info}(D_v)$$

**信息增益**（用属性 $A$ 划分带来的熵减少量）：

> $$\text{Gain}(A) = \text{Info}(S) - \text{Info}(S|A)$$

**选择准则**：选择**信息增益最大**的属性。

#### 数值例子

设二分类数据集有 6 正例、6 负例，三个属性：Color（红/蓝/绿）、Size（大/小）、Shape（圆/方）。

- $\text{Info}(S) = H(3/6, 3/6) = 1$ bit
- $\text{Gain(Color)} = 1 - \text{Info}(S|\text{Color}) = 0.54$ bits
- $\text{Gain(Shape)} = 1 - \text{Info}(S|\text{Shape}) = 0$ bits
- $\text{Gain(Size)} = 1 - \text{Info}(S|\text{Size}) = 0.46$ bits

因此选择 **Color** 作为根节点属性。

### 4.4 增益率 (Gain Ratio) — C4.5 算法

**ID3 的问题**：信息增益偏向选择取值多的属性（例如 ID 属性，每个样本唯一，增益最大但无用）。

**C4.5 的解决方案**：用分裂信息 (Split Information) 对增益进行归一化：

$$\text{SplitInfo}(A) = -\sum_{v=1}^V \frac{|D_v|}{|S|} \log_2 \frac{|D_v|}{|S|}$$

> $$\text{GainRatio}(A) = \frac{\text{Gain}(A)}{\text{SplitInfo}(A)}$$

分裂信息度量了属性 $A$ 把数据分散到多少个子集的程度——取值越多，$\text{SplitInfo}$ 越大，从而抑制了"多值偏好"。

### 4.5 Gini 指数 (Gini Index) — CART 算法

$$\text{Gini}(S) = 1 - \sum_{i=1}^K p_i^2$$

**直觉**：若随机从节点 $S$ 中取两个样本，它们类别不同的概率即为 Gini 指数。

**划分后的 Gini 指数**：

$$\text{Gini}(S|A) = \sum_{v=1}^V \frac{|D_v|}{|S|} \text{Gini}(D_v)$$

**选择准则**：选择使 $\Delta\text{Gini}(A) = \text{Gini}(S) - \text{Gini}(S|A)$ 最大的属性。

### 4.6 三种准则的比较

| 准则 | 算法 | 公式 | 特点 |
|------|------|------|------|
| **信息增益** | ID3 | $\text{Gain}(A) = \text{Info}(S) - \text{Info}(S|A)$ | 偏好多值属性 |
| **增益率** | C4.5 | $\text{GainRatio}(A) = \text{Gain}(A)/\text{SplitInfo}(A)$ | 消除多值偏好 |
| **Gini 减少** | CART | $\Delta\text{Gini}(A) = \text{Gini}(S) - \text{Gini}(S|A)$ | 计算简单，可解释 |

**Entropy vs. Gini 的实践比较**：
- Gini 计算更简单（无 $\log$ 运算）
- Gini 更易解释（随机分类错误率）
- Entropy 对类别不平衡数据更有效
- Entropy 对噪声更不敏感
- 实践中两者结果差异通常很小

---

## 5. 分类树算法比较：ID3、C4.5、CART

| 特性 | ID3 | C4.5 | CART |
|------|-----|------|------|
| **提出者/年份** | Ross Quinlan, 1986 | Quinlan, 1993 | Breiman et al., 1984 |
| **树结构** | 多路树 | 多路树 | 二叉树 |
| **分裂准则** | 信息增益 | 增益率 | Gini 指数（分类）/ MSE（回归） |
| **连续属性** | 不支持 | 支持（二分法） | 支持（二分法） |
| **缺失值** | 不支持 | 支持 | 支持 |
| **剪枝** | 不支持 | 支持 | 代价复杂度剪枝 |
| **回归** | 不支持 | 不支持 | 支持 |
| **多输出** | 不支持 | 不支持 | 支持 |

---

## 6. 回归树 (Regression Tree)

### 6.1 基本思想

回归树的构建方式与分类树几乎相同，但**不纯度度量改为均方误差（MSE）或残差平方和（SSE）**。

**叶节点 $c$ 的预测值**：

$$\bar{y}_c = \frac{1}{N_c}\sum_{i \in c} y_i$$

**叶节点 $c$ 的 MSE**：

$$e_c = \frac{\sum_{i \in c}(y_i - \bar{y}_c)^2}{N_c}$$

**总 SSE（偏好使用）**：

$$S = \sum_{c \in \text{leaves}} \sum_{i \in c}(y_i - \bar{y}_c)^2$$

### 6.2 回归树生长算法

```
1. 从包含所有样本的根节点开始，计算 ȳ 和 S
2. 对每个节点：
   a. 若所有样本的所有特征值相同，停止
   b. 否则，搜索所有变量的所有二分法，找使 S 减少最多的分割点
   c. 若最大 S 减少量小于阈值 δ，或任一子节点样本数小于 q，停止
   d. 否则执行该分割，创建两个子节点
3. 对每个新节点递归执行步骤 2
```

**寻找最优分割点**：设阈值 $w_1$ 将根节点分为左子节点 $c_L$、右子节点 $c_R$：

$$S_{w_1} = \sum_{i \in c_L}(y_i - \bar{y}_{c_L})^2 + \sum_{i \in c_R}(y_i - \bar{y}_{c_R})^2$$

$$w_1^* = \arg\max_{w_1}(S - S_{w_1})$$

---

## 7. 过拟合与剪枝 (Pruning)

### 7.1 过拟合问题

决策树有强烈的**过拟合**倾向——一棵完全生长的树可以完美拟合训练数据，但在测试数据上表现很差（预测曲线高度不平滑）。

**过拟合根本原因**：贪心算法每步只追求局部最优，且完全生长的树在训练集上参数极多（每条到叶节点的路径都是一条规则）。

### 7.2 预剪枝 (Pre-pruning)

在树生长过程中提前停止。通过设置停止条件：
- 最大深度
- 节点最小样本数
- 最小信息增益阈值

**优点**：计算效率高，防止过度复杂  
**缺点**：可能过早停止，错过有用的划分（局部无用但全局有益的划分）

### 7.3 后剪枝 (Post-pruning) / Reduced-Error Pruning

先生长完整树，再从叶节点向上回剪。

**算法流程**：

```
1. 将训练数据进一步划分为训练集和验证集
2. 基于训练集生长一棵完整的深树
3. 重复，直到进一步剪枝有害：
   a. 评估剪去每个可能节点对验证集的影响
   b. 贪心地删除使验证集准确率提升最大的节点
```

**核心操作**：将一棵子树替换为叶节点，叶节点的预测值为该子树区域内训练样本的多数类（分类）或均值（回归）。

**判断准则**：若子树在验证集上的期望错误率 > 该子树被替换为叶节点后的错误率，则剪枝。

**后剪枝的效果**：随着剪枝程度增加，训练误差上升，但测试误差先下降（剪枝消除过拟合）后上升（剪枝过度）。最优剪枝点在"红线"处。

### 7.4 代价复杂度剪枝 (Cost-Complexity Pruning)

CART 使用的后剪枝策略，通过最小化带复杂度惩罚的代价函数：

$$R_\alpha(T) = R(T) + \alpha |T|$$

其中 $R(T)$ 是树 $T$ 的训练误差，$|T|$ 是叶节点数，$\alpha \geq 0$ 是正则化参数（通过交叉验证选择）。

---

## 8. 连续特征与缺失值处理

### 8.1 连续特征处理

对于连续属性 $A$，通过**二分法 (Binary Split)**将其离散化：

1. 将该属性的所有取值排序
2. 考虑所有相邻值对的中点作为候选分割阈值
3. 选择信息增益（或 Gini 减少）最大的阈值 $t$，划分为 $A \leq t$ 和 $A > t$ 两组

**注意**：连续属性可以在不同节点以不同阈值多次使用。

**局限性**：对连续变量进行分桶会引入量化误差（quantization error），这是决策树的固有局限之一。

### 8.2 缺失值处理（C4.5 策略）

**训练阶段**：
- 计算信息增益时，忽略该属性缺失的样本（用非缺失样本计算），但用"有效样本比例"加权调整
- 缺失的样本按各分支的样本比例"分配"到各子节点（软分配）

**预测阶段**：
- 若测试样本在某属性上缺失，将其按该节点各分支的概率分配，最终加权投票

---

## 9. 决策树优缺点总结

### 优点

| 优点 | 说明 |
|------|------|
| **可解释性强** | 可以直接读取决策规则（if-then rules） |
| **数据探索** | 揭示特征的重要性和数据结构 |
| **预处理少** | 不需要特征归一化，可处理缺失值 |
| **混合数据类型** | 可同时处理数值型和类别型特征 |
| **非参数方法** | 不假设数据分布 |

### 缺点

| 缺点 | 说明 |
|------|------|
| **过拟合** | 是最主要的实践难题，需剪枝解决 |
| **不稳定性** | 训练数据微小变化可能导致完全不同的树 |
| **连续变量量化误差** | 对连续特征的处理会损失信息 |
| **局部最优** | 贪心算法无法保证全局最优 |
| **不平衡数据** | 偏向多数类，需要类权重调整 |

---

## 10. 集成方法 (Ensemble Methods)

### 10.1 单棵决策树的问题

- **剪枝后的浅树**：预测能力弱（欠拟合）
- **完整生长的深树**：过拟合训练集

### 10.2 集成思想

> **集成模型 (Ensemble Model)** 的核心思想：构建多个多样化的决策树，将它们的预测进行组合（分类：多数投票；回归：取平均）作为最终预测。

**哲学**："三个臭皮匠，胜过诸葛亮"——群体智慧往往优于个体智慧（Wisdom of the Crowd）。

**关键前提**：各个基学习器（base learners）之间需要有足够的**多样性 (Diversity)**，才能通过组合降低整体误差。

---

## 11. Bagging

### 11.1 Bootstrap Aggregating 原理

**Step 1 — Bootstrap 采样（数据层面随机性）**：对训练集进行**有放回采样 (Sampling with Replacement)**，得到 $T$ 个大小与原始训练集相同的不同子集。

- 每次采样，每个样本被选中的概率为 $\frac{1}{m}$
- 期望约 $1 - (1-\frac{1}{m})^m \approx 1 - e^{-1} \approx 63.2\%$ 的样本被选到
- 约 $36.8\%$ 的样本不被选到（可用作"袋外 OOB"验证）

**Step 2 — 在每个 Bootstrap 样本上训练完整深树**（不剪枝）。

**Step 3 — 聚合预测（Aggregation）**：
- 分类：多数投票 (Majority Voting)
- 回归：取平均 (Average)

### 11.2 为什么 Bagging 有效？

**偏差-方差分解视角**：
- 单棵深树：低偏差、高方差（过拟合）
- Bagging 平均：低偏差、低方差（减少了方差，偏差基本不变）

Bagging 树越多，预测误差越低（趋于平稳）。

### 11.3 Bagging 的局限性

由于不同 Bootstrap 样本间有大量重叠样本，Bagging 产生的树之间相关性较高，**多样性不够**。

---

## 12. Random Forest（随机森林）

### 12.1 Random Forest = Bagging + 特征随机性

Random Forest 在 Bagging 的基础上，引入**分裂属性随机化**：

> 每次寻找最优分裂时，不考虑所有 $N$ 个特征，而是**随机选择 $m$ 个特征的子集**，只在这 $m$ 个特征中选最优分裂。

**推荐的 $m$ 值**：
- 分类树：$m = \sqrt{N}$
- 回归树：$m = N/3$

### 12.2 Random Forest vs. Bagging vs. 单棵决策树

| 模型 | 数据层面随机性 | 特征层面随机性 | 预测误差 |
|------|-------------|-------------|---------|
| 单棵决策树 | 无 | 无 | 最高（过拟合） |
| Bagging | 有（Bootstrap） | 无 | 中等 |
| Random Forest | 有（Bootstrap） | 有（特征子集） | 最低 |

**为什么特征随机化有效**？Bagging 中的树由于样本高度重叠，在根节点往往选择相同（最重要的）特征，导致树之间相关性高。特征随机化迫使不同树使用不同特征，增加了多样性，从而在平均时更有效地降低方差。

### 12.3 集成模型为何能缓解过拟合

每棵单独的决策树过拟合于不同的数据集和特征子集，通过集成平均，这些"对不同局部的过拟合"相互抵消，最终不再过拟合原始固定数据集。

---

## 13. Boosting（提升方法）

> **注**：课程讲义中未直接覆盖 Boosting，以下内容为重要补充，基于教材（Murphy MLAPP Chapter 16、Burkov Hundred-Page Chapter 5）。

### 13.1 Boosting 的基本思想

Boosting 与 Bagging 的本质区别：

| | Bagging | Boosting |
|-|---------|---------|
| **训练方式** | 并行（独立训练多棵树） | 串行（每棵树依赖前一棵） |
| **目标** | 降低方差 | 降低偏差 |
| **样本权重** | 均等（Bootstrap 采样） | 加权（前一轮错误样本权重更高） |

**核心思想**：每一轮训练重点关注前一轮**分类错误**的样本，使得后续的弱学习器能弥补前面的错误，最终组合成强学习器。

### 13.2 AdaBoost 算法

**算法流程**：

1. 初始化样本权重：$w_i^{(1)} = \frac{1}{m}$，$\forall i$
2. 对 $t = 1, 2, \ldots, T$：
   a. 在权重 $\{w_i^{(t)}\}$ 下训练弱学习器 $h_t$
   b. 计算加权错误率：$\epsilon_t = \sum_{i: h_t(\mathbf{x}_i) \neq y_i} w_i^{(t)}$
   c. 计算学习器权重：$\alpha_t = \frac{1}{2}\ln\frac{1-\epsilon_t}{\epsilon_t}$
   d. 更新样本权重：
      - 分类正确：$w_i^{(t+1)} \propto w_i^{(t)} e^{-\alpha_t}$
      - 分类错误：$w_i^{(t+1)} \propto w_i^{(t)} e^{+\alpha_t}$
   e. 归一化权重
3. 最终预测：$H(\mathbf{x}) = \text{Sgn}\left(\sum_{t=1}^T \alpha_t h_t(\mathbf{x})\right)$

**关键公式**：

> $$\alpha_t = \frac{1}{2}\ln\frac{1 - \epsilon_t}{\epsilon_t}$$

- $\epsilon_t < 0.5$（比随机猜测好）：$\alpha_t > 0$
- $\epsilon_t \to 0$（接近完美）：$\alpha_t \to \infty$
- $\epsilon_t = 0.5$（随机猜测）：$\alpha_t = 0$，该学习器被忽略

### 13.3 Gradient Boosting

**思想**：将 Boosting 统一为**梯度下降**框架——每棵新树拟合当前模型残差（负梯度）。

**算法**（以平方损失为例）：

1. 初始化 $F_0(\mathbf{x}) = \bar{y}$
2. 对 $t = 1, 2, \ldots, T$：
   a. 计算伪残差（负梯度）：$r_i^{(t)} = y_i - F_{t-1}(\mathbf{x}_i)$
   b. 用 $(x_i, r_i^{(t)})$ 训练回归树 $h_t$
   c. 更新模型：$F_t(\mathbf{x}) = F_{t-1}(\mathbf{x}) + \eta h_t(\mathbf{x})$（$\eta$ 为学习率）
3. 最终模型：$F_T(\mathbf{x}) = \sum_{t=0}^T \eta h_t(\mathbf{x})$

**Gradient Boosting 变体**：
- **XGBoost**：使用二阶梯度、正则化、列采样，极为高效
- **LightGBM**：基于直方图的近似，处理大规模数据
- **CatBoost**：专门处理类别特征

### 13.4 AdaBoost vs. Gradient Boosting

| | AdaBoost | Gradient Boosting |
|-|---------|-------------------|
| **权重更新** | 样本权重 | 拟合残差/伪梯度 |
| **损失函数** | 指数损失（隐式） | 任意可微损失函数 |
| **鲁棒性** | 对噪声和异常点敏感 | 更鲁棒（取决于损失函数） |
| **解释性** | 直接 | 一般框架 |

---

## 14. 补充资料

### 教材参考

- **Murphy "MLAPP"**：Chapter 16 — Adaptive Basis Function Models（AdaBoost 和 Gradient Boosting 详细推导）；Chapter 16.2 — Regression Trees
- **Bishop "PRML"**：Chapter 14.3 — Boosting；Chapter 14.1 — Bagging（*Pattern Recognition and Machine Learning*）
- **Burkov "Hundred-Page ML Book"**：Chapter 5 — Basic Practice（Ensemble Methods 精简介绍）

### 经典算法参考

- **ID3**: R. Quinlan (1986), "Induction of Decision Trees," *Machine Learning*, 1(1), 81–106
- **C4.5**: R. Quinlan (1993), *C4.5: Programs for Machine Learning*, Morgan Kaufmann
- **CART**: L. Breiman et al. (1984), *Classification and Regression Trees*, Chapman & Hall
- **Random Forests**: L. Breiman (2001), "Random Forests," *Machine Learning*, 45(1), 5–32

### 在线资源

- scikit-learn 决策树文档: https://scikit-learn.org/stable/modules/tree.html
- Georgia Tech 决策树笔记: https://faculty.cc.gatech.edu/~bboots3/CS4641-Fall2018/Lecture2/02_DecisionTrees.pdf
- CS540 Wisconsin 决策树: https://pages.cs.wisc.edu/~dyer/cs540/notes/11_learning-decision-pdf

### 关键公式速查

| 公式 | 内容 |
|------|------|
| $\text{Info}(S) = -\sum_i p_i \log_2 p_i$ | 熵（ID3） |
| $\text{Gain}(A) = \text{Info}(S) - \text{Info}(S|A)$ | 信息增益 |
| $\text{GainRatio}(A) = \text{Gain}(A)/\text{SplitInfo}(A)$ | 增益率（C4.5） |
| $\text{Gini}(S) = 1 - \sum_i p_i^2$ | Gini 指数（CART） |
| $S = \sum_c \sum_{i\in c}(y_i - \bar{y}_c)^2$ | 回归树的 SSE |
| $\alpha_t = \frac{1}{2}\ln\frac{1-\epsilon_t}{\epsilon_t}$ | AdaBoost 学习器权重 |
