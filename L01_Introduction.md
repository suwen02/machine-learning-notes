# DDA3020 机器学习 — Lecture 1: Introduction（导论）

> **课程**: DDA3020 Machine Learning, CUHK-SZ  
> **讲师**: Baoyuan Wu  
> **日期**: 2026年1月5/7日

---

## 目录 (Table of Contents)

1. [课程概述](#1-课程概述-course-overview)
2. [机器学习的定义](#2-机器学习的定义-definition-of-machine-learning)
3. [ML 与 AI 的关系及学科交叉](#3-ml-与-ai-的关系及学科交叉)
4. [机器学习的应用](#4-机器学习的应用)
5. [机器学习的两大基本范式](#5-机器学习的两大基本范式)
6. [监督学习 (Supervised Learning)](#6-监督学习-supervised-learning)
7. [无监督学习 (Unsupervised Learning)](#7-无监督学习-unsupervised-learning)
8. [监督学习的基本概念](#8-监督学习的基本概念)
9. [ML 实践流程](#9-ml-实践流程)
10. [扩展阅读：其他学习范式](#10-扩展阅读其他学习范式)
11. [补充资料](#11-补充资料-supplementary-resources)

---

## 1. 课程概述 (Course Overview)

### 1.1 课程基本信息

- **课程编号**: DDA3020 Machine Learning
- **学校**: School of Artificial Intelligence, CUHK-SZ
- **讲师**: Baoyuan Wu (Session 1) / Kui Jia (Session 2)

### 1.2 课程大纲（按周）

| 周次 | 内容 | 实验/作业 |
|------|------|-----------|
| W1 | Introduction（导论）| — |
| W2 | Review of Probability & Linear Algebra | Python & sklearn |
| W3 | Linear Regression I | 线性回归编程 |
| W4 | Linear Regression II & Logistic Regression | H1 发布 |
| W5 | Support Vector Machines (SVM) | SVM 编程 |
| W6 | Decision Tree and Random Forest | H1 截止 |
| W7 | Neural Networks I (MLP & CNN) | PyTorch 编程 |
| W8 | Neural Networks II (RNN & Transformer) | H2 发布 |
| W9 | Overfitting, Bias-Variance Trade-off | Demo |
| W10 | Performance Evaluation | H2 截止 |
| W11 | Intro. to Unsupervised Learning, K-Means | K-means 编程 |
| W12 | Mixture Models, EM algorithm | H3 发布 |
| W13 | PCA | PCA 编程 |
| W14 | Review（复习）| — |

### 1.3 评分方式 (Grading Policy)

- **25%** 书面作业（Written homework assignments, 3次）
- **25%** 编程作业（Programming homework, Python/Scikit-learn, 3次）
- **50%** 期末考试（Final exam）

> **注意**: 迟交扣分策略：(0, 48] 小时 → 50分；超过48小时 → 0分。  
> 课程要求具备概率论、线性代数基础知识及 Python 编程能力。

### 1.4 推荐教材 (Learning Materials)

**必读 (Required)**:
- Andriy Burkov, *The Hundred-Page Machine Learning Book*, 2019
- K. Murphy, *Machine Learning: A Probabilistic Perspective*, MIT Press, 2012

**推荐 (Recommended)**:
- Andreas C. Müller & Sarah Guido, *Introduction to Machine Learning with Python*, O'Reilly, 2017
- C. Bishop, *Pattern Recognition and Machine Learning*, Springer, 2011
- Jeff Leek, *The Elements of Data Analytic Style*, Lean Publishing, 2015

---

## 2. 机器学习的定义 (Definition of Machine Learning)

### 2.1 两种经典定义

**Arthur Samuel（1959）的定义**：

> "The field of study that gives computers the ability to learn without being explicitly programmed."
> 
> （让计算机在没有被显式编程的情况下，具备学习能力的研究领域。）

这是最早对 ML 的概括性定义，强调"学习"而非"硬编码规则"。

---

**Tom Mitchell（1997）的定义（更精确）**：

> "A computer program is said to learn from **experience E** with respect to some class of **tasks T** and **performance measure P**, if its performance at tasks in T, as measured by P, improves with experience E."

中文：若一个计算机程序在任务集合 $T$ 中的某类任务上，通过经验 $E$ 的积累，其由性能度量 $P$ 衡量的表现不断提升，则称该程序从经验 $E$ 中学习。

### 2.2 Mitchell 定义的三要素

| 要素 | 符号 | 含义 | 垃圾邮件过滤举例 |
|------|------|------|----------------|
| Task（任务）| $T$ | 程序要执行的任务 | 将邮件分类为垃圾/非垃圾 |
| Experience（经验）| $E$ | 学习的数据来源 | 观察用户标注垃圾邮件的历史行为 |
| Performance Measure（性能度量）| $P$ | 评估表现的指标 | 正确分类的邮件比例 |

> **直觉解释**: ML 的本质是"从数据中自动归纳规律"，而不是"人工写出所有规则"。这使得 ML 在处理复杂、高维问题（如图像识别、语言理解）时远优于传统规则系统。

---

## 3. ML 与 AI 的关系及学科交叉

### 3.1 ML 是 AI 的子领域

**Artificial Intelligence (AI)** 是指由机器展示的智能，与人类和动物的自然智能相区别，涉及意识和情感。

AI 涵盖众多子领域：
- **Machine Learning (ML)** — 从数据中学习
- **Computer Vision (CV)** — 图像与视频理解
- **Natural Language Processing (NLP)** — 文本与语言处理
- **Speech Processing** — 语音识别与合成

> ML 是 AI 中最重要的分支之一，是现代 AI 取得突破的核心技术驱动力。

### 3.2 ML 的跨学科性质

ML 是高度**跨学科**（interdisciplinary）的领域，它与以下学科深度交叉：

```
          概率论与统计学
         (Probability & Statistics)
                  |
  计算机科学 ——— ML ——— 数学优化
  (Computer Science)  (Optimization)
                  |
             线性代数
          (Linear Algebra)
```

- **概率论与统计学**：建模不确定性、参数估计（MLE, MAP）、贝叶斯推断
- **线性代数**：数据表示（向量/矩阵）、模型参数计算、矩阵分解
- **数学优化**：训练模型 = 求解优化问题（梯度下降等）
- **计算机科学**：算法设计与实现、数据结构

> **直觉解释**: 要真正掌握 ML，需要打牢这几门底层学科的基础。本课程 L2（概率论）和 L3（线性代数）就是为此做铺垫。

---

## 4. 机器学习的应用

ML 已广泛应用于众多关键任务，例如：

| 应用领域 | 代表案例 |
|----------|----------|
| 博弈游戏 (Game) | AlphaGo（围棋）、AlphaStar（星际争霸）|
| 生物信息学 | AlphaFold2（蛋白质结构预测）|
| 医疗影像 | 医学图像诊断（CT、MRI 分析）|
| 生物信号处理 | EEG 信号处理 |
| 计算机视觉 | 人脸识别、目标检测 |
| 自然语言处理 | 大语言模型（LLM）、机器翻译 |

---

## 5. 机器学习的两大基本范式

给定数据 $x_1, x_2, x_3, \ldots$，机器学习有两种基本范式：

### 5.1 监督学习 (Supervised Learning)

- **输入**: 带标签的数据对 $\{(x_i, y_i)\}_{i=1}^N$，其中 $y_i$ 为人工标注的输出
- **目标**: 学习从输入 $x_i$ 到输出 $y_i$ 的映射函数
- **类比**: 像老师教学生——有标准答案 (labels)

### 5.2 无监督学习 (Unsupervised Learning)

- **输入**: 无标签数据 $\{x_i\}_{i=1}^N$
- **目标**: 发现数据的内在结构（聚类、降维等）
- **类比**: 自学——没有标准答案，自己从数据中归纳规律

### 5.3 强化学习 (Reinforcement Learning)（本课程不重点讲授）

- **原理**: 智能体在环境中执行动作 $a_i$，改变状态 $x_i$，获得奖励/惩罚 $r_i$，逐渐学会在不同状态下采取最优动作
- **类比**: 在奖惩机制中学习——通过试错积累经验

> **本课程重点**: 监督学习（Supervised Learning）和无监督学习（Unsupervised Learning）。

---

## 6. 监督学习 (Supervised Learning)

### 6.1 数据形式

在监督学习中，数据集为带标签的样本集合：

$$\{(x_i, y_i)\}_{i=1}^N$$

其中：
- $x_i = [x_i^{(1)}, x_i^{(2)}, \ldots, x_i^{(D)}]^\top \in \mathcal{X}$，称为**特征向量 (feature vector)**，$D$ 为特征维度
- $y_i \in \mathcal{Y}$，称为**标签 (label)**，可以是：
  - 有限类别集合 $\{1, 2, \ldots, C\}$ —— 分类任务
  - 实数值 —— 回归任务

### 6.2 监督学习的两类任务

#### 回归 (Regression)

- **输出**: 连续实数值（continuous output）
- **示例**: 根据钻石质量（克拉数）预测价格
  - Task $T$: 预测钻石价格
  - Experience $E$: 历史价格数据
  - Performance $P$: 预测值的精度

#### 分类 (Classification)

- **输出**: 离散类别（finite and discrete/categorical outputs）
- **示例**: 将输入数据分为两类（二分类）
  - Task $T$: 判断邮件是否为垃圾邮件
  - Experience $E$: 历史带标签邮件
  - Performance $P$: 分类准确率

#### 回归 vs. 分类的判断

| 问题特征 | 应选 |
|----------|------|
| 输出是连续数值（如价格、温度、销售量）| Regression |
| 输出是离散类别（如是/否、类别1/2/3）| Classification |

> **例题**: 预测未来3个月某商品销售数量 → **Regression**（输出是连续数值）；  
> 判断某账户是否被黑客入侵 → **Classification**（输出是二元类别：是/否）。

### 6.3 监督学习工作流程

1. **数据收集**: $\{(x_i, y_i)\}_{i=1}^N$
2. **训练 (Training)**: 在训练数据上学习模型参数
3. **推断/测试 (Inference/Test)**: 用训练好的模型预测未见数据 $x$ 的输出

---

## 7. 无监督学习 (Unsupervised Learning)

### 7.1 数据形式

无监督学习的数据集为**无标签**的样本集合：

$$\{x_i\}_{i=1}^N$$

目标：建立一个模型，将特征向量 $x$ 映射为另一个向量或某个值，以揭示数据的内在结构。

### 7.2 两类主要任务

#### 聚类 (Clustering)

- **目标**: 将无标签数据点划分为若干簇（clusters）
- **性能评估**:
  - 同一簇内的点相互靠近（intra-cluster distance 小）
  - 不同簇的点相互远离（inter-cluster distance 大）
  - 所有数据有合适的覆盖
- **核心问题**: 如何定义"远近"（取决于所选特征和距离度量，如 Euclidean 距离、局部距离等）
- **算法示例**: K-Means（W11讲授）

#### 降维 (Dimensionality Reduction)

- **目标**: 将高维数据映射到低维空间
- **目的**:
  - 数据简化：非线性 → 线性
  - 数据可视化：高维 → 2D/3D
  - 降噪：去除冗余维度（噪声）
  - 变量筛选：学习稀疏模型，剔除冗余特征
- **算法示例**: PCA（W13讲授）

> **直觉解释**: 聚类相当于"自动分类"，而降维相当于"提炼精华"——在保留最重要信息的同时去除冗余，提高后续模型的效率。

---

## 8. 监督学习的基本概念

### 8.1 Training Set 和 Testing Set

**训练集 (Training Set)**：

$$\mathcal{D}_{train} = \{(x_i, y_i)\}_{i=1}^n$$

- 有监督：$x_i \in \mathcal{X}$（特征），$y_i \in \mathcal{Y}$（标签）
- 无监督：$\mathcal{D}_{train} = \{x_i\}_{i=1}^n$

**测试集 (Testing Set)**：

$$\mathcal{D}_{test} = \{(x_i, y_i)\}_{i=1}^m$$

用于评估训练好的模型在**未见数据**上的泛化性能（generalization performance）。

**i.i.d. 假设（Independent and Identically Distributed）**:

> 标准 ML 假设所有样本均为独立同分布（i.i.d.）的随机变量的观测值，且训练集与测试集服从相同的分布。

这是 ML 理论推导的基础假设，违背 i.i.d.（如 distribution shift/分布偏移）会导致模型泛化能力下降。

---

### 8.2 Target Function 与 Hypothesis

**Target Function（目标函数）** $t: \mathcal{X} \to \mathcal{Y}$:
- 是训练/测试数据背后真实的输入输出映射
- **未知**，ML 的目标就是去逼近它

**Hypothesis（假设）** $h$:
- 是描述未知目标函数的候选函数，例如：

$$h(x) = 1 \times x_1 + 2 \times x_2 = [1, 2] \cdot \begin{bmatrix} x_1 \\ x_2 \end{bmatrix}$$

**Hypothesis Space（假设空间）** $\mathcal{H}$:
- 所有合法假设的集合，例如线性函数族：

$$\mathcal{H}_w(x) = w_1 \times x_1 + w_2 \times x_2 = \mathbf{w}^\top \mathbf{x}$$

> **直觉解释**: 假设空间就是"搜索范围"——你在哪类函数里找最优解。选择合适的假设空间（如线性、多项式、神经网络）是 ML 设计的关键决策之一。

---

### 8.3 Loss Function / Cost Function / Objective Function

**Cost Function（代价函数）**: 衡量假设 $h$ 在估计 $x$ 与 $y$ 关系时的好坏程度。

常见损失函数举例——**平方损失 (Square Loss)**：

$$\ell(h(x), y) = (h(x) - y)^2$$

**Objective Function（目标函数）**: 我们要优化（最小化/最大化）的函数。当最小化时，也叫 Cost Function、Loss Function 或 Error Function（本课程中这些术语**可互换**使用）。

---

### 8.4 训练与测试

**训练/学习 (Training/Learning)**:

在假设空间 $\mathcal{H}$ 中，通过在训练集上优化目标函数，找到最优假设：

$$h^* = \arg\min_{h \in \mathcal{H}} \frac{1}{n} \sum_{(x_i, y_i) \in \mathcal{D}_{train}} (h(x_i) - y_i)^2$$

**测试/评估 (Testing/Evaluation)**:

在测试集上评估 $h^*$ 的泛化性能：

$$\text{Test Error} = \frac{1}{m} \sum_{(x_i, y_i) \in \mathcal{D}_{test}} (h^*(x_i) - y_i)^2$$

> **关键原则**: 训练误差低不等于测试误差低。模型可能**过拟合（Overfitting）**训练数据，导致在测试集上表现差。反之，如果模型过于简单，则会**欠拟合（Underfitting）**。

---

### 8.5 Overfitting、Underfitting 与 Cross-Validation

#### Overfitting（过拟合）

- 模型在训练集上表现很好，但在测试集上表现差
- 原因：模型过于复杂，"记住"了训练数据中的噪声
- 解决：正则化（Regularization）、增加数据量、简化模型、Cross-Validation

#### Underfitting（欠拟合）

- 模型在训练集和测试集上均表现差
- 原因：模型过于简单，无法捕捉数据的真实规律
- 解决：使用更复杂的模型、增加特征

#### Cross-Validation（交叉验证）

- 将训练数据分为若干份（folds），轮流将其中一份作为验证集，其余作为训练集
- 目的：在无需使用测试集的情况下评估模型性能，辅助超参数调优

> **K-折交叉验证 (K-Fold Cross-Validation)**：将数据分为 $K$ 份，进行 $K$ 次训练/验证，取平均性能作为估计。

---

### 8.6 No Free Lunch (NFL) Theorem

> **No Free Lunch 定理（无免费午餐定理）**: 不存在一种学习算法在所有任务上都比其他算法更好。对于某类任务表现好的算法，必然在另一类任务上表现差。

**含义**: 没有"万能"的 ML 算法。选择算法必须结合具体问题的先验知识和数据特征。这正是为什么 ML 中有如此多不同的算法（线性回归、SVM、神经网络……）。

---

## 9. ML 实践流程

### 9.1 一般机器学习工作流程（6步）

```
1. 数据收集 (Data Collection)
         ↓
2. 数据预处理 (Data Preprocessing)
         ↓
3. 确定假设空间、目标函数、优化方法
   (Determine Hypothesis Space, Objective Function, Optimization Method)
         ↓
4. 训练 (Training)
         ↓
5. 测试 (Testing)
         ↓
6. 性能提升 (Performance Improvement)
         ↑___________________________|
```

**各步骤详解**:

1. **数据收集**: 从 Excel、数据库、文本文件等来源收集历史数据。数据的**多样性、密度和数量**越好，学习效果越佳。

2. **数据预处理**: 评估数据质量，处理**缺失值（missing data）**和**异常值（outliers）**。

3. **确定三要素**:
   - **假设空间** $\mathcal{H}$（选什么模型，如线性模型、深度神经网络）
   - **目标函数**（选什么损失函数，如 MSE、交叉熵）
   - **优化方法**（如梯度下降、Adam）

4. **训练**: 通过优化目标函数，在训练数据上学习假设函数的参数

5. **测试**: 在测试数据上评估学习得到的模型

6. **性能提升**: 可能涉及更换模型、添加特征、调整超参数等。**数据收集和预处理**往往需要花费大量时间，是性能提升的关键。

### 9.2 实践工具

| 工具 | 用途 | 链接 |
|------|------|------|
| **Scikit-Learn** | 经典 ML 算法库（Python）| https://scikit-learn.org/ |
| **UCI Data Repository** | 标准数据集合集 | https://archive.ics.uci.edu/ |
| **Kaggle** | ML 竞赛平台，含海量数据集 | https://www.kaggle.com/ |

---

## 10. 扩展阅读：其他学习范式

除本课程重点讲授的监督学习和无监督学习外，ML 还有多种其他范式：

| 学习范式 | 特点 | 应用 |
|----------|------|------|
| **Reinforcement Learning（强化学习）** | 智能体通过奖惩机制与环境交互学习最优策略 | 游戏 AI、机器人控制 |
| **Semi-supervised Learning（半监督学习）** | 标注数据 + 未标注数据结合使用 | 标注成本高的场景 |
| **Ensemble Learning（集成学习）** | 多个 ML 模型组合，优于单一模型（三个臭皮匠，顶个诸葛亮）| 随机森林、Boosting |
| **Transfer Learning（迁移学习）** | 在源域数据上学习，应用于目标域（尤其目标数据不足时）| 预训练模型微调 |
| **Federated Learning（联邦学习）** | 在本地服务器学习，上传参数到中心服务器，保护用户隐私 | 手机端 ML |
| **Machine Unlearning（机器遗忘）** | 从已训练模型中抹去特定训练样本的影响，保护隐私 | 数据删除请求 |

---

## 11. 补充资料 (Supplementary Resources)

### 教材章节对照

| 内容 | Murphy (MLPP) | Bishop (PRML) | Burkov (100-page ML) |
|------|---------------|----------------|----------------------|
| ML 定义与基础概念 | Ch. 1 | Ch. 1 | Ch. 1, 2 |
| 监督学习（回归/分类）| Ch. 1, 7, 8 | Ch. 1, 3, 4 | Ch. 2, 3 |
| 无监督学习（聚类/降维）| Ch. 11, 15 | Ch. 9, 12 | Ch. 9 |
| Overfitting / 正则化 | Ch. 7 | Ch. 1, 3 | Ch. 4 |
| No Free Lunch 定理 | Ch. 1 | — | Ch. 1 |

### 补充学习资源

- **Supervised vs. Unsupervised Learning**: [Towards Data Science 文章](https://towardsdatascience.com/supervised-vs-unsupervised-learning-14f68e32ea8d)
- **Scikit-Learn 官方教程**: https://scikit-learn.org/stable/tutorial/
- **Kaggle ML 入门教程**: https://www.kaggle.com/learn/intro-to-machine-learning

### 关键公式速查

> **训练目标（最小化均方误差）**:
> $$h^* = \arg\min_{h \in \mathcal{H}} \frac{1}{n} \sum_{i=1}^{n} (h(x_i) - y_i)^2$$

> **Tom Mitchell ML 定义三要素**:
> - Task $T$: 要解决的问题
> - Experience $E$: 训练数据
> - Performance Measure $P$: 评估指标

> **i.i.d. 假设**:
> - 所有样本独立同分布
> - 训练集与测试集同分布
