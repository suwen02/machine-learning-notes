# L13: Performance Evaluation (性能评估)

**课程**: DDA3020 Machine Learning, CUHK-SZ  
**主讲**: Baoyuan Wu  
**日期**: April 8/13, 2026

---

## 目录 (Table of Contents)

1. [动机：为什么需要性能评估](#1-动机为什么需要性能评估)
2. [交叉验证 (Cross-Validation)](#2-交叉验证-cross-validation)
   - 2.1 [超参数 vs 参数](#21-超参数-vs-参数)
   - 2.2 [超参数调优的四种思路](#22-超参数调优的四种思路)
   - 2.3 [K-Fold Cross-Validation](#23-k-fold-cross-validation)
   - 2.4 [K 值选择与实践考量](#24-k-值选择与实践考量)
   - 2.5 [补充：其他交叉验证变体](#25-补充其他交叉验证变体)
3. [回归评估指标 (Evaluation Metrics for Regression)](#3-回归评估指标-evaluation-metrics-for-regression)
   - 3.1 [MSE 与 MAE](#31-mse-与-mae)
   - 3.2 [补充：RMSE、R² 等其他指标](#32-补充rmser²-等其他指标)
4. [分类评估指标 (Evaluation Metrics for Classification)](#4-分类评估指标-evaluation-metrics-for-classification)
   - 4.1 [混淆矩阵 (Confusion Matrix)](#41-混淆矩阵-confusion-matrix)
   - 4.2 [Accuracy、Precision、Recall](#42-accuracyprecisionrecall)
   - 4.3 [补充：F1 Score 与调和平均](#43-补充f1-score-与调和平均)
   - 4.4 [Cost-Sensitive Accuracy](#44-cost-sensitive-accuracy)
   - 4.5 [TPR、FPR、TNR、FNR 及其关系](#45-tprfprtnrfnr-及其关系)
   - 4.6 [决策阈值对分类结果的影响](#46-决策阈值对分类结果的影响)
   - 4.7 [Equal Error Rate (EER)](#47-equal-error-rate-eer)
5. [ROC 曲线与 AUC](#5-roc-曲线与-auc)
   - 5.1 [DET 曲线](#51-det-曲线)
   - 5.2 [ROC 曲线](#52-roc-曲线)
   - 5.3 [AUC 的定义与数学公式](#53-auc-的定义与数学公式)
   - 5.4 [AUC 计算示例](#54-auc-计算示例)
   - 5.5 [AUC 的四种典型情形](#55-auc-的四种典型情形)
   - 5.6 [AUC 的性质](#56-auc-的性质)
6. [多分类的混淆矩阵](#6-多分类的混淆矩阵)
7. [计算性能与可维护性 (Optional)](#7-计算性能与可维护性-optional)
8. [总结与公式速查表](#8-总结与公式速查表)
9. [补充资料 (Supplementary Resources)](#9-补充资料-supplementary-resources)

---

## 1. 动机：为什么需要性能评估

回顾 Tom Mitchell 对机器学习的定义：

> "A computer program is said to learn from **experience E** with respect to some class of **tasks T** and **performance measure P**, if its performance at tasks in T, as measured by P, improves with experience E."

- **Experience E** → 训练数据
- **Task T** → 分类或回归任务
- **Performance measure P** → 本讲的核心主题

### 机器学习的工作流回顾

此前课程中，学习算法由三个核心组件构成：

1. **损失函数** (Loss function)：如 MSE $\frac{1}{n}\sum_{i=1}^{n}(f(x_i; w, b) - y_i)^2$
2. **目标函数/优化准则** (Objective function)：如最小化上述 MSE
3. **优化算法** (Optimization routine)：如梯度下降

但这只是工作流的一部分。训练完成后，我们需要回答一个关键问题：

> **模型在新数据上的表现如何？** 即如何基于有限数据评估算法的泛化能力。

这就是性能评估 (Performance Evaluation) 的动机。

---

## 2. 交叉验证 (Cross-Validation)

### 2.1 超参数 vs 参数

| 类别 | 定义 | 举例 |
|------|------|------|
| **Hyper-parameters（超参数）** | 在学习算法**外部**确定，不由训练数据直接学习 | 模型选择（LR vs SVM）、多项式阶数、决策树最大深度、Random Forest 树的数量、SVM 中的 $C$、学习率 $\eta$ |
| **Parameters（参数）** | 由学习算法基于训练集**学习**得到 | 线性模型中的 $w, b$（$y = w^\top x + b$） |

> **核心问题**：如何调优超参数 (Hyper-parameter tuning)？

### 2.2 超参数调优的四种思路

**Idea 1：在全部数据上选最优超参数**

```
| Your Dataset (全部数据) |
```

- 选择在全部数据上误差最低的超参数
- **问题**：模型对训练数据过拟合，在新数据上可能表现很差
- **结论**：❌ 不可行

**Idea 2：Train / Test 二分**

```
| train                    | test |
```

- 在训练集上训练，在测试集上选超参数
- **问题**：测试集参与了超参数选择，无法反映在**未见数据**上的真实性能
- **结论**：❌ 不可行

**Idea 3：Train / Validation / Test 三分**

```
| train          | validation | test |
```

- 在训练集训练，在验证集选超参数，在测试集评估最终性能
- **改进**：测试集未参与训练过程
- **问题**：性能受 train/validation 随机划分的影响，结果不稳定
- **结论**：⚠️ 有改进但不够鲁棒

**Idea 4：K-Fold Cross-Validation** ✅

- 将训练数据分为 $K$ 折
- 每次用 1 折作验证集、其余 $K-1$ 折作训练集
- 在所有折上取平均性能，选择最优超参数
- **结论**：✅ 广泛使用的标准方法

### 2.3 K-Fold Cross-Validation

**算法流程：**

1. 将训练数据随机分为 $K$ 个大小相近的子集（folds）：$D_1, D_2, \ldots, D_K$
2. 对每个 $k = 1, 2, \ldots, K$：
   - 将 $D_k$ 作为验证集
   - 将 $D \setminus D_k$（其余 $K-1$ 个fold）作为训练集
   - 训练模型，在 $D_k$ 上计算评估指标 $e_k$
3. 最终评估指标为所有折的平均值：

> $$\text{CV Score} = \frac{1}{K}\sum_{k=1}^{K} e_k$$

4. 对每组超参数重复上述过程，选择 CV Score 最优的超参数组合

**示意图（以 K=5 为例）：**

| 轮次 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Test |
|------|--------|--------|--------|--------|--------|------|
| 1 | **Val** | Train | Train | Train | Train | 不参与 |
| 2 | Train | **Val** | Train | Train | Train | 不参与 |
| 3 | Train | Train | **Val** | Train | Train | 不参与 |
| 4 | Train | Train | Train | **Val** | Train | 不参与 |
| 5 | Train | Train | Train | Train | **Val** | 不参与 |

**应用示例：** 使用 K-fold CV 确定 Lasso/Ridge 回归中正则化参数 $\alpha$ 的最优值。随 $\alpha$ 变化绘制 CV score 曲线，选择使 CV score 最高的 $\alpha$。

### 2.4 K 值选择与实践考量

**K 的影响：**

| K 的大小 | 优势 | 劣势 |
|---------|------|------|
| K 过大 | 训练集更大，bias 更低 | 验证集太小导致 variance 大；各轮训练集高度重叠导致过拟合；计算成本高 |
| K 过小 | 计算成本低 | 训练数据不足，可能 underfitting |
| **K = 5~10（推荐）** | 良好的 bias-variance 平衡 | 标准实践选择 |

**特殊情况：Leave-One-Out CV (LOOCV)**

- 当 $K = n$（样本总数）时，每次只留一个样本做验证
- 优点：bias 极低（训练集几乎是全部数据）
- 缺点：计算成本 $O(n)$ 次训练；variance 高（各轮训练集高度相关）

**深度学习中的局限：**
- 深度学习通常需要大规模数据集，K-fold CV 的计算成本过高
- 实践中常用**单次 train/val/test 划分**（如 80/10/10）

### 2.5 补充：其他交叉验证变体

| 方法 | 描述 | 适用场景 |
|------|------|---------|
| **Stratified K-Fold** | 保证每折中各类别比例与原数据一致 | 类别不平衡数据集 |
| **Repeated K-Fold** | 重复多次 K-Fold（每次不同随机划分），取平均 | 需要更稳定的估计 |
| **Group K-Fold** | 保证同一"组"的样本不会同时出现在训练集和验证集中 | 有组结构的数据（如同一患者的多次测量） |
| **Time Series Split** | 保证训练集始终在验证集之前（时间顺序） | 时间序列数据 |

> **教材参考**：详见 "Introduction to Machine Learning with Python" (Müller & Guido) 第5章。

---

## 3. 回归评估指标 (Evaluation Metrics for Regression)

### 3.1 MSE 与 MAE

设 $y_i$ 为第 $i$ 个样本的真实值，$\hat{y}_i$ 为预测值，共 $n$ 个样本。

**Mean Squared Error (MSE)**

> $$\text{MSE} = \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{n}$$

- 对**大误差惩罚更重**（平方放大离群值的影响）
- 可微，适合作为优化目标
- 单位为原始值的平方

**Mean Absolute Error (MAE)**

> $$\text{MAE} = \frac{\sum_{i=1}^{n}|y_i - \hat{y}_i|}{n}$$

- 对所有误差**等权**惩罚
- 对离群值更鲁棒
- 单位与原始值相同
- 在零点不可微

### 3.2 补充：RMSE、R² 等其他指标

**Root Mean Squared Error (RMSE)**

> $$\text{RMSE} = \sqrt{\text{MSE}} = \sqrt{\frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{n}}$$

- 单位与原始值相同，更易解释

**R² (Coefficient of Determination)**

> $$R^2 = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2} = 1 - \frac{SS_{res}}{SS_{tot}}$$

- $R^2 = 1$：完美预测
- $R^2 = 0$：模型等同于预测均值 $\bar{y}$
- $R^2 < 0$：模型比预测均值更差

**Adjusted R²**

> $$R^2_{adj} = 1 - \frac{(1-R^2)(n-1)}{n-p-1}$$

其中 $p$ 为特征数。修正了 R² 随特征数增加而虚增的问题。

**指标对比：**

| 指标 | 对离群值敏感 | 可微 | 单位 | 常见用途 |
|------|------------|------|------|---------|
| MSE | 是（平方放大） | 是 | 原始单位² | 优化目标 |
| RMSE | 是 | 是 | 原始单位 | 报告误差 |
| MAE | 否（鲁棒） | 否（零点） | 原始单位 | 鲁棒评估 |
| R² | 是 | 是 | 无量纲 | 解释性 |

---

## 4. 分类评估指标 (Evaluation Metrics for Classification)

### 4.1 混淆矩阵 (Confusion Matrix)

对于**二分类**问题，设 class-1 为 **Positive class**（正类），class-2 为 **Negative class**（负类）。

混淆矩阵包含四个条目：

|  | **Predicted Positive ($\hat{P}$)** | **Predicted Negative ($\hat{N}$)** |
|--|-------|-------|
| **Actual Positive (P)** | **TP** (True Positive) | **FN** (False Negative) |
| **Actual Negative (N)** | **FP** (False Positive) | **TN** (True Negative) |

- **TP**：正样本被正确预测为正类
- **TN**：负样本被正确预测为负类
- **FN**：正样本被**错误**预测为负类 → **Type II Error（漏报）**
- **FP**：负样本被**错误**预测为正类 → **Type I Error（误报）**

> **直觉记忆**：首字母 T/F 表示预测是否正确，第二个字母 P/N 表示模型的预测结果。

**数值示例（来自课堂）：**

| | Predicted Class-1 | Predicted Class-2 |
|--|---|---|
| Actual Class-1 | TP = 7 | FN = 7 |
| Actual Class-2 | FP = 2 | TN = 25 |

- 14个正样本中有7个分类正确
- 27个负样本中有25个分类正确
- 两种错误：FP（负类被误判为正类）和 FN（正类被误判为负类）

### 4.2 Accuracy、Precision、Recall

**Accuracy（准确率）**

> $$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

- 所有正确分类的样本占总样本的比例
- **局限**：在类别不平衡时可能误导（如 99% 负样本的数据集中，全部预测为负类也有 99% accuracy）

**Precision（精确率/查准率）**

> $$\text{Precision} = \frac{TP}{TP + FP}$$

- 在所有**被预测为正类**的样本中，真正为正类的比例
- 回答："预测为正的结果中，有多少是对的？"
- 高 Precision → 低误报率

**Recall（召回率/查全率）**

> $$\text{Recall} = \frac{TP}{TP + FN}$$

- 在所有**真正为正类**的样本中，被正确预测为正类的比例
- 回答："所有真正的正样本中，有多少被找出来了？"
- 高 Recall → 低漏报率

**Precision vs Recall 的权衡：**

| 场景 | 侧重 | 原因 |
|------|------|------|
| 垃圾邮件过滤 | Precision 优先 | 不希望误将正常邮件标为垃圾（低 FP） |
| 癌症筛查 | Recall 优先 | 不希望漏诊恶性肿瘤（低 FN） |
| 信息检索 | 两者均衡 | 既要结果相关，又要覆盖全面 |

### 4.3 补充：F1 Score 与调和平均

**F1 Score** 是 Precision 和 Recall 的调和平均数，用于在两者之间取得平衡：

> $$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2 \cdot TP}{2 \cdot TP + FP + FN}$$

- $F_1 = 1$：完美的 Precision 和 Recall
- $F_1 = 0$：Precision 或 Recall 为零
- 使用调和平均（而非算术平均）：只要 P 或 R 有一个很低，$F_1$ 就会被拉低

**$F_\beta$ Score（广义版本）：**

> $$F_\beta = (1 + \beta^2) \cdot \frac{\text{Precision} \cdot \text{Recall}}{(\beta^2 \cdot \text{Precision}) + \text{Recall}}$$

- $\beta > 1$：更侧重 Recall（如 $F_2$）
- $\beta < 1$：更侧重 Precision（如 $F_{0.5}$）

### 4.4 Cost-Sensitive Accuracy

在实际应用中，不同类型的错误代价可能不同。例如：

> 在医疗诊断中，将恶性肿瘤预测为良性（FN）的代价远高于将良性预测为恶性（FP）。

**代价矩阵 (Cost Matrix)：**

| | Predicted Positive | Predicted Negative |
|--|---|---|
| Actual Positive | $C_{p,p}$（正确分类收益） | $C_{p,n}$（漏报代价） |
| Actual Negative | $C_{n,p}$（误报代价） | $C_{n,n}$（正确分类收益） |

**Cost-Sensitive Accuracy：**

> $$\text{Cost-Sensitive Accuracy} = \frac{C_{p,p} \cdot TP + C_{n,n} \cdot TN}{C_{p,p} \cdot TP + C_{n,n} \cdot TN + C_{p,n} \cdot FN + C_{n,p} \cdot FP}$$

### 4.5 TPR、FPR、TNR、FNR 及其关系

将 TP、TN、FP、FN 归一化到 [0, 1]：

> $$\text{TPR (True Positive Rate)} = \frac{TP}{TP + FN} = \text{Recall} = \text{Sensitivity}$$

> $$\text{FNR (False Negative Rate)} = \frac{FN}{TP + FN} = 1 - \text{TPR}$$

> $$\text{TNR (True Negative Rate)} = \frac{TN}{FP + TN} = \text{Specificity}$$

> $$\text{FPR (False Positive Rate)} = \frac{FP}{FP + TN} = 1 - \text{TNR}$$

**约束关系：**

- $\text{TPR} + \text{FNR} = 1$（覆盖所有正类样本）
- $\text{TNR} + \text{FPR} = 1$（覆盖所有负类样本）

**类别平衡时的 Accuracy：**

> $$\text{Accuracy} = \frac{\text{TPR} + \text{TNR}}{2} = 1 - \frac{\text{FPR} + \text{FNR}}{2}$$

**各指标的别名对照：**

| 指标 | 别名 | 含义 |
|------|------|------|
| TPR | Recall, Sensitivity, Hit Rate | 正类被正确识别的比例 |
| FPR | Fall-out, 1 - Specificity | 负类被错误识别为正类的比例 |
| TNR | Specificity, Selectivity | 负类被正确识别的比例 |
| FNR | Miss Rate, 1 - Recall | 正类被错误识别为负类的比例 |

### 4.6 决策阈值对分类结果的影响

对于输出连续预测值的分类器（如 Logistic Regression），需要设置一个**决策阈值** $\tau$，将连续输出映射为离散类别：

- 预测值 $\hat{y} \geq \tau$ → Positive
- 预测值 $\hat{y} < \tau$ → Negative

**阈值变化的影响：**

| 阈值方向 | TP | FP | FN | TN | Precision | Recall |
|---------|----|----|----|----|-----------|--------|
| $\tau$ 上移 | ↓ | ↓ | ↑ | ↑ | ↑ | ↓ |
| $\tau$ 下移 | ↑ | ↑ | ↓ | ↓ | ↓ | ↑ |

**课堂示例（6个数据点）：**

| 样本 | N1 | N2 | P1 | N3 | P2 | P3 |
|------|----|----|----|----|----|----|
| 预测值 $y$ | -1.1 | -0.5 | -0.1 | 0.2 | 0.6 | 0.9 |

- 阈值 $\tau = 0$（默认）：TP=2, FN=1, FP=1, TN=2，共 2 个错误
- 阈值上移：不同的 TP/FP/FN/TN 组合
- 阈值下移：又一组不同的分类结果

**分布视角：**
- 将预测输出归一化后，正类和负类各自呈分布曲线
- 改变阈值 $\tau$ → 改变两个分布的划分位置 → 不同的 FPR 和 FNR

每个阈值设定称为一个 **operating point**（工作点），对应一组唯一的 (FPR, FNR) 值或 (FPR, TPR) 值。

### 4.7 Equal Error Rate (EER)

将阈值 $\tau$ 从 0 变化到 1：

- FPR 呈**递减**趋势
- FNR 呈**递增**趋势

两条曲线的**交点**即 $\text{FPR} = \text{FNR}$ 的位置，称为 **Equal Error Rate (EER)**。

> **EER 越低，分类器性能越好。**

EER 常用于生物识别（如人脸识别、指纹识别）系统的评估。

---

## 5. ROC 曲线与 AUC

### 5.1 DET 曲线

**DET 曲线 (Detection Error Trade-off Curve)**

- **x 轴**：FPR
- **y 轴**：FNR
- 曲线越**靠近左下角** → 分类性能越好
- 用于展示两种错误率之间的权衡

### 5.2 ROC 曲线

**ROC 曲线 (Receiver Operating Characteristic Curve)**

- **x 轴**：FPR（即 $1 - \text{Specificity}$）
- **y 轴**：TPR（即 Recall / Sensitivity）
- 通过改变决策阈值 $\tau$ 从 0 到 1，得到一系列 (FPR, TPR) 点，连成曲线

**ROC 曲线的关键特征：**

| 特征 | 含义 |
|------|------|
| 曲线越靠近**左上角** $(0, 1)$ | 分类性能越好（FPR=0, TPR=1 为完美分类） |
| 对角线 $y = x$ | 随机猜测基准线 |
| 曲线在对角线**上方** | 分类器优于随机猜测 |
| 曲线在对角线**下方** | 分类器劣于随机猜测（翻转预测即可改善） |

> **直觉**：ROC 曲线综合展示了分类器在**所有可能阈值**下的性能表现。

### 5.3 AUC 的定义与数学公式

**AUC (Area Under the ROC Curve)** 是 ROC 曲线下的面积，提供了一个**单一数值**来衡量分类器在所有阈值下的整体性能。

**概率解释：**

> AUC 可以理解为：**随机抽取一个正样本和一个负样本，模型对正样本的预测分数高于负样本的概率。**

**数学公式：**

设有 $m^+$ 个正样本（$i = 1, \ldots, m^+$）和 $m^-$ 个负样本（$j = 1, \ldots, m^-$）。

令 $g(\mathbf{x})$ 为预测函数，定义：

$$e_{ij} = g(\mathbf{x}_i^+) - g(\mathbf{x}_j^-)$$

定义 Heaviside 阶跃函数：

$$u(e) = \begin{cases} 1 & \text{if } e > 0 \\ 0.5 & \text{if } e = 0 \\ 0 & \text{if } e < 0 \end{cases}$$

> $$\text{AUC} = \frac{1}{m^+ \cdot m^-}\sum_{i=1}^{m^+}\sum_{j=1}^{m^-} u(e_{ij})$$

**AUC 的取值范围与含义：**

| AUC 值 | 含义 |
|--------|------|
| **AUC = 1** | 完美分类——所有正样本的预测分数都高于所有负样本 |
| **AUC = 0.5** | 等同于随机猜测——分类器无区分能力 |
| **AUC = 0** | 完全反向分类——翻转预测即为完美分类 |
| **0.5 < AUC < 1** | 分类器优于随机猜测，越接近 1 越好 |

### 5.4 AUC 计算示例

**题目：** 给定 4 个样本，2 个正样本和 2 个负样本：
$$\{(\mathbf{x}_1^-, -), (\mathbf{x}_2^-, -), (\mathbf{x}_1^+, +), (\mathbf{x}_2^+, +)\}$$

预测分数分别为 $g = 0.1, 0.4, 0.35, 0.8$。

**计算过程：**

$m^+ = 2, \quad m^- = 2$，需计算 $2 \times 2 = 4$ 个 $e_{ij}$：

$$e_{11} = g(\mathbf{x}_1^+) - g(\mathbf{x}_1^-) = 0.35 - 0.1 = 0.25 > 0 \implies u(e_{11}) = 1$$

$$e_{12} = g(\mathbf{x}_1^+) - g(\mathbf{x}_2^-) = 0.35 - 0.4 = -0.05 < 0 \implies u(e_{12}) = 0$$

$$e_{21} = g(\mathbf{x}_2^+) - g(\mathbf{x}_1^-) = 0.8 - 0.1 = 0.7 > 0 \implies u(e_{21}) = 1$$

$$e_{22} = g(\mathbf{x}_2^+) - g(\mathbf{x}_2^-) = 0.8 - 0.4 = 0.4 > 0 \implies u(e_{22}) = 1$$

$$\text{AUC} = \frac{1}{2 \times 2}(1 + 0 + 1 + 1) = \frac{3}{4} = 0.75$$

> **解读**：AUC = 0.75，说明随机取一对正负样本，模型有 75% 的概率对正样本给出更高的分数。

### 5.5 AUC 的四种典型情形

**情形 1：AUC = 1（完美分类）**

- 正类和负类的预测分布**完全分离**，无重叠
- 所有正样本的预测输出 > 所有负样本的预测输出
- ROC 曲线紧贴左上角

**情形 2：0.5 < AUC < 1（较好的分类器）**

- 正类和负类的预测分布**有部分重叠**
- 对于任意阈值，$\text{TPR} > \text{FPR}$（除 threshold=0 处）
- ROC 曲线在对角线上方

**情形 3：AUC = 0.5（随机猜测）**

- 正类和负类的预测分布**完全重叠**
- 对于任意阈值，$\text{FPR} = \text{TPR}$
- ROC 曲线就是对角线 $y = x$

**情形 4：AUC = 0（完全反向）**

- 所有正样本的预测输出 < 所有负样本的预测输出
- ROC 曲线在对角线下方
- 翻转预测标签即可获得 AUC = 1

### 5.6 AUC 的性质

> 1. **Scale-invariant（尺度不变性）**：改变预测输出的范围不影响 AUC 值。因为 AUC 只依赖样本的**排序**，不依赖绝对值。
> 
> 2. **Classification-threshold-invariant（阈值不变性）**：给定任何阈值，AUC 不变。AUC 衡量的是**所有阈值下的整体性能**。

**AUC 的优势与局限：**

| 方面 | 说明 |
|------|------|
| **优势** | 不受阈值选择影响；不受类别不平衡影响；提供全局性能度量 |
| **局限** | 当只关心某个特定阈值区间时，AUC 可能不够精确；不区分 FP 和 FN 的相对代价 |

---

## 6. 多分类的混淆矩阵

对于 $C$ 个类别的多分类问题，混淆矩阵扩展为 $C \times C$ 矩阵：

| | $\hat{P}_1$ (Predicted) | $\hat{P}_2$ (Predicted) | $\cdots$ | $\hat{P}_C$ (Predicted) |
|--|---|---|---|---|
| $P_1$ (Actual) | $P_{1,\hat{1}}$ | $P_{1,\hat{2}}$ | $\cdots$ | $P_{1,\hat{C}}$ |
| $P_2$ (Actual) | $P_{2,\hat{1}}$ | $P_{2,\hat{2}}$ | $\cdots$ | $P_{2,\hat{C}}$ |
| $\vdots$ | $\vdots$ | $\vdots$ | $\ddots$ | $\vdots$ |
| $P_C$ (Actual) | $P_{C,\hat{1}}$ | $P_{C,\hat{2}}$ | $\cdots$ | $P_{C,\hat{C}}$ |

- **对角线元素** $P_{k,\hat{k}}$：类别 $k$ 被正确分类的数量
- **非对角线元素** $P_{k,\hat{j}}$（$k \neq j$）：类别 $k$ 被错误分类为类别 $j$ 的数量

**多分类 Accuracy：**

$$\text{Accuracy} = \frac{\sum_{k=1}^{C} P_{k,\hat{k}}}{\sum_{k=1}^{C}\sum_{j=1}^{C} P_{k,\hat{j}}}$$

**注意**：多分类的 ROC 曲线非常复杂，目前学术界尚无统一定义。常见替代方案：
- **Macro-average**：对每个类别计算二分类 ROC/AUC，然后取平均
- **Micro-average**：将所有类别的 TP/FP/FN 汇总后计算

---

## 7. 计算性能与可维护性 (Optional)

除了预测准确性外，实际部署中还需考虑：

- **计算速度和效率** (Computational speed and efficiency)
- **模型的可维护性** (Maintainability)

> **注意**：高计算性能与软件质量属性（如灵活性、可扩展性、可用性、模块化、可维护性）之间往往存在**矛盾**。

- 当需要低级编程来充分利用并行硬件时，面向对象编程等常见编程范式可能无法使用
- 需要在**计算效率**和**可维护性/可移植性**之间谨慎权衡

参考文献：[ACM Digital Library - Software Quality vs Computational Performance](https://dl.acm.org/doi/10.5555/3019106.3019109)

---

## 8. 总结与公式速查表

### 完整性能评估流程

```
数据集
  │
  ├── 测试集（最终评估，不参与任何训练/调参）
  │
  └── 训练 + 验证（K-Fold CV 用于超参数调优）
         │
         ├── 选择最优超参数
         ├── 用全部训练数据重新训练最终模型
         └── 在测试集上评估最终性能
```

### 回归指标速查

| 指标 | 公式 | 特点 |
|------|------|------|
| MSE | $\frac{1}{n}\sum(y_i - \hat{y}_i)^2$ | 对大误差敏感 |
| RMSE | $\sqrt{\frac{1}{n}\sum(y_i - \hat{y}_i)^2}$ | 单位与原值一致 |
| MAE | $\frac{1}{n}\sum\|y_i - \hat{y}_i\|$ | 对离群值鲁棒 |
| R² | $1 - \frac{SS_{res}}{SS_{tot}}$ | 无量纲，可解释性强 |

### 二分类指标速查

| 指标 | 公式 | 别名 |
|------|------|------|
| Accuracy | $\frac{TP+TN}{TP+TN+FP+FN}$ | — |
| Precision | $\frac{TP}{TP+FP}$ | 查准率 |
| Recall / TPR | $\frac{TP}{TP+FN}$ | 查全率, Sensitivity |
| F1 Score | $\frac{2 \cdot P \cdot R}{P+R}$ | — |
| FPR | $\frac{FP}{FP+TN}$ | 1 - Specificity |
| TNR | $\frac{TN}{FP+TN}$ | Specificity |
| AUC | $\frac{1}{m^+m^-}\sum_i\sum_j u(e_{ij})$ | — |

### 关键概念对照

| 概念 | 核心要点 |
|------|---------|
| Cross-Validation | 使用 K-Fold 稳定评估泛化性能 |
| Confusion Matrix | 分类结果的完整描述 (TP/TN/FP/FN) |
| ROC Curve | 展示所有阈值下 TPR vs FPR |
| AUC | ROC 曲线下面积，阈值无关的整体性能度量 |
| EER | FPR = FNR 的工作点，越低越好 |
| Cost-Sensitive | 为不同错误赋予不同代价 |

---

## 9. 补充资料 (Supplementary Resources)

### 教材参考

| 教材 | 章节 | 内容 |
|------|------|------|
| Murphy, *Machine Learning: A Probabilistic Perspective* (2012) | Ch 5.3 | Cross-validation |
| Murphy, *MLAPP* | Ch 5.7 | Model selection |
| Bishop, *Pattern Recognition and Machine Learning* (2006) | Ch 1.3 | Model selection, Cross-validation |
| Müller & Guido, *Introduction to ML with Python* (2017) | Ch 5 | Cross-validation 详细实践 |
| Burkov, *The Hundred-Page Machine Learning Book* (2019) | Ch 5 | Model performance assessment |
| Goodfellow et al., *Deep Learning* (2016) | Ch 5.2-5.3 | Estimating generalization error |

### 在线资源

- [Wikipedia: Confusion Matrix](https://en.wikipedia.org/wiki/Confusion_matrix) — TP/TN/FP/FN 及各种指标的详细定义
- [Scikit-learn: ROC & AUC](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.auc.html) — AUC 计算的 Python 实现
- [Scikit-learn: Cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html) — 各种 CV 方法的实现与示例
- [Google ML Crash Course: Classification](https://developers.google.com/machine-learning/crash-course/classification) — ROC 与 AUC 的交互式教程

### 与其他讲次的联系

| 本讲内容 | 相关讲次 |
|---------|---------|
| Cross-validation | L12 Bias-Variance（模型选择动机） |
| MSE/MAE | L05 Linear Regression（损失函数） |
| 混淆矩阵 | L06 Logistic Regression（分类任务） |
| 阈值选择 | L06 Logistic Regression（Sigmoid 输出） |
| Cost-Sensitive | L07 SVM（不平衡数据处理） |
| AUC | L08 Decision Tree（Random Forest 评估） |
