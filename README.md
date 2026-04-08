# DDA3020 Machine Learning 课程笔记

> CUHK-SZ · School of Artificial Intelligence · Spring 2026
> 
> Instructor: Baoyuan Wu

## 📖 课程概述

本仓库包含 DDA3020 Machine Learning 课程的完整学习笔记，涵盖从数学基础到深度学习的全部 12 讲内容。笔记以课堂 slides 为主体，补充了教材推导、直觉解释和高置信度学术资料。

## 📂 笔记目录

### 第一部分：数学基础 (Lectures 1–4)

| 文件 | 主题 | 核心内容 |
|------|------|---------|
| [L01_Introduction.md](L01_Introduction.md) | 机器学习导论 | ML 定义、监督/无监督学习、基本概念、实践流程 |
| [L02_Probability.md](L02_Probability.md) | 概率论基础 | 随机变量、贝叶斯定理、常见分布、MLE/MAP |
| [L03_Linear_Algebra.md](L03_Linear_Algebra.md) | 线性代数基础 | 向量/矩阵运算、特征分解、SVD、矩阵微积分 |
| [L04_Optimization.md](L04_Optimization.md) | 优化方法 | 凸优化、梯度下降、SGD、牛顿法、KKT 条件 |

### 第二部分：经典机器学习算法 (Lectures 5–8)

| 文件 | 主题 | 核心内容 |
|------|------|---------|
| [L05_Linear_Regression.md](L05_Linear_Regression.md) | 线性回归 | OLS、正规方程、MLE 视角、Ridge/Lasso 正则化 |
| [L06_Logistic_Regression.md](L06_Logistic_Regression.md) | 逻辑回归 | Sigmoid 函数、交叉熵损失、Softmax 多分类 |
| [L07_SVM.md](L07_SVM.md) | 支持向量机 | Hard/Soft Margin、对偶问题、核方法 |
| [L08_Decision_Tree.md](L08_Decision_Tree.md) | 决策树与集成方法 | ID3/C4.5/CART、Random Forest、Boosting |

### 第三部分：深度学习与模型评估 (Lectures 9–13)

| 文件 | 主题 | 核心内容 |
|------|------|---------|
| [L09_Neural_Networks.md](L09_Neural_Networks.md) | 神经网络 (MLP) | 激活函数、前向/反向传播、优化器、正则化 |
| [L10_CNN.md](L10_CNN.md) | 卷积神经网络 | 卷积运算、经典架构 (LeNet→ResNet)、迁移学习 |
| [L11_RNN_Transformer.md](L11_RNN_Transformer.md) | RNN 与 Transformer | LSTM/GRU、Attention 机制、Transformer 架构 |
| [L12_Bias_Variance.md](L12_Bias_Variance.md) | Bias-Variance 权衡 | 偏差-方差分解推导、学习曲线、模型选择 |
| [L13_Performance_Evaluation.md](L13_Performance_Evaluation.md) | 性能评估 | Cross-Validation、Confusion Matrix、ROC/AUC |

## 📚 参考教材

| 教材 | 作者 | 备注 |
|------|------|------|
| **The Hundred-Page Machine Learning Book** | Andriy Burkov, 2019 | 课程必读，简明概览 |
| **Machine Learning: A Probabilistic Perspective (MLAPP)** | Kevin P. Murphy, 2012 | 课程必读，概率视角的系统性教材 |
| **Pattern Recognition and Machine Learning (PRML)** | Christopher Bishop, 2006 | 推荐，贝叶斯方法经典教材 |
| **Deep Learning** | Ian Goodfellow, Yoshua Bengio, Aaron Courville, 2016 | 推荐，深度学习部分参考 |
| **Introduction to ML with Python** | Andreas C. Müller & Sarah Guido, 2017 | 推荐，编程实践指南 |

## 📄 关键论文

- Vaswani et al., "Attention Is All You Need", NeurIPS 2017 — Transformer 架构原始论文
- He et al., "Deep Residual Learning for Image Recognition", CVPR 2016 — ResNet
- Hochreiter & Schmidhuber, "Long Short-Term Memory", Neural Computation, 1997 — LSTM
- Cortes & Vapnik, "Support-Vector Networks", Machine Learning, 1995 — SVM

## 📝 笔记特点

- **中文为主体**，专业术语和公式保持英文
- **LaTeX 数学公式**，完整推导过程
- **`>` 引用块**标注关键公式
- **直觉解释**，每个概念的"为什么重要"
- **教材对照**，标注对应 Murphy/Bishop/Burkov 章节
- **对比表格**，横向比较相似算法

## ⚠️ 声明

本笔记基于课堂 slides 整理，仅供学习参考。使用 LLM 辅助整理，特此声明。
