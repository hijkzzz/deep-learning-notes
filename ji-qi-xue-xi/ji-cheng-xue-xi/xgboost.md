# XGBoost

## 介绍

> [XGBoost: A Scalable Tree Boosting System](https://arxiv.org/abs/1603.02754)

Tree boosting是一种高效且广泛使用的机器学习方法。 在本文中，我们描述了一个可扩展的端到端树推进系统XGBoost，它被数据科学家广泛使用，以在许多机器学习挑战中实现最先进的结果。我们提出了一种新的稀疏数据稀疏算法和近似树学习的加权量化框架。更重要的是，我们提供了有关缓存访问模式，数据压缩和分片的见解，以构建可扩展的树提升系统。通过结合这些见解，XGBoost使用比现有系统少得多的资源来扩展数十亿个示例。

## 方法

### TREE BOOSTING IN A NUTSHELL

我们回顾了梯度树提升算法。这个推导是从梯度提升中现有迭代的相同思想得出的。特别是第二阶方法起源于弗里德曼等人。\[12\]。我们对规定的目标进行了改进，这在实践中是有帮助的。

![](../../.gitbook/assets/image%20%2872%29.png)

#### Regularized Learning Objective

集成树模型使用K个树预测的结果相加

![](../../.gitbook/assets/image%20%28156%29.png)

其中 $$\mathcal{F}=\left\{f(\mathrm{x})=w_{q(\mathrm{x})}\right\}\left(q : \mathbb{R}^{m} \rightarrow T, w \in \mathbb{R}^{T}\right)$$ ，q表示每个树的结构，它将一个例子映射到相应的叶子索引。T是树中叶子的数量。 每个 $$f_{k}$$ 对应于独立的树结构q和叶子权重w。与决策树不同，每个回归树包含每个叶子上的连续分数，我们使用 $$w_i$$ 代表i-th 叶子的分数。

树的求解目标是最小化以下损失

![](../../.gitbook/assets/image%20%2878%29.png)

其中第二项是正则化，用来防止过拟合问题。

#### Gradient Tree Boosting

因为是加法模型，每次训练一颗新树，我们的求解目标为：

![](../../.gitbook/assets/image%20%2844%29.png)

二阶泰勒展开

![](../../.gitbook/assets/image%20%2888%29.png)

其中 $$g_{i}=\partial_{\hat{y}^(t-1)} l\left(y_{i}, \hat{y}^{(t-1)}\right) \text { and } h_{i}=\partial_{\hat{y}^{(t-1)}}^{2} l\left(y_{i}, \hat{y}^{(t-1)}\right)$$ 

去除常量

![](../../.gitbook/assets/image%20%28166%29.png)

设 $$I_{j}=\left\{i | q\left(\mathbf{x}_{i}\right)=j\right\}$$ 是叶子节点 $$j$$ 上的实例，代入展开：

![](../../.gitbook/assets/image%20%2896%29.png)

对于固定结构q\(x\)，我们可以计算出叶子的最佳权重

![](../../.gitbook/assets/image%20%2860%29.png)

并计算相应的最优值

![](../../.gitbook/assets/image%20%285%29.png)

等式\(6\)可用作衡量树形结构质量的评分函数q。这个分数就像评估决策树的不纯分数，除了它是为更广泛的目标函数而衍生的。图2举例说明了如何计算这个分数。



![](../../.gitbook/assets/image%20%2873%29.png)

通常，不可能枚举所有可能的树结构。 使用一种贪婪算法，该算法从单个叶子开始并迭代地将分支添加到树中。一个节点分裂成L、R两部分后的评分可以表示为：

![](../../.gitbook/assets/image.png)

#### Shrinkage and Column Subsampling

除了第二节中提到的正则化目标。 2.1，使用另外两种技术来进一步防止过度拟合。第一种技术是Fried-man引入的收缩\[11\]。 在树木提升的每个步骤之后，收缩比例新增加了因子η。 与随机优化中的学习率相似，收缩减少了每棵树的影响，并为未来的树木留下了空间来改进模型。第二种技术是列\(特征\)二次采样。这种技术在随机森林中使用。根据用户反馈，使用列子采样比传统的行子采样\(也受支持\)更能防止过度拟合。列子样本的使用也加速了后面描述的并行算法的计算。

### SPLIT FINDING ALGORITHMS

#### Basic Exact Greedy Algorithm

树学习中的一个关键问题是找到最佳分裂，如公式（7）所示。 为此，split finding算法枚举所有特征上的所有可能拆分。我们称之为exact greedy algorithm。大多数现有的单机树提升实现，例如scikit-learn \[20\]，R的gbm \[21\]以及XGBoost的单机版本支持精确的贪婪算法。确切的贪婪算法如Alg1所示。 计算连续特征的所有可能分裂的计算要求很高。 为了有效地执行此操作，算法必须首先根据特征值对数据进行排序，并按排序顺序访问数据，以便在方程（7）中累积结构分数的梯度统计数据。

![](../../.gitbook/assets/image%20%2824%29.png)

#### Approximate Algorithm

精确贪婪算法非常强大，因为它贪婪地收集所有可能的分裂点。然而，当数据没有完全装入内存时，就不可能有效地这样做。为了在这两个设置中支持有效的梯度树增强，需要近似算法。

我们总结了一个近似的框架，它类似于过去的文献\[17,2,22\]中提出的思想。 2.总之，该算法首先根据特征分布的百分位数提出了可以分割的分裂点（具体标准将在3.3节中给出）。然后，算法将连续特征映射到由这些候选点分割的分数点，聚合 统计数据，并根据聚合统计数据找到提案中的最佳解决方案。

![](../../.gitbook/assets/image%20%2823%29.png)

#### Weighted Quantile Sketch

给定 $$\mathcal{D}_{k}=\left\{\left(x_{1 k}, h_{1}\right),\left(x_{2 k}, h_{2}\right) \cdots\left(x_{n k}, h_{n}\right)\right\}$$ 表示每个训练实例的第k个特征值和二阶梯度统计。 我们可以定义一个排名函数。

![](../../.gitbook/assets/image%20%28179%29.png)

表示特征值小于z的实例的比例。目标是找到候选分割点 $$\left\{s_{k 1}, s_{k 2}, \cdots s_{k l}\right\}$$ ：

![](../../.gitbook/assets/image%20%2859%29.png)

这里 $$\epsilon$$ 是一个近似因子，直觉上，这意味着大约有 $$1/\epsilon$$ 个候选点。为了了解为什么 $$h_{i}$$ 会代表权重，我们可以将等式\(3\)改写为：

![](../../.gitbook/assets/image%20%28165%29.png)

这是用 $$g_{i} / h_{i}$$ 和 $$h_{i}$$ 精确加权的平方损失。

#### Sparsity-aware Split Finding

在许多现实问题中，输入稀疏是很常见的。使算法了解数据中的稀疏模式是非常重要的。为了做到这一点，我们建议在每个树节点中添加一个默认方向，如图4所示。当稀疏矩阵x中缺少A值时，实例将被分类为默认方向。

![](../../.gitbook/assets/image%20%28116%29.png)

从数据中学习最佳默认方向。 该算法显示在Alg3中。 关键的改进是只访问non-missing entries。

![](../../.gitbook/assets/image%20%28221%29.png)

### SYSTEM DESIGN

### Column Block for Parallel Learning



