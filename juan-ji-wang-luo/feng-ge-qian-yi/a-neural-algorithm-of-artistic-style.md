# A Neural Algorithm of Artistic Style

## 介绍

> [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576)

在美术领域，尤其是绘画领域，人类已经掌握了通过在一幅图像的内容和风格之间构成复杂的相互作用来创造独特视觉体验的技能。到目前为止，这一过程的算法基础尚不清楚，也不存在具有类似能力的人工系统。然而，在视觉感知的其他关键领域，例如物体和面部识别，最近一类被称为深度神经网络的生物启发视觉模型证明了接近人类的表现。这里我们介绍一个基于深度神经网络的人工系统，它可以创建高感知质量的艺术图像。该系统利用神经表示分离和重组任意图像的内容和风格，为艺术图像的创作提供神经算法。此外，鉴于性能优化的人工神经网络和生物视觉之间的惊人相似性，我们的工作提供了一条通向人类如何创造和感知艺术意象的算法基础的道路。

## 方法

### VGG

![](../../.gitbook/assets/image%20%28119%29.png)

![](../../.gitbook/assets/image%20%28105%29.png)

如上图，作者通过可视化卷积层的输出发现，卷积神经网络的深层输出图片的语义信息，而浅层输出图片的风格信息。所以风格迁移的关键思想是使输出图片的语义信息近似给定内容图，风格信息近似给定风格图。

### 损失函数

#### 语义损失

直接用特征图的距离作为损失

$$
\mathcal{L}_{\text {content}}(\vec{p}, \vec{x}, l)=\frac{1}{2} \sum_{i, j}\left(F_{i j}^{l}-P_{i j}^{l}\right)^{2}
$$

其中$$l$$为所使用的卷积层，即：conv4\_2

对应的梯度

$$
\frac{\partial \mathcal{L}_{\text {content}}}{\partial F_{i j}^{l}}=\left\{\begin{array}{ll}{\left(F^{l}-P^{l}\right)_{i j}} & {\text { if } F_{i j}^{l}>0} \\ {0} & {\text { if } F_{i j}^{l}<0}\end{array}\right.
$$

#### 风格损失

这里用到了Gram matrix度量两张图片的风格距离

$$
G_{i j}^{l}=\sum_{k} F_{i k}^{l} F_{j k}^{l}
$$

Gram Matrix可看做是图像各特征之间的偏心协方差矩阵（即没有减去均值的协方差矩阵），Gram计算的是两两特征之间的相关性，哪两个特征是同时出现的，哪两个是此消彼长的等等。另一方面，Gram的对角线元素，还体现了每个特征在图像中出现的量，因此，Gram矩阵可以度量各个维度自己的特性以及各个维度之间的关系，所以可以反映整个图像的大体风格。只需要比较Gram矩阵就可以比较两个图像的风格差异了

$$
E_{l}=\frac{1}{4 N_{l}^{2} M_{l}^{2}} \sum_{i, j}\left(G_{i j}^{l}-A_{i j}^{l}\right)^{2}
$$

$$
\mathcal{L}_{s t y l e}(\vec{a}, \vec{x})=\sum_{l=0}^{L} w_{l} E_{l}
$$

其中$$l$$为使用的卷积层，即：conv1\_1’, ‘conv2\_1’, ‘conv3\_1’, ‘conv4\_1’ and ‘conv5\_1， $$w_l=1/5$$ 

对应的梯度

$$
\frac{\partial E_{l}}{\partial F_{i j}^{l}}=\left\{\begin{array}{ll}{\frac{1}{N_{t}^{2} M_{t}^{2}}\left(\left(F^{l}\right)^{\mathrm{T}}\left(G^{l}-A^{l}\right)\right)_{j i}} & {\text { if } F_{i j}^{l}>0} \\ {0} & {\text { if } F_{i j}^{l}<0}\end{array}\right.
$$



#### 整体损失

$$
\mathcal{L}_{\text {total}}(\vec{p}, \vec{a}, \vec{x})=\alpha \mathcal{L}_{\text {content}}(\vec{p}, \vec{x})+\beta \mathcal{L}_{\text {style}}(\vec{a}, \vec{x})
$$

### 训练

我们对白噪声图像执行梯度下降，以找到与原始图像的特征响应相匹配的另一幅图像。

## 效果

![](../../.gitbook/assets/image%20%28172%29.png)



