# Visualizing and Understanding Convolutional Networks

## 介绍

> [Visualizing and Understanding Convolutional Networks](https://arxiv.org/pdf/1311.2901.pdf)

大型卷积网络模型最近在ImageNet基准测试中表现出了令人印象深刻的分类性能（Krizhevsky等，2012）。但是，他们对于为什么它们如此良好地表现或者如何改进它们并不清楚。 在本文中，我们解决了这两个问题。我们介绍了一种新颖的可视化技术，可以深入了解中间特征层的功能和分类器的操作。 这些可视化用于诊断角色，使我们能够找到比ImageNet分类基准更优于Krizhevskyet的模型架构。 我们还进行消融研究，以发现不同模型层的性能贡献。 我们展示了我们的ImageNet模型能够很好地推广到其他数据集：当softmax分类器被重新训练时，它令人信服地胜过了当前最先进的结果:Caltech-101和Caltech-256数据集。

## 方法

为了检查convnet，如图1所示，将一个解卷积网络连接到它的每一层，提供一条返回图像像素的连续路径。

![](../../.gitbook/assets/image%20%285%29.png)

### Unpooling

在convnet中，最大池化运算是不可逆的，但是我们可以通过在一组switch variables中记录每个池区域内最大值的位置来获得近似逆运算。在deconvnet中，unpooling操作使用这些switch将重建从上面的层放到适当的位置，保持激活的结构。有关该程序的说明，请参见图1 \(底部\)。

### Rectification

该网络使用了Relu非线性激活函数，它可以纠正特征图，从而确保特征图始终为正。 为了在每一层获得有效的特征重建（也应该是正的），我们通过Relu重建信号。

### Filtering

convnet使用学习的filter来与前一层的特征图卷积。为了反转这个，deconvnet使用相同filter的转置版本，但应用于rectified后的特征图，而不是之前层的输出。 实际上，这意味着垂直和水平翻转每个filter。

从较高层向下投影使用convneton中最大池在向上投影时产生的switch settings。由于这些switch settings特定于给定的输入图像，因此从单次激活获得的重建类似于原始输入图像的一小部分，其结构根据它们对特征激活的贡献而被加权。由于模型是有判别式训练的，因此它们隐含地显示输入图像的哪些部分是有区别的。 请注意，这些预测不是来自模型的样本，因为不涉及生成过程。

## 可视化卷积



![](../../.gitbook/assets/image%20%286%29.png)

![](../../.gitbook/assets/image%20%287%29.png)



