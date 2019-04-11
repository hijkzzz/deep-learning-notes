# Batch Normalization

## 介绍

> [Batch Normalization: Accelerating Deep Network Training by ReducingInternal Covariate Shift](https://arxiv.org/abs/1502.03167)

训练深度神经网络很复杂，因为在训练过程中每层输入的分布发生变化，因为前一层的参数发生了变化。 这通过要求较低的学习率和仔细的参数初始化来减慢训练，并且使得训练具有饱和非线性的模型变得非常困难。我们将这种现象称为内部协变量移位，并通过规范化层输入来解决问题。我们的方法的优势在于将标准化作为模型体系结构的一部分，并为每个训练批次执行标准化。批量标准化使我们能够获得更高的学习率并且不太关心初始化，并且在某些情况下可以消除对Dropout的需求。 应用于最先进的图像分类模型，批量标准化实现了相同的准确度，培训步骤减少了14倍，并且显着地超过了原始模型。使用批量标准化网络的集合，我们改进了ImageNet分类的最佳公布结果：达到4.82％的前5个测试错误，超出了人类评估者的准确性。

## 算法

### 批标准化

其中scale/shift变换系数是可训练的

![](../../.gitbook/assets/image%20%2887%29.png)

参数的反向传播公式

![](../../.gitbook/assets/image%20%28133%29.png)

### 伪代码

![](../../.gitbook/assets/image%20%28140%29.png)

需要注意的是训练的时候使用批次样本的均值与方差，预测的时候使用所有训练样本的均值与方差。

### 卷积网络

注意前面写的都是对于一般情况，对于卷积神经网络有些许不同。因为卷积神经网络的特征是对应到一整张特征响应图上的，所以做BN时也应以响应图为单位而不是按照各个维度。

### 调参技巧

* Increase learning rate
* Remove Dropout
* Shuffle training examples more thoroughly
* Reduce the L2weight regularization
* Accelerate the learning rate decay
* Remove Local Response Normalization
* Reduce the photometric distortions

## 实验

![](../../.gitbook/assets/image%20%28144%29.png)



