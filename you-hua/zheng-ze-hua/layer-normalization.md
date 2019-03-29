# Layer Normalization

## 介绍

> [Layer Normalization](https://arxiv.org/pdf/1607.06450.pdf)

训练最先进的深层神经网络在计算上是昂贵的。减少训练时间的一个方法是使神经元的激活规范化。最近引入的称为批量标准化的技术使用在一小批训练案例中对神经元求和输入的分布来计算方法和方差，然后将其用于在每个训练案例中对该神经元的总和输入进行标准化。这显着减少了前馈神经网络中的训练时间。 然而，批量标准化的效果取决于小批量大小，并且如何将其应用于递归神经网络并不明显。 在本文中，我们通过计算用于归一化的均值和方差从一个层中的所有总输入到一个层中的神经元，将批量归一化转换为层归一化。与批量归一化一样，我们也给每个神经元提供了自己的自适应偏差和增益，这些偏差和增益在归一化之后但在非线性之前应用。 与批量标准化不同，层标准化在训练和测试时间执行完全相同的计算。通过在每个时间步骤分别计算正规化统计量，也可以直接应用于递归神经网络。层规范化对于稳定递归网络中的隐藏状态动态非常有效。实际上，我们表明层标准化可以大大减少与以前公布的技术相比的训练时间。

## 方法

### BNN

如下面的公式，在BNN中，通过小批次数据的均值/方差对网络输出进行归一化。

$$
\overline{a}_{i}^{l}=\frac{g_{i}^{l}}{\sigma_{i}^{l}}\left(a_{i}^{l}-\mu_{i}^{l}\right) \quad \mu_{i}^{l}=\underset{\mathbf{x} \sim P(\mathbf{x})}{\mathbb{E}}\left[a_{i}^{l}\right] \quad \sigma_{i}^{l}=\sqrt{\underset{\mathbf{x} \sim P(\mathbf{x})}{\mathbb{E}}\left[\left(a_{i}^{l}-\boldsymbol{\mu}_{i}^{l}\right)^{2}\right]}
$$

### Layer Normalization

但是这种方法不适用于RNN，因为对于不同的输入，RNN的深度是不确定的，因此无法为每层保存单独的均值/方差。

$$
\mu^{l}=\frac{1}{H} \sum_{i=1}^{H} a_{i}^{l} \quad \sigma^{l}=\sqrt{\frac{1}{H} \sum_{i=1}^{H}\left(a_{i}^{l}-\mu^{l}\right)^{2}}
$$

所以我们的方法是，对同一层之间的输出进行跨通道规范化，这样便不依赖于小批次均值和方差。

在RNN中计算过程如下所示：

$$
\mathbf{h}^{t}=f\left[\frac{\mathbf{g}}{\sigma^{t}} \odot\left(\mathbf{a}^{t}-\mu^{t}\right)+\mathbf{b}\right] \quad \mu^{t}=\frac{1}{H} \sum_{i=1}^{H} a_{i}^{t} \quad \sigma^{t}=\sqrt{\frac{1}{H} \sum_{i=1}^{H}\left(a_{i}^{t}-\mu^{t}\right)^{2}}
$$

## 性能

在卷积网络中效果差于BNN，在RNN中效果很好。



