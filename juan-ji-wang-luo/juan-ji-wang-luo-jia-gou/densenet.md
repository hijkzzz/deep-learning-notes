# DenseNet

## 介绍

> [Densely Connected Convolutional Networks](https://arxiv.org/pdf/1608.06993.pdf)

最近的工作表明，卷积网络可以更深入，更准确，更有效，如果它们包含靠近输入的层和靠近输出的层之间的较短连接。 在本文中，我们将观察并引入密集的网络（DenseNet），它以前馈的方式将每个层连接到每个层。在传统的L层卷积网络中有L个连接，每层与其后续层之间一个，我们的网络有 $$\frac{L(L+1)}{2}$$ 个直接连接。对于每一层，前面所有层的特征映射都被用作输入，而它自己的特征映射被用作所有后续层的输入。DenseNets有几个优点:它们减轻了消失梯度问题，加强了特征传播，鼓励了特征重用，并大大减少了参数的数量。我们在四个高度竞争的对象识别基准测试任务（CIFAR-10，CIFAR-100，SVHN和ImageNet）上评估我们提出的架构。DenseNets在大多数方面获得了显著的进步，同时需要更少的计算来实现高性能。代码和预先训练的模型可从以下网址获得：[https://github.com/liuzhuang13/DenseNet](https://github.com/liuzhuang13/DenseNet)。

![](../../.gitbook/assets/image%20%282%29.png)

## 方法

### ResNets

$$
\mathbf{x}_{\ell}=H_{\ell}\left(\mathbf{x}_{\ell-1}\right)+\mathbf{x}_{\ell-1}
$$

ResNets的一个优点是梯度可以通过等价函数直接从后面的层流到更早的层。 然而，等价函数和 $$H_{\ell}$$ 的输出通过求和相结合，这可能阻碍网络中的信息流。

### Dense connectivity

为了进一步改善层之间的信息流，我们提出了一个不同的连接模式：我们引入了来自任何后续层的直接连接。图1展示出了所得到的DenseNet的结构。 因此，第 $$\ell^{t h}$$ 层接收所有前面层的特征图：

$$
\mathbf{x}_{\ell}=H_{\ell}\left(\left[\mathbf{x}_{0}, \mathbf{x}_{1}, \ldots, \mathbf{x}_{\ell-1}\right]\right) \ \ \ \ \ \ (2)
$$

### Composite function

我们将 $$H_{\ell}$$ 定义为三个连续操作的复合函数：批量归一化（BN），然后是整流的线性单元（ReLU）和3×3卷积（Conv）。

### Pooling layers

在 $$Eq(2)$$ 中使用的串联操作。 当特征图的大小改变时是不可行的。然而，卷积网络的一个重要部分是下采样层，它改变了特征映射的大小。为了便于在我们的架构中进行下采样，我们将网络分成多个密集连接的模块；参见图2。



![](../../.gitbook/assets/image%20%28119%29.png)

我们将Block之间的层称为转移层，它们进行卷积和池化。 在我们的实验中使用的转换层包括批量标准化层和1×1卷积层，接着是2×2平均池化层。

### Bottleneck layers

尽管每个图层仅生成输出k个特征图，但它通常具有更多输入。 在\[37,11\]中已经注意到，在每个3×3卷积之前可以将1×1卷积作为瓶颈层进行处理，以减少输入特征图的数量，从而提高计算效率。我们发现这种设计对DenseNet特别有效，我们参考了具有这种瓶颈层的网络。

### Compression

为了进一步提高模型的紧凑性，我们可以在转移层减少特征映射数量。

最后得到模型的架构

![](../../.gitbook/assets/image%20%2811%29.png)

## 测试

![](../../.gitbook/assets/image%20%2848%29.png)



