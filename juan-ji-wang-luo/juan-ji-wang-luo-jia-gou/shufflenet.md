# ShuffleNet

## 介绍

[ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices](https://arxiv.org/pdf/1707.01083.pdf)

我们提出了一个名为ShuffleNet的极其计算效率的CNN架构，该架构专为具有非常有限的计算能力（例如，10-150 MFLOP）的移动设备而设计。新的体系结构采用了两种新的运算，逐点群卷积和信道洗牌，在保持精确性的同时大大降低了计算成本。ImageNet分类和MSCOCO对象检测的实验证明了ShuffleNet优于其他结构的性能，例如： 在40-MFLOPs的计算预算下，在Ima-geNet分类任务中，最近的MobileNet \[12\]比较低的top-1error（绝对7.8％）。在基于ARM的移动设备上，ShuffleNet实现〜13倍于AlexNet的实际加速，同时保持了相当的准确性。

## 方法

### Channel Shuffle for Group Convolutions

最先进的网络，如Xception \[3\]和ResNeXt \[40\]，将高效的深度可分离卷积或群组卷积引入构建块，以在表示能力和计算成本之间取得良好的折衷。但是，我们注意到两种设计都没有完全采用 考虑到1×1对称（也称为pointwise convolutionsin \[12\]），这需要相当大的计算。

![](../../.gitbook/assets/image%20%2854%29.png)

为了解决这个问题，一个简单的解决方案是：如果我们允许组卷积从不同组获得输入数据（如图1（b）所示），输入和输出通道将完全相关，同时降低了计算量。

### ShuffleNet Unit

利用通道shuffle操作，我们提出了一个专为小网络设计的新型ShuffleNet单元，如图2c所示。

![](../../.gitbook/assets/image%20%2821%29.png)

### Network Architecture

建议的网络主要由一组分为三个阶段的ShuffleNet单元组成。

![](../../.gitbook/assets/image%20%28208%29.png)

## 实验

![](../../.gitbook/assets/image%20%28159%29.png)

