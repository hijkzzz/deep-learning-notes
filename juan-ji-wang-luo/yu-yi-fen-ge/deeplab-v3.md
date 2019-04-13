# DeepLab V3+

## 介绍

> [Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](https://arxiv.org/pdf/1802.02611v3.pdf)

空间金字塔池模块或编码 - 解码器结构用于深度神经网络中的语义分割任务。 通过以多个速率和多个有效视野进行过滤器或池化操作来探测传入特征，然后后者网络可以通过逐渐恢复空间信息来捕获更清晰的对象边界，从而能够编码多尺度上下文信息。在这项工作中，我们建议结合两种方法的优点。 具体来说，我们提出的模型DeepLabv3 +通过添加一个简单但有效的解码器模块来扩展DeepLabv3，以便特别是沿着对象边界细化分割结果。我们进一步探索Xception模型，并将深度可分离卷积应用于Atrous Spatial Pyramid Pooling和解码器模块，从而产生更快更强的编码器 - 解码器网络。我们在PASCAL VOC 2012和city scapes数据集上验证了该模型的有效性，无需任何后处理，测试集性能分别达到89.0 %和82.1 %。[https://github.com/tensorflow/models/tree/master/research/deeplab](https://github.com/tensorflow/models/tree/master/research/deeplab)

## 方法

![](../../.gitbook/assets/image%20%2875%29.png)

### Encoder-Decoder with Atrous Convolution

#### Atrous convolution

Atrous卷积是一种强大的工具，它允许我们明确地控制深度卷积神经网络计算的特征的分辨率，并调整滤波器的视场以捕获多尺度信息，推广标准卷积运算。在二维信号的情况下，对于每个位置 $$i$$ ，输出特征图 $$ y$$ 和卷积滤波器 $$w$$ ，在输入特征图上应用atrous卷积如下：

$$
\boldsymbol{y}[i]=\sum_{\boldsymbol{k}} \boldsymbol{x}[i+r \cdot \boldsymbol{k}] \boldsymbol{w}[\boldsymbol{k}]
$$

#### Depthwise separable convolution

深度可分离卷积，将标准卷积转化为深度卷积，然后是点卷积\(即1×1卷积\)，大大降低了计算复杂度。具体来说，深度卷积独立地为每个输入通道执行空间卷积，而逐点卷积用于组合深度卷积的输出。

![](../../.gitbook/assets/image%20%28177%29.png)

在这项工作中，我们将所得到的卷积称为atrous separable convolution，并发现atrous separable convolution显着降低了所提出模型的计算复杂度，同时保持了相似（或更好）的性能。

#### DeepLabv3 as encoder

DeepLabv3 \[23\]采用atrous卷积\[69,70,8,71\]来提取在任意分辨率下由深度卷积神经网络计算的特征。我们将输入图像空间分辨率与最终输出分辨率（在全局池或完全连接层之前）的比率表示为输出步幅。对于图像分类的任务，最终特征图的空间分辨率通常比输入图像分辨率小32倍，因此输出stride = 32。对于语义分割的任务，可以通过去除最后一个（或两个）块中的striding和相应地应用atrous卷积来使输出stride = 16（或8）来进行更密集的特征提取。此外，DeepLabv3增强了Atrous SpatialPyramid Pooling模块，该模块通过应用不同rates的atrous卷积来探测多个尺度的卷积特征，具有图像级别的特征。在我们提出的编码器-解码器结构中，我们使用原始DeepLabv3作为中逻辑之前的最后一个特征映射作为编码器输出。注意编码器输出特征图包含256个通道和丰富的语义信息。此外，根据计算预算，可以通过应用高阶卷积以任意分辨率提取特征。

#### Proposed decoder

DeepLabv3的编码器通常在输出步幅= 16的情况下进行计算。在\[23\]的工作中，这些特征是双线性上采样的16倍，这可以被认为是一个简单的解码器模块。然而，这个简单的解码器模块可能无法成功恢复对象分割细节。因此，我们提出了一个简单而有效的解码器模块，如图2所示。



![](../../.gitbook/assets/image%20%2878%29.png)

### Modified Aligned Xception

Xception模型\[26\]在ImageNet \[74\]上展示了具有快速计算能力的图像分类结果。最近，MSRA团队\[31\]修改了Xception模型（称为Aligned Xception）并进一步推动了对象检测任务的性能。受这些发现的启发，我们在同一方向上工作，以使Xception模型适应语义图像分割的任务。特别是，我们在MSRA的修改之上做了一些更改，即（1）与\[31\]相同的更深层次的Xception，除了我们不修改入口流网络结构以实现快速计算和内存效率，（2）所有最大池化 操作由带有跨步的深度可分离卷积代替，这使我们能够应用巨大的可分离卷积以任意分辨率提取特征图（另一种选择是将激励算法扩展到最大池操作），以及（3）额外的BatchNormalization \[75\]和ReLU 在每次3×3深度卷积之后添加激活，类似于MobileNet设计\[29\]。 有关详细信息，请参见图4。

![](../../.gitbook/assets/image%20%2890%29.png)

## 测试

### ResNet-101

![](../../.gitbook/assets/image%20%2832%29.png)

### Xception

![](../../.gitbook/assets/image%20%2855%29.png)



