# FCN

## 介绍

> [Fully convolutional networks for semantic segmentation](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Long_Fully_Convolutional_Networks_2015_CVPR_paper.pdf)

卷积网络是一种功能强大的可视化模型，它可以生成特性的层次结构。我们表明，卷积网络本身，经过训练的端到端，像素到像素，在语义分离方面超过了最先进的水平。我们的主要观点是建立“完全卷积”网络，它接受任意大小的输入，并以有效的推理和学习产生相应大小的输出。我们定义并详细描述了全卷积网络的空间，解释了它们在空间密度预测任务中的应用，并将它们与之前的模型联系起来。我们将当代的分类网络\(AlexNet\[20\]、VGG net\[31\]和GoogLeNet\[32\]\)调整为完全卷积的网络，并通过微调\[3\]将其学习到的表示转移到分割任务中。然后，我们定义了askip架构，它结合了来自深层、粗糙层的语义信息和来自底层、精细层的外观信息，从而生成精确、详细的segmentations。我们的全卷积网络实现了PASCAL VOC\(相对于2012年的62.2%的平均IU，提高了20%\)、NYUDv2和SIFTFlow的最先进分割，而一个典型图像的推理时间不到五分之一秒。

## 方法

简单的说，FCN与CNN的区别在于FCN把CNN最后的全连接层换成卷积层，输出一张已经label好的图。

![](../../.gitbook/assets/image%20%28164%29.png)

![](../../.gitbook/assets/image%20%2875%29.png)

具体来说，通过CNN下采样提取图片特征，然后通过上采样的方式输出分类特征图（skip连接补充细节信息），即每个像素输出N个值（所以输出有N张特征图），表示该像素属于哪个类（即softmax分类）。

![](../../.gitbook/assets/image%20%28119%29.png)

### 实现细节

作者提供的网络模型里第一个卷积层对输入图像添加了100个像素的padding，这是因为如果输入图片大小不够，会导致下采样无法进行，这也到导致FCN引入了不少噪音。在DeepLab中使用空洞卷积解决这个问题。

## 实验

对于混合不同的下采样stride的细节信息，效果是不一样的，总体来说stride越小效果越好。



![](../../.gitbook/assets/image%20%28154%29.png)

数据集测试

![](../../.gitbook/assets/image%20%2889%29.png)

![](../../.gitbook/assets/image%20%28105%29.png)

![](../../.gitbook/assets/image%20%28127%29.png)



