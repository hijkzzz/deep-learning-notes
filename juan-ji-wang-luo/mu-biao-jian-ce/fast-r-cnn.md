# Fast R-CNN

## 介绍

> [Fast R-CNN](http://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Girshick_Fast_R-CNN_ICCV_2015_paper.pdf)

本文提出了一种基于快速区域的卷积网络方法（Fast R-CNN）用于对象检测。 FastR-CNN建立在以前的工作基础上，使用深度卷积网络有效地分类对象提议。 与之前的工作相比，Fast R-CNN采用了多种创新技术来提高训练和测试速度，同时提高了检测精度。Fast R-CNN训练非常深VGG16网络比R-CNN快9倍，测试时间快213倍，并在PASCAL VOC2012上实现更高的mAP。 与SPPnet相比，Fast R-CNN将VGG16训练速度提高了3倍，测试速度提高了10倍，并且更加准确。

## 方法

![](../../.gitbook/assets/image%20%28154%29.png)

### The RoI pooling layer

RoI池化层使用最大池化将任何有效感兴趣区域内的特征转换成固定空间范围为H×W的小特征图。其基本原理是对整张特征图采用多个尺度不同的池化，然后合并成一个向量。

![](../../.gitbook/assets/image%20%2899%29.png)



### Initializing from pre-trained networks

我们尝试了三个预先训练过的ImageNet \[4\]网络，每个网络有五个最大池层，五个和十三个转换层之间（参见第4.1节网络详细信息）。 当预训练的网络初始化快速R-CNN网络时，它会经历三次转换。

首先，最后一个最大池层被一个RoIpooling层替换，RoIpooling层由H、W配置，使其与网络的第一个完全连接的层可兼容。

第二，网络的最后一个完全连接层和软最大值\(针对1000向图像网分类进行了训练\)被前面描述的两个同级层\(完全连接层和软最大值overK+1cat-egories和特定于类别的边界框回归器\)代替。

第三，修改网络以获取两个数据输入：图像列表和那些图像中的RoI列表。

### Fine-tuning for detection

损失函数由分类损失和box预测损失组成：

![](../../.gitbook/assets/image%20%2866%29.png)

![](../../.gitbook/assets/image%20%28109%29.png)

RoI池化反向传播

![](../../.gitbook/assets/image%20%2897%29.png)

其中 $$i^{*}(r, j)=\operatorname{argmax}_{i^{\prime} \in \mathcal{R}(r, j)}$$ 是最大池化选择的像素，x为RoI层输入，y为RoI层输出。

## 实验

![](../../.gitbook/assets/image%20%2878%29.png)

