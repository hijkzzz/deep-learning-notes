# Deep Face Recognition

## 介绍

> [Deep Face Recognition](https://www.robots.ox.ac.uk/~vgg/publications/2015/Parkhi15/parkhi15.pdf)

本文的目标是人脸识别 - 来自单张照片或视频中跟踪的一组脸部。 该领域的最新进展归因于两个因素：（i）使用卷积神经网络（CNN）的任务的端到端学习，以及（ii）非常大规模的训练数据集的可用性。

我们做了两个贡献：首先，我们展示了如何通过循环中的自动化和人工组合来组装一个非常大规模的数据集（2.6M图像，超过2.6K人），并讨论数据纯度和时间之间的权衡 ; 第二，我们通过深度网络训练和人脸识别的复杂性来提出方法和程序，以达到标准LFW和YTF面部基准的可比较的最新技术成果。

## 方法

### Dataset Collection

![](../../.gitbook/assets/image%20%2810%29.png)

1. 构建数据集的第一个阶段是获取候选身份名称列表以获取面孔。这个想法是关注名人和公众人物，如演员或政治家，以便可以找到足够数量的不同图像。 网络，以及在下载图像时避免任何隐私问题。
2. 在Google和Bing图像搜索中都有2,622个名人名字，然后在名称后附加关键字“actor”。
3. 使用分类器自动删除每组中的任何错误的面部。
4. 由两个不同的搜索引擎发现的相同图像的完全重复图像，或者在不同的互联网位置上找到的相同图像的副本被删除。
5. 此时，有2，622个身份，每个身份最多有1，000个图像。这个最后阶段的目的是使用人工注释来提高数据的纯度\(精度\)。

### Network architecture and training

#### Learning a face classifier

首先用深度卷积神经网络训练一个人脸分类器，训练好后移除softmax分类层，前一层输出的分数向量可视为人脸特征。

#### Learning a face embedding using a triplet loss

三联损失训练旨在学习在最终应用中表现良好的得分向量，即。 通过比较欧氏空间中的人脸描述符进行身份验证。这类似于“度量学习”的灵感，并且像许多度量学习方法一样，用于学习同时具有独特性和紧凑性的投影，同时实现降维。即用一个仿射变换投影 $$\mathbf{x}_{t}=W^{\prime} \phi\left(\ell_{t}\right) /\left\|\phi\left(\ell_{t}\right)\right\|_{2}, W^{\prime} \in \mathbb{R}^{L \times D}$$ ，其中L&lt;&lt;D。用于训练的损失函数为：

![](../../.gitbook/assets/image%20%28114%29.png)

三元组 $$(a,p,n)$$ 包含一个人脸图像锚点以及锚点身份的正面和负面示例。

#### Architecture

![](../../.gitbook/assets/image%20%2828%29.png)

## 实验

![](../../.gitbook/assets/image%20%2815%29.png)





