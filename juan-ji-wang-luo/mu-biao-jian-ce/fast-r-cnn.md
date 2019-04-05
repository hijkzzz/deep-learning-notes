# Fast R-CNN

## 介绍

> [Fast R-CNN](http://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Girshick_Fast_R-CNN_ICCV_2015_paper.pdf)

本文提出了一种基于快速区域的卷积网络方法（Fast R-CNN）用于对象检测。 FastR-CNN建立在以前的工作基础上，使用深度卷积网络有效地分类对象提议。 与之前的工作相比，Fast R-CNN采用了多种创新技术来提高训练和测试速度，同时提高了检测精度。Fast R-CNN训练非常深VGG16网络比R-CNN快9倍，测试时间快213倍，并在PASCAL VOC2012上实现更高的mAP。 与SPPnet相比，Fast R-CNN将VGG16训练速度提高了3倍，测试速度提高了10倍，并且更加准确。

## 方法

![](../../.gitbook/assets/image%20%28121%29.png)

### The RoI pooling layer

### Initializing from pre-trained networks

### Fine-tuning for detection

### Scale invariance

## 实验

![](../../.gitbook/assets/image%20%2860%29.png)

