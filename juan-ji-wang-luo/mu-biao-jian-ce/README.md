# 目标检测

### 介绍

如图\(b\)即目标检测，神经网络需预测目标的类型以及所在区域。

![](../../.gitbook/assets/image%20%2823%29.png)

### 性能评价

precision（精度）和recall（召回率）分别从预测正类的正确率和预测正类的全面性评价分类器的性能。

![](../../.gitbook/assets/image%20%2830%29.png)

此外，accuracy（准确度）无论正负预测对的比例：

$$
\operatorname{accuracy}=\frac{\text {Truepositives}+\text {Truenegatives}}{\text {Truepositives}+\text {Falsenegatives}+\text {Truenegatives}+\text {Falsepositives}}
$$

有了精度和召回率，可以得到PR曲线，PR曲线下方面积越大则分类器综合性能越好：

![](../../.gitbook/assets/image%20%283%29.png)

IoU 的全称为交并比（Intersection over Union），IoU 计算的是 “预测的边框” 和 “真实的边框” 的交集和并集的比值：

![](../../.gitbook/assets/image%20%2832%29.png)

IoU使我们便能定义目标检测中正类和负类，如IoU&gt;50%为正类，否则为负类。

我们可以用这个分类计算目标检测网络的精度和召回率，并画出PR曲线，其下部面积就是AP。至于mAP则是测试样本的均值。

