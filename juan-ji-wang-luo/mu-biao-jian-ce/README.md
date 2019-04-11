# 目标检测

### 介绍

如图\(b\)即目标检测，神经网络需预测目标的类型以及所在区域。

![](../../.gitbook/assets/image%20%2893%29.png)

### 性能评价

precision（精度）和recall（召回率）分别从预测正类的正确率和预测正类的全面性评价分类器的性能。

![](../../.gitbook/assets/image%20%28113%29.png)

此外，accuracy（准确度）无论正负预测对的比例：

$$
\operatorname{accuracy}=\frac{\text {Truepositives}+\text {Truenegatives}}{\text {Truepositives}+\text {Falsenegatives}+\text {Truenegatives}+\text {Falsepositives}}
$$

有了精度和召回率，可以得到PR曲线，PR曲线下方面积越大则分类器综合性能越好：

![](../../.gitbook/assets/image%20%2812%29.png)

IoU 的全称为交并比（Intersection over Union），IoU 计算的是 “预测的边框” 和 “真实的边框” 的交集和并集的比值：

![](../../.gitbook/assets/image%20%28129%29.png)

IoU使我们便能定义目标检测中正类和负类，如IoU&gt;50%为正类，否则为负类。

我们可以用这个分类计算目标检测网络的精度和召回率，并画出PR曲线，其下部面积就是AP。至于mAP则是测试样本的均值。

### NMS

 非极大值抑制（Non-Maximum Suppression，NMS），顾名思义就是抑制不是极大值的元素，可以理解为局部最大搜索。这个局部代表的是一个邻域，邻域有两个参数可变，一是邻域的维数，二是邻域的大小。这里不讨论通用的NMS算法\(参考论文《[Efficient Non-Maximum Suppression](https://pdfs.semanticscholar.org/52ca/4ed04d1d9dba3e6ae30717898276735e0b79.pdf)》对1维和2维数据的NMS实现\)，而是用于目标检测中提取分数最高的窗口的。例如在行人检测中，滑动窗口经提取特征，经分类器分类识别后，每个窗口都会得到一个分数。但是滑动窗口会导致很多窗口与其他窗口存在包含或者大部分交叉的情况。这时就需要用到NMS来选取那些邻域里分数最高（是行人的概率最大），并且抑制那些分数低的窗口。  
NMS在计算机视觉领域有着非常重要的应用，如视频目标跟踪、数据挖掘、3D重建、目标识别以及纹理分析等。

比如对于目标检测来说，通常神经网络会输出多个重叠的框  


![](../../.gitbook/assets/image%20%28149%29.png)

NMS的处理流程为：先找到置信概率最大的输出框，然后计算其它框与改框的IoU，如果大于一个阈值如0.5则扔掉这个框，并标记置信度最大的框。接下来选择下一个置信度最大的框，继续剔除可能的重叠框。如此迭代即可。

