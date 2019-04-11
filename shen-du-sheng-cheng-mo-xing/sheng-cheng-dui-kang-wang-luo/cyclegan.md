# CycleGAN

## 介绍

> [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593)

图像到图像的转换是一类视觉和图形问题，其目标是使用一组对齐的图像对来学习输入图像和输出图像之间的映射。但是，对于许多任务，将无法使用配对的培训数据。在没有成对例子的情况下，我们提出了一种学习将图像从源域翻译成目标域的方法。我们的目标是通过对抗损失学习映射 $$G : X \rightarrow Y$$ 使得来自分布 $$G(X)$$ 的图像与分布 $$Y$$ 无法区分。为这个映射是高度欠约束的，我们用一个逆映射对它进行耦合 $$F : Y \rightarrow X$$ 并引入循环一致性损失来强制执行 $$F(G(X)) \approx X$$ \(反之亦然\)。定性结果显示在几个不存在配对训练数据的任务中，包括收集风格转移，对象变形，季节转移，照片增强等。与几种现有方法的定量比较表明了我们方法的优越性。

![](../../.gitbook/assets/image%20%2838%29.png)

## 方法

### 整体框架

![](../../.gitbook/assets/image%20%28133%29.png)

### Adversarial Loss

我们对两个映射函数 $$G、F$$ 都应用了对抗性损失\[16\]。 对于映射函数 $$G：X→Y和$$ 它的判别器 $$D_Y$$ ，我们将目标表达为：

![](../../.gitbook/assets/image%20%2814%29.png)

通过$$\min _{F} \max _{D_{X}} \mathcal{L}_{\text { GAN }}\left(F, D_{X}, Y, X\right)$$ 训练G和D。

### Cycle Consistency Loss

![](../../.gitbook/assets/image%20%2859%29.png)

### Full Objective

![](../../.gitbook/assets/image%20%286%29.png)

整体训练目标可以表示为：

![](../../.gitbook/assets/image%20%2877%29.png)

