# DCGAN

## 介绍

> [UNSUPERVISED REPRESENTATION LEARNING WITH DEEP CONVOLUTIONAL GENERATIVE ADVERSARIAL NETWORKS](https://arxiv.org/pdf/1511.06434.pdf)

近年来，使用卷积网络（CNN）的监督学习在计算机视觉应用中得到了广泛的应用。 相比之下，CNN的无监督学习受到的关注较少。 在这项工作中，我们希望帮助弥合有线电视新闻网在监督学习和无视学习方面的成功之间的差距。 我们介绍了一类称为深度卷积生成对抗网络（DCGAN）的CNN，它们具有一定的架构约束，并证明它们是无监督学习的有力候选者。 在各种图像数据集的训练中，我们展示了令人信服的证据，证明我们的深层卷积对抗性对在生成器和鉴别器中学习了来自对象部分的表示层次结构。 此外，我们将学习的特征用于新任务 - 证明它们作为一般图像表示的适用性。

## 方法

### MODEL ARCHITECTURE

![](../../.gitbook/assets/image%20%28112%29.png)

![](../../.gitbook/assets/image%20%2899%29.png)

### EMPIRICAL VALIDATION OFDCGANS CAPABILITIES

评估无监督表示学习算法质量的一种常见技术是将它们作为监督数据集上的特征提取器来应用，并评估在这些特征之上拟合的线性模型的性能。

![](../../.gitbook/assets/image%20%28104%29.png)

![](../../.gitbook/assets/image%20%2820%29.png)





