# Tiled Matrix Multiplication

## 介绍

> [Tiled Matrix Multiplication](https://www.fz-juelich.de/SharedDocs/Downloads/IAS/JSC/EN/slides/cuda/10-cuda-dgemm-tiled.pdf?__blob=publicationFile)

本文讨论CUDA实现高效的矩阵乘法。如图，对于普通的矩阵乘法，每次运算需要传递 $$2 * n$$个 $$tile$$ ，如果矩阵的规模非常大，这将会导致数据传递、内存开销很大。

![](../../.gitbook/assets/image%20%28189%29.png)

为了解决这个问题，我们需要调整矩阵乘法的计算顺序。如下

## 方法

### 图示

每次只传输3个 $$tile$$ ，其中 $$c' = a * b + c$$ ，每个$$tile$$需与 $$n$$ 个 $$tile$$ 相乘：

![](../../.gitbook/assets/image%20%28205%29.png)

![](../../.gitbook/assets/image%20%28179%29.png)

![](../../.gitbook/assets/image%20%2898%29.png)

### 循环顺序

![](../../.gitbook/assets/image%20%2834%29.png)

### 实现

* 对于每个步骤，设备上只需要有3个tile
* 使用固定内存进行切片可以执行异步主机到设备副本并加快数据传输速度
* 在cublasDgemm调用中将beta设置为1以重用以前的计算结果

![](../../.gitbook/assets/image%20%288%29.png)

### 工作过程

![](../../.gitbook/assets/image%20%281%29.png)

### 多Streams

我们可以用多个CUDA Streams同时计算多个 $$tile$$ 组：

![](../../.gitbook/assets/image%20%2862%29.png)

Steams的工作原理和CPU流水线类似

![](../../.gitbook/assets/image%20%2830%29.png)









