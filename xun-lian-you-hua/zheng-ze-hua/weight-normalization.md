# Weight Normalization

## 介绍

> [Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks](https://arxiv.org/pdf/1602.07868.pdf)

我们提出了权重归一化:神经网络中权重向量的重新参数化，将权重向量的长度与其方向分离。通过用这种方法重新定义权重，我们改进了优化问题的条件，加快了随机梯度下降的收敛速度。我们的重新参数化是受批处理规范化的启发，但并不在小批处理中的示例之间产生任何依赖关系。这意味着，我们的方法也可以成功地应用于递归模型，如LSTM，以及噪声敏感的应用，如深度强化学习或生成模型，批处理规范化不太适合这些应用。虽然我们的方法更简单，但它仍然提供了的full batch normalization的加速。此外，我们方法的计算开销较低，允许在相同的时间内采取更多的优化步骤。我们演示了我们的方法在监督图像识别、生成模型和深度强化学习中的应用。

## 方法

我们考虑标准的人工神经网络，其中每个神经元的计算包括摄入输入特征的加权和，然后是元素非线性：

$$
y=\phi(\mathbf{w} \cdot \mathbf{x}+b)
$$

在将损失函数与一个或多个神经元输出相关联之后，这种神经网络通常通过每个神经元的参数w，b中的随机梯度下降来训练。为了加速这一优化过程的收敛，我们建议对参数向量和标量参数的每个加权向量重新参数化，并改为对这些参数执行随机梯度描述。我们通过使用表达新参数的权重向量来实现：

$$
\mathbf{w}=\frac{g}{\|\mathbf{v}\|} \mathbf{v}
$$

$$
\mathbf{v} \text { is a } k \text { -dimensional vector, } g \text { is a scalar, and }\|\mathbf{v}\| \text { denotes the Euclidean norm of } \mathbf{v}
$$

这种重新参数化的效果是：固定了权重向量的欧几里德范数。我们现在有 $$||w||=g$$ ，且独立于参数 $$v$$ 。因此，我们称之为reparameterizaton weight normalization。

我们也可以对标量使用指数参数化 $$g=e^{s}$$ ，而不是直接使用。

### Gradients

训练的时候，我们直接对 $$v、g$$ 使用标准梯度下降，其中梯度为：

$$
\nabla_{g} L=\frac{\nabla_{\mathbf{w}} L \cdot \mathbf{v}}{\|\mathbf{v}\|}, \quad \nabla_{\mathbf{v}} L=\frac{g}{\|\mathbf{v}\|} \nabla_{\mathbf{w}} L-\frac{g \nabla_{g} L}{\|\mathbf{v}\|^{2}} \mathbf{v}
$$

另一种梯度的方法表达是

$$
\nabla_{\mathbf{v}} L=\frac{g}{\|\mathbf{v}\|} M_{\mathbf{w}} \nabla_{\mathbf{w}} L, \quad \text { with } \quad \mathbf{M}_{\mathbf{w}}=\mathbf{I}-\frac{\mathbf{w} \mathbf{w}^{\prime}}{\|\mathbf{w}\|^{2}}
$$

这表明权重归一化实现了两件事:它将权重梯度按 $$g/||v||$$ 缩放，并将梯度从当前权重向量投影出去。这两种效应都有助于使梯度的协方差矩阵更接近身份和利益优化，正如我们在下面解释的那样。



