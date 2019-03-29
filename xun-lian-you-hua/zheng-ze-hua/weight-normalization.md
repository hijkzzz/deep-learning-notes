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

这表明权重归一化实现了两件事:它将权重梯度按 $$g/||v||$$ 缩放，并将梯度从当前权重向量用 $$M_w$$ 投影出去。这两种效应都有助于使梯度的协方差矩阵更接近identity和利于优化，正如我们在下面解释的那样。

由于投射远离 $$w$$ ，当使用标准梯度下降没有动量学习具有权重归一化的神经网络时， $$v$$ 的范数随着权重更新次数单调增长： $$Let \ \mathbf{v}^{\prime}=\mathbf{v}+\Delta \mathbf{v}$$且 $$\Delta \mathbf{v} \propto \nabla_{\mathbf{v}} L$$ ， $$\Delta \mathbf{v}$$ 一定正交于当前权重向量 $$w$$ ，因为我们在计算 $$\nabla_{\mathbf{v}} L$$ 时投影远离它。又 $$v$$与 $$w$$ 成比例，因此其与 $$v$$ 也是正交的，由于毕达哥拉斯定理，并且增加了它的范数。具体来说： $$\left\|\mathbf{v}^{\prime}\right\|=\sqrt{\|\mathbf{v}\|^{2}+c^{2}\|\mathbf{v}\|^{2}}=\sqrt{1+c^{2}\|\mathbf{v}\|} \geq\|\mathbf{v}\|$$ ，其中 $$\|\Delta \mathbf{v}\| /\|\mathbf{v}\|=c$$ 。因为范数增大 $$\frac{g}{\|\mathbf{v}\|}$$ 变小，会使得梯度变小，网络的学习会更稳定。对于像Adam这样对单个参数使用单独学习速率的优化器，此属性并不严格。我们在实验中使用，或在使用动量时使用。 但是，从质量上来说，我们仍然会发现同样的效果。

经验上，我们发现增长范数\| \| v \| \|的能力使得具有权重归一化的神经网络的优化对学习率的值非常鲁棒:如果学习率很大，则非归一化权重的范数快速增长，直到达到适当的有效学习率。一旦权重的范数相对于更新的范数变大，有效学习率稳定。因此，与使用正常参数化时相比，具有权重归一化的神经网络在更宽的学习率范围内工作良好。已经观察到，具有批量归一化的神经网络也具有\[特性，这也可以通过该分析来解释。

Weight Normalization与BNN也有一定的相关性。对于我们的网络只有一层的特殊情况，该层的输入特征被白化\(以零均值和单位方差独立分布\)，这些统计数据由 $$μ[ t ] = 0$$ 和 $$σ[t] =||v||$$ 。在这种情况下，使用批量标准化来标准化预激活等同于使用权重标准化来标准化权重。

### Data-Dependent Initialization of Parameters

除了重新参数化效果之外，批量标准化还具有固定由神经网络的每个层生成的特征的规模的益处。 这使得优化对于参数初始化是稳健的，对于这些参数初始化，这些尺度在各层之间变化 由于权重归一化缺乏这种属性，我们发现正确初始化参数很重要。我们建议从具有固定尺度的简单分布中对 $$v$$ 元素进行采样，这在我们的实验中是平均零和标准差0.05的正态分布。在开始培训之前，我们然后初始化 $$b$$ 和 $$g$$ 参数来修复我们网络中所有预激活的小批量统计信息，就像批量规范化一样，但仅限于单个小批量数据并且仅在初始化期间。这可以通过对单个数据小批量执行通过我们网络的初始前馈来有效地完成，在每个神经元处使用以下计算：

$$
t=\frac{\mathbf{v} \cdot \mathbf{x}}{\|\mathbf{v}\|}, \quad \text { and } \quad y=\phi\left(\frac{t-\mu[t]}{\sigma[t]}\right)
$$

然后我们可以初始化神经元的biase 和 scale 为：

$$
g \leftarrow \frac{1}{\sigma[t]}, \quad \quad b \leftarrow \frac{-\mu[t]}{\sigma[t]}
$$

像批量标准化一样，该方法确保在应用非线性之前，所有特征最初具有零均值和单位方差。对于我们的方法，这仅适用于我们用于初始化的小批次，随后的小批次可能具有稍微不同的统计数据，但是通过实验，我们发现这种初始化方法工作良好。

### Mean-only Batch Normalization













