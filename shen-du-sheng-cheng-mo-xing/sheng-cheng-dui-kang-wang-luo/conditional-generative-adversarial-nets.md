# Conditional Generative Adversarial Nets

## 介绍

> [Conditional Generative Adversarial Nets](https://arxiv.org/abs/1411.1784)

生成对抗网\[ 8 \]最近被引入作为一种训练生成模型的新方法。在这项工作中，我们介绍了生成对抗网的条件版本，它可以通过简单地输入数据来构造，y，我们希望对生成器和鉴别器进行条件化。我们证明了该模型可以生成以类标签为条件的最大位数。我们还举例说明了该模型如何用于学习多模态模型，并提供了图像标签应用的初步示例，其中我们演示了该方法如何生成不属于训练标签的描述性标签。

## 方法

### Generative Adversarial Nets

$$
\min _{G} \max _{D} V(D, G)=\mathbb{E}_{\boldsymbol{x} \sim p_{\text { data }}(\boldsymbol{x})}[\log D(\boldsymbol{x})]+\mathbb{E}_{\boldsymbol{z} \sim p_{z}(\boldsymbol{z})}[\log (1-D(G(\boldsymbol{z})))]
$$

上式中， $$G$$ 为生成器网络， $$D$$ 为判别器网络。生成器的作用是从随机噪音生成数据，判别器的作用是学会判断输入数据的真假性。内层的 $$max$$ 表示判别器网络的损失函数：极大化真实数据的似然函数+极小化生成器输出数据的似然函数；外层 $$max$$ 表示生成器网络的损失函数：极大化生成器输出数据的似然函数。简单的说，判别器网络认为真实数据为真，生成数据为假，生成器网络改进自身让判别器网络认为自己的输出为真。

### Conditional Adversarial Nets

$$
\min _{G} \max _{D} V(D, G)=\mathbb{E}_{\boldsymbol{x} \sim p_{\operatorname{dat}}(\boldsymbol{x})}[\log D(\boldsymbol{x} | \boldsymbol{y})]+\mathbb{E}_{\boldsymbol{z} \sim p_{\boldsymbol{z}}(\boldsymbol{z})}[\log (1-D(G(\boldsymbol{z} | \boldsymbol{y})))]
$$

条件生成对抗网络即使数据分布服从条件概率。这样做的意义是：比如我们要生成0-9十个数字的图像，此时只要指定条件值为0-9就能得到相对应的图像。在原始的生成对抗网络中，生成器输出的数据仅仅近似真实分布，而不能决定确定类型。

![](../../.gitbook/assets/image%20%2872%29.png)

![](../../.gitbook/assets/image%20%2845%29.png)

