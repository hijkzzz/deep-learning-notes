# 优化器

## 标准梯度下降

### SGD

假设要学习训练的模型参数为 $$W$$ ，代价函数为 $$J(W)$$ ，则代价函数关于模型参数的偏导数即相关梯度为 $$ΔJ(W)$$ ，学习率为 $$η_t$$ ，则使用梯度下降法更新参数为：

$$
W_{t+1}=W_{t}-\eta_{t} \Delta J\left(W_{t}\right)
$$

### BGD

其中可以使用批量样本来估计梯度，提升梯度估计的准确性

$$
W_{t+1}=W_{t}-\eta_{t} \sum_{i=1}^{n} \Delta J_{i}\left(W_{t}, X^{(i)}, Y^{(i)}\right)
$$

## 动量

### **Momentum** <a id="1-momentum"></a>

该方法使用动量缓冲梯度，减小随机梯度引起的噪声，并且解决Hessian矩阵病态问题（SGD在收敛过程中和正确梯度相比来回摆动比较大的问题）

由于当前权值的改变会受到上一次权值改变的影响，类似于小球向下滚动的时候带上了惯性。这样可以加快小球向下滚动的速度。

$$
\left\{\begin{array}{l}{v_{t}=\alpha v_{t-1}+\eta_{t} \Delta J\left(W_{t}, X^{\left(i_{s}\right)}, Y^{\left(i_{s}\right)}\right)} \\ {W_{t+1}=W_{t}-v_{t}}\end{array}\right.
$$

## 自适应学习率 <a id="&#x81EA;&#x9002;&#x5E94;&#x5B66;&#x4E60;&#x7387;&#x4F18;&#x5316;&#x7B97;&#x6CD5;"></a>

###  **AdaGrad**

###  **RMSProp**

###  **Adam**

###  **AdaDelta**

## 各种优化器的比较 <a id="&#x5404;&#x79CD;&#x4F18;&#x5316;&#x5668;&#x7684;&#x6BD4;&#x8F83;"></a>

