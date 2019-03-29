# 梯度下降

## 标准梯度下降

### GD

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

该方法使用动量缓冲梯度，减小随机梯度引起的噪声，并且解决Hessian矩阵病态问题（BGD在收敛过程中和正确梯度相比来回摆动比较大的问题）。

由于当前权值的改变会受到上一次权值改变的影响，类似于小球向下滚动的时候带上了惯性。这样可以加快小球向下滚动的速度。

$$
\left\{\begin{array}{l}{v_{t}=\alpha v_{t-1}+\eta_{t} \Delta J\left(W_{t}, X^{\left(i_{s}\right)}, Y^{\left(i_{s}\right)}\right)} \\ {W_{t+1}=W_{t}-v_{t}}\end{array}\right.
$$

### **NAG** <a id="2-nag"></a>

牛顿加速梯度（Nesterov accelerated gradient）是动量的变种。

Nesterov动量梯度的计算在模型参数施加当前速度之后，因此可以理解为往标准动量中添加了一个校正因子。在Momentun中小球会盲目地跟从下坡的梯度，容易发生错误。所以需要一个更聪明的小球，能提前知道它要去哪里，还要知道走到坡底的时候速度慢下来而不是又冲上另一个坡。

$$
\left\{\begin{array}{l}{v_{t}=\alpha v_{t-1}+\eta_{t} \Delta J\left(W_{t}-\alpha v_{t-1}\right)} \\ {W_{t+1}=W_{t}-v_{t}}\end{array}\right.
$$

## 自适应学习率 <a id="&#x81EA;&#x9002;&#x5E94;&#x5B66;&#x4E60;&#x7387;&#x4F18;&#x5316;&#x7B97;&#x6CD5;"></a>

###  **AdaGrad**

独立地适应所有模型参数的学习率，缩放每个参数反比于其所有梯度历史平均值总和的平方根。具有代价函数最大梯度的参数相应地有个快速下降的学习率，而具有小梯度的参数在学习率上有相对较小的下降。

$$
W_{t+1}=W_{t}-\frac{\eta_{0}}{\sqrt{\sum_{t^{\prime}=1}^{t}\left(g_{t^{\prime}, i}\right)+\epsilon}} \odot g_{t, i}
$$



Adagrad 的主要优势在于不需要人为的调节学习率，它可以自动调节；缺点在于，随着迭代次数增多，学习率会越来越小，最终会趋近于0。

### **RMSProp**

RMSProp算法修改了AdaGrad的梯度积累为指数加权的移动平均，使得其在非凸设定下效果更好。

$$
\left\{\begin{array}{l}{E\left[g^{2}\right]_{t}=\alpha E\left[g^{2}\right]_{t-1}+(1-\alpha) g_{t}^{2}} \\ {W_{t+1}=W_{t}-\frac{\eta_{0}}{\sqrt{E\left[g^{2}\right]_{t}+\epsilon}} \odot g_{t}}\end{array}\right.
$$

RMSProp算法在经验上已经被证明是一种有效且实用的深度神经网络优化算法。目前它是深度学习从业者经常采用的优化方法之一。

### **AdaDelta**

AdaDelta是对AdaGrad的扩展，最初方案依然是对学习率进行自适应约束，但是进行了计算上的简化。 AdaGrad会累加之前所有的梯度平方，而AdaDelta只累加固定大小的项，并且也不直接存储这些项，仅仅是近似计算对应的平均值。即：

$$
\begin{array}{l}{n_{t}=\nu * n_{t-1}+(1-\nu) * g_{t}^{2}} \\ {\Delta \theta_{t}=-\frac{\eta}{\sqrt{n_{t}+\epsilon}} * g_{t}}\end{array}
$$

在此处Ada**D**elta其实还是依赖于全局学习率的，但是作者做了一定处理，经过近似牛顿迭代法之后：

$$
\begin{array}{l}{E\left|g^{2}\right|_{t}=\rho * E\left|g^{2}\right|_{t-1}+(1-\rho) * g_{t}^{2}} \\ {\Delta x_{t}=-\frac{\sqrt{\sum_{r=1}^{t-1} \Delta x_{r}}}{\sqrt{E\left|g^{2}\right|_{t}+\epsilon}}}\end{array}
$$

此时，可以看出AdaDelta已经不用依赖于全局学习率了。

### **Adam**

Adam\(Adaptive Moment Estimation\)本质上是带有动量项的RMSprop，它利用梯度的一阶矩估计和二阶矩估计动态调整每个参数的学习率。Adam的优点主要在于经过偏置校正后，每一次迭代学习率都有个确定范围，使得参数比较平稳。公式如下：

$$
\left\{\begin{array}{l}{m_{t}=\beta_{1} m_{t-1}+\left(1-\beta_{1}\right) g_{t}} \\ {v_{t}=\beta_{2} v_{t-1}+\left(1-\beta_{2}\right) g_{t}^{2}} \\ {\hat{m}_{t}=\frac{m_{t}}{1-\beta_{1}^{t}}, \hat{v}_{t}=\frac{v_{t}}{1-\beta_{2}^{t}}} \\ {W_{t+1}=W_{t}-\frac{\eta}{\sqrt{\hat{v}_{t}}+\epsilon} \hat{m}_{t}}\end{array}\right.
$$

其中， $$m_t$$ 和 $$v_t$$ 分别为一阶动量项和二阶动量项。 $$β1, β2$$ 为动力值大小通常分别取0.9和0.999； $$\hat{m}_{t}, \hat{v}_{t}$$ 分别为各自的修正值。

## 性能比较 <a id="&#x5404;&#x79CD;&#x4F18;&#x5316;&#x5668;&#x7684;&#x6BD4;&#x8F83;"></a>

![](../../.gitbook/assets/20180426113728916.gif)



