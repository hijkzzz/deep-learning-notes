# 使用Numpy实现卷积神经网络（推导前馈和反向传播）

> [Only Numpy: Implementing Convolutional Neural Network using Numpy \( Deriving Forward Feed and Back Propagation \) with interactive code](https://becominghuman.ai/only-numpy-implementing-convolutional-neural-network-using-numpy-deriving-forward-feed-and-back-458a5250d6e4)

卷积神经网络（CNN）很多人都听说过它的名字，我想知道它是前馈过程以及反向传播过程。 由于我只专注于神经网络部分，我不会解释卷积操作是什么，如果你不知道这个操作请从songho阅读这个“[2D卷积的例子](http://www.songho.ca/dsp/convolution/convolution2d_example.html)”.。

### 初始化权重，声明超参数和训练数据

![](../../.gitbook/assets/image%20%2840%29.png)

让我们保持简单，我们有四张（3 \* 3）图像。 （LOL太小了，不能称它们为图像，但它会完成这项工作）。 正如您在真实标签数据（Y）中所看到的，如果图像有更多1，则结果输出会增加。 由于我们使用logistic sigmoid函数作为最终输出，因此max设置为1.1。

### 网络架构

![](../../.gitbook/assets/image%20%28155%29.png)

如上所述，我们有一个非常简单的网络结构

* X → 3\*3 图像
* K →卷积操作（右边是矩阵形式，左边是矢量化形式）
* 绿色开始→结果图像\(右矩阵形式，左是矢量化形式\)

如果上面的图像让您感到困惑，请参阅下图。

![](../../.gitbook/assets/image%20%2865%29.png)

基本上，3×3像素卷积操作可以被认为是将具有给定权重的位于不同图像中的某些像素相乘。

### 前馈

![](../../.gitbook/assets/image%20%28174%29.png)

当我们在线性行中写卷积运算时，我们可以像上面一样表达每个节点。但是请注意橙色方框，它将L1表示为\[ 1 \* 4 \]向量。另外，请查看下面红色方框中的每个变量代表什么。

![](../../.gitbook/assets/image%20%28188%29.png)

如上所示，每个节点表示卷积运算的结果图像。

###  **W2 的反向传播**

![](../../.gitbook/assets/image%20%28167%29.png)

\*\*\*\*

标准SGD反向传播，上述操作没有什么特别之处**。**

###  **W\(1,1\) 的反向传播**

![](../../.gitbook/assets/image%20%28190%29.png)

有很多事情要发生，我将从最简单的一个开始

* 橙色盒子/橙色星星→我没有足够的空间来写所有的tanh \( \)的导数，所以每个“dL”符号代表对tanh \( \)的导数。
* 蓝色方框→同样没有足够的空间写下等式，无论向量之间的点积多么简单。
* 绿色方框星python代码实现中关于W\(1，1 \)的导数的第一部分，如下所示。

![](../../.gitbook/assets/image%20%28151%29.png)

如上所示，我们转置W2，因此尺寸从（1,4）变为（4,1）。 我们将使用符号'g'来表示操作的结果。

* 绿盒星2→变量g和导数D1数组之间的点积，因此维数保持为\( 1，4 \)。

![](../../.gitbook/assets/image%20%28181%29.png)

![](../../.gitbook/assets/image%20%2820%29.png)

###  所有权重反向传播

![](../../.gitbook/assets/image%20%2822%29.png)

我跳过了导数符号，但写下了导数所需的实际变量。另外，请注意变量“g”代表青色\(浅绿色\)框中的变量。

仔细看看导数的所有方程，注意到什么了吗？\(特别是仔细观察变量X \)。这是卷积运算。我是什么意思？见下文。

![](../../.gitbook/assets/image%20%283%29.png)

整个导数可以如上所述写入，即输入图像和第1层中所有节点的导数之间的卷积运算。在python代码中，我们可以像下面这样实现它

![](../../.gitbook/assets/image%20%2842%29.png)

这里有两点需要注意。

* 1→grad  _1_  part  _1_ 整形:将向量整形为\( 2 \* 2 \)图像
* 2→突出显示的部分正在旋转'grad\_1\_temp\_1'变量......为什么......？

让我们仔细看看核，如下所示

![](../../.gitbook/assets/image%20%28186%29.png)

如上所示，矩阵的形式如下：

 — — — — — — — — — — — —   
\| gdL1\(_2,2_\) \| gdL1\(_2,1_\) \|  
 — — — — — — — — — —— —   
\| gdL1\(_1,2_\) \| gdL1\(_1,1_\) \|  
 — — — — — — — — — —— — 

然而，我们的向量有这个顺序的变量

\[ gdL1\(1,1\), gdL1\(1,2\), gdL1\(2,1\), gdL1\(2,2\) \]

所以当我们把上面的向量转换成矩阵时，它会如下所示。

 — — — — — — — — — — — —   
\| gdL1\(**1,1**\) \| gdL1\(**1,2**\) \|  
 — — — — — — — — — —— —   
\| gdL1\(**2,1**\) \| gdL1\(**2,2**\) \|  
 — — — — — — — — — —— — 

这就是我们在卷积运算之前旋转矩阵的原因

### 其它

所以这个理论是正确的，但是有一个重要的方面我还没有提到。

![](../../.gitbook/assets/image%20%2867%29.png)

如红框中所示，计算的梯度大约为

![](../../.gitbook/assets/image%20%2869%29.png)

所以当更新权重时，我们需要再次转置计算的梯度，我在代码中没有这样做

![](../../.gitbook/assets/image%20%2813%29.png)

![](../../.gitbook/assets/image%20%28166%29.png)

### 实现

```python
import numpy as np
from scipy import signal

def tanh(x):
    return np.tanh(x)
def d_tanh(x):
    return 1 - np.tanh(x) ** 2

def log(x):
    return 1/(1 + np.exp(-1*x))
def d_log(x):
    return log(x) * ( 1 - log(x) )

np.random.seed(598765)

x1 = np.array([[0,0,0],[0,0,0],[0,0,0]])
x2 = np.array([[1,1,1],[0,0,0],[0,0,0]])
x3 = np.array([[0,0,0],[1,1,1],[1,1,1]])
x4 = np.array([ [1,1,1],[1,1,1],[1,1,1]])
X = [x1,x2,x3,x4]
Y = np.array([
    [0.53],
    [0.77],
    [0.88],
    [1.1]
])

# 0. Declare Weights
w1 = np.random.randn(2,2) * 4 
w2 = np.random.randn(4,1) * 4

# 1. Declare hyper Parameters
num_epoch = 1000
learning_rate = 0.7

cost_before_train = 0
cost_after_train = 0
final_out,start_out =np.array([[]]),np.array([[]])

# ---- Cost before training ------
for i in range(len(X)):
    
    layer_1 = signal.convolve2d(X[i],w1,'valid')
    layer_1_act = tanh(layer_1)

    layer_1_act_vec = np.expand_dims(np.reshape(layer_1_act,-1),axis=0)
    layer_2 = layer_1_act_vec.dot(w2)
    layer_2_act = log(layer_2)   
    cost = np.square(layer_2_act- Y[i]).sum() * 0.5
    cost_before_train = cost_before_train + cost
    start_out = np.append(start_out,layer_2_act)
    
# ----- TRAINING -------
for iter in range(num_epoch):
    
    for i in range(len(X)):
    
        layer_1 = signal.convolve2d(X[i],w1,'valid')
        layer_1_act = tanh(layer_1)

        layer_1_act_vec = np.expand_dims(np.reshape(layer_1_act,-1),axis=0)
        layer_2 = layer_1_act_vec.dot(w2)
        layer_2_act = log(layer_2)

        cost = np.square(layer_2_act- Y[i]).sum() * 0.5
        #print("Current iter : ",iter , " Current train: ",i, " Current cost: ",cost,end="\r")

        grad_2_part_1 = layer_2_act- Y[i]
        grad_2_part_2 = d_log(layer_2)
        grad_2_part_3 = layer_1_act_vec
        grad_2 =   grad_2_part_3.T.dot(grad_2_part_1*grad_2_part_2)      

        grad_1_part_1 = (grad_2_part_1*grad_2_part_2).dot(w2.T)
        grad_1_part_2 = d_tanh(layer_1)
        grad_1_part_3 = X[i]

        grad_1_part_1_reshape = np.reshape(grad_1_part_1,(2,2))
        grad_1_temp_1 = grad_1_part_1_reshape * grad_1_part_2
        grad_1 = np.rot90(
          signal.convolve2d(grad_1_part_3, np.rot90(grad_1_temp_1, 2),'valid'),
          2)

        w2 = w2 - grad_2 * learning_rate
        w1 = w1 - grad_1 * learning_rate
        
# ---- Cost after training ------
for i in range(len(X)):
    
    layer_1 = signal.convolve2d(X[i],w1,'valid')
    layer_1_act = tanh(layer_1)

    layer_1_act_vec = np.expand_dims(np.reshape(layer_1_act,-1),axis=0)
    layer_2 = layer_1_act_vec.dot(w2)
    layer_2_act = log(layer_2)   
    cost = np.square(layer_2_act- Y[i]).sum() * 0.5
    cost_after_train = cost_after_train + cost
    final_out = np.append(final_out,layer_2_act)

    
# ----- Print Results ---
print("\nW1 :",w1, "\n\nw2 :", w2)
print("----------------")
print("Cost before Training: ",cost_before_train)
print("Cost after Training: ",cost_after_train)
print("----------------")
print("Start Out put : ", start_out)
print("Final Out put : ", final_out)
print("Ground Truth  : ", Y.T)




# -- end code --
```

