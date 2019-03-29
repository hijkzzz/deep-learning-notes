# L-BFGS

### 牛顿法

设有损失函数 $$f(x)$$ 

$$
f(x) \approx \phi(x)=f\left(x^{(k)}\right)+\nabla f\left(x^{(k)}\right)^{T}\left(x-x^{(k)}\right)+\frac{1}{2}\left(x-x^{(k)}\right)^{T} \nabla^{2} f\left(x^{(k)}\right)\left(x-x^{(k)}\right)
$$

其中 $$\nabla^{2} f\left(x^{(k)}\right)$$ 是二阶的海森矩阵，为了使函数最小化，对后面两项求导：

$$
\nabla f\left(x^{(k)}\right)+\nabla^{2} f\left(x^{(k)}\right)\left(x-x^{(k)}\right)=0
$$

得到牛顿迭代法的公式：

$$
x^{(k+1)}=x^{(k)}-\nabla^{2} f\left(x^{(k)}\right)^{-1} \nabla f\left(x^{(k)}\right)
$$

### 拟牛顿法

前面介绍了牛顿法，它的突出优点是收敛很快，但是运用牛顿法需要计算二阶偏导数，而且目标函数的Hesse矩阵可能非正定。为了克服牛顿法的缺点，人们提出了拟牛顿法，它的基本思想是用不包含二阶导数的矩阵近似牛顿法中的Hesse矩阵的逆矩阵。由于构造近似矩阵的方法不同，因而出现不同的拟牛顿法。

### L-BFGS

Limited-memory BFGS是一种节省内存的拟牛顿法。





