# 归一化/正则化

### 归一化

归一化层用于加速深层神经网络的训练，目前主要有这几个方法：

* Batch Normalization（2015年）
* Layer Normalization（2016年）
* Weight Normalization（2016年）
* Instance Normalization（2017年）
* Group Normalization（2018年）
* Switchable Normalization（2018年）

将输入的图像张量形状记为 $$[N, C, H, W]$$ ，这几个方法主要的区别就是在：

* BatchNormalization 是在mini-batch上，对NHW做归一化，对小的batch-size效果不好
* LayerNormalization在通道方向上，对CHW归一化，主要对RNN作用明显
* Weight Normalization对权重进行规范化，可用于噪音敏感的任务，如强化学习
* InstanceNormalization在图像像素上，对HW做归一化，可用于神经风格迁移
* GroupNorm将Channel分组，然后再做归一化
* SwitchableNorm是将BatchNormalization、LayerNormalization、InstanceNormalization

  方法结合，赋予权重，让网络自己去学习归一化层应该使用什么方法

### 正则化

正则化的主要作用是防止网络过拟合，方法主要有L1、L2、Dropout以及上面提到的的归一化

* L1让权重趋向于稀疏解
* L2让权重的值更小
* Dropout随机drop神经元的输出，增强神经网络抗干扰性
* BN归一化也具有一定的正则化效果

