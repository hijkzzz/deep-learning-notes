# 正则化/归一化

### 正则化

正则化的用于防止神经网络网络在小训练集上过拟合，主要的正则化方法有：

* L1让权重趋向于稀疏解，可用于压缩模型大小
* L2让权重趋向于较小的值，比L1更容易求解
* Dropout随机关闭神经元的输出，用类似随机森林的思想防止过拟合
* 下面的归一化方法也具有一定的正则化效果

### 归一化

归一化层用于加速深层神经网络的训练，主要的归一化方法有：

将输入的图像张量形状记为 $$[N, C, H, W]$$ （依次为样本、通道、高、宽）

* BatchNormalization在维度$$[N,H,W]$$ 上对每个输出特征图进行归一化，小的BatchSize效果不好
* LayerNormalization不涉及多个样本，在维度 $$[C, H, W]$$ 上对每一层的输出归一化，主要对RNN作用明显
* Weight Normalization对权重本身进行归一化，可用于噪音敏感的任务，如强化学习
* InstanceNormalization不涉及多个样本，在维度 $$[H, W]$$ 上对输出特征图做归一化，可用于神经风格迁移
* GroupNormalization不涉及多个样本，将特征图分组，在维度 $$ $$ $$[C/G, H, W]$$ 上进行归一化
* SwitchableNorm是将BatchNormalization、LayerNormalization、InstanceNormalization

  方法结合，赋予权重，让网络自己去学习归一化层应该使用什么方法



