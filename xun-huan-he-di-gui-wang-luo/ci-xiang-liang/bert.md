# BERT

## 介绍

> [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)

我们引入了一种名为BER的新语言表示模型，它代表来自变换器的双向编码器表示。与最近的语言表示模型\(Peters et al., 2018; Radfordet al., 2018）不同，BERT旨在通过在所有层中的左右上下文中联合调节来预先训练双向表示。因此，预训练的BERT表示可以通过一个附加输出层进行微调，以创建适用于各种任务的最先进模型，例如问答和语言推断，而无需特定任务的特定结构修改。

BERT在概念上简单且经验丰富。 它获得了11项自然语言处理任务的最新成果，包括将GLUE基准推向80.4％（绝对改进率为7.6％）。MultiNLI精度达到86.7％（绝对改进5.6％）和SQuAD v1.1问题回答测试F1至93.2（1.5绝对改善），优于人类表现2.0。

我们的主要贡献是：

* 我们证明了双向预训练对于语言表达的重要性。
* 我们表明，预先训练的表示能够满足许多经过大量工程设计的特定于任务的体系结构的需求
* BERT推进了11种NLP任务的最新水准。

## 方法

### Model Architecture

BERT的模型架构是一个多层双向Transformer编码器，基于Vaswani等人（2017）中原始描述的实现，并在tensor2tensor library中发布。

选择 $$BERT_{BASE }$$ 与OpenAI GPT具有相同的模型大小以进行比较。至关重要的是，BERT Transformer使用双向自我关注，而GPT Transformer使用受限制的自我关注，而且只有左侧的上下文才能使用。

![](../../.gitbook/assets/image%20%28197%29.png)

### Input Representation

我们的输入表示能够在单词序列中明确地表示单个文本句子或一对文本句子（例如，\[问题，答案\]）。对于给定的token，其输入代表是通过对相应的token，segment和位置嵌入进行求和来构造的。

* 我们使用WordPiece嵌入（Wu et al., 2016）和30,000个token词汇表。 用\#\#分词。
* 我们使用学习的位置嵌入，支持的序列长度最多为512个token。
* 每个序列的第一个标记都是特殊分类嵌入（\[CLS\]）。 对应于该token的最终隐藏状态（即，Transformer的输出）被用作分类任务的聚合序列表示。 对于非分类任务，将忽略此向量。
* 句子对被打包成单个序列。我们用两种方式区分句子。首先，我们用一个特殊的记号\(\[SEP\)把它们分开。第二，我们在第一个句子的每一个token上加上一个可学习的sentence Aembedding，在第二个句子的每一个token上加上一个可学习的sentence B embedding。
* 对于单句输入，我们只使用sentence A embeddings

![](../../.gitbook/assets/image%20%28140%29.png)

### Pre-training Tasks

与Peters等人不同。 （2018年）和Radford等人（2018年），我们不使用传统的从左到右或从右到左的语言模型预训练BERT。相反，我们使用两个新的非预测预测任务预训练BERT。 

#### Task \#1: Masked LM

为了训练深度双向表示，我们采用直接的方法随机屏蔽一定比例的输入token，然后仅预测那些被屏蔽的token。

![](../../.gitbook/assets/image%20%28141%29.png)

#### Task \#2: Next Sentence Prediction

许多重要的下游任务，例如问答（QA）和自然语言推理（NLI），都是基于对两个文本句子之间的理解的理解，而这两个文本句子并不直接通过语言建模来捕获。为了训练一个理解句子关系的模型，我们预先训练一个二进制的下一个句子预测任务，这个预测任务可以从任何单语语料库中简单地生成。

![](../../.gitbook/assets/image%20%28150%29.png)

## 实验

![](../../.gitbook/assets/image%20%28127%29.png)

