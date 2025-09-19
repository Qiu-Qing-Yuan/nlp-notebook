# SkipGram_Negative_Sampling

Pytorch implementation of skipgram model with negative sampling.

https://zhuanlan.zhihu.com/p/672817686
https://www.jianshu.com/p/4517181ca9c3
## Concept of word2vec
Word2Vec 是一种由 Google 在 2013 年提出的革命性的**词嵌入（Word Embedding）**技术，它的核心作用是将自然语言中的词语转换为计算机可以理解和处理的**数值向量**，并且这些向量能够捕捉词语之间的语义和语法关系。

简单来说，Word2Vec 的“用处”可以概括为：**让机器“理解”词语的含义和它们之间的关系。**

以下是 Word2Vec 的主要用途和价值：

### 1. 生成高质量的词向量（Word Vectors）
*   **核心功能**：Word2Vec 将每个词表示为一个固定长度的实数向量（例如 100 维或 300 维）。
*   **语义捕捉**：这些向量不是随机的，而是通过在大量文本上训练得到的。**语义或语法上相似的词，在向量空间中的位置也会很接近**。
    *   例如：“国王”和“王后”的向量距离会很近；“猫”和“狗”也很接近。
    *   更神奇的是，它能捕捉复杂的**类比关系**：
        *   `vec("国王") - vec("男人") + vec("女人") ≈ vec("王后")`
        *   `vec("巴黎") - vec("法国") + vec("意大利") ≈ vec("罗马")`

### 2. 作为下游 NLP 任务的强大基础
Word2Vec 生成的词向量是许多自然语言处理任务的**关键输入特征**，极大地提升了这些任务的性能：
*   **文本分类**：在垃圾邮件识别、新闻分类、情感分析（如判断评论是正面还是负面）中，使用词向量比传统的“词袋模型”（Bag-of-Words）效果好得多，因为它考虑了词义。
*   **机器翻译**：帮助模型理解源语言和目标语言中词语的对应关系。
*   **信息检索**：改进搜索引擎，让搜索结果不仅匹配关键词，还能返回语义相关的文档（例如，搜索“汽车”，也能返回包含“轿车”、“车辆”的相关结果）。
*   **问答系统**：帮助系统理解问题和候选答案之间的语义相似度。
*   **命名实体识别（NER）**：识别文本中的人名、地名、机构名等。

### 3. 发现词语间的语义关系
*   **找近义词/反义词**：给定一个词，可以快速找到向量空间中最接近的词，这些通常是它的近义词。
*   **词义消歧**：虽然 Word2Vec 本身为每个词生成一个向量（无法直接处理一词多义），但其捕捉的语境信息为后续更复杂的模型（如 ELMo, BERT）处理词义消歧奠定了基础。
*   **探索语言模式**：研究人员可以利用词向量来分析语言的演变、文化偏见等。

### 4. 改进传统 NLP 方法
*   在深度学习普及之前，Word2Vec 提供了一种比稀疏的 one-hot 编码或词袋模型高效得多的词表示方法，显著提升了传统机器学习模型（如 SVM、逻辑回归）在文本任务上的表现。

### 总结

**Word2Vec 的最大用处是：它教会了机器用“数值”来理解“词义”，并揭示了词语之间丰富的语义网络。**

你可以把它想象成一个“词语的 GPS 系统”：
*   每个词都有一个精确的“坐标”（向量）。
*   语义相近的词在地图上是邻居。
*   复杂的关系（如“男人之于国王，如同女人之于王后”）可以通过向量的加减运算来表达。

虽然现在有更先进的模型（如 BERT、GPT 等），但 Word2Vec 因其**简单、高效、效果好**的特点，仍然是 NLP 领域的基石技术之一，被广泛应用于工业界和学术界，是理解和处理自然语言不可或缺的工具。


- [https://jalammar.github.io/illustrated-word2vec/](https://jalammar.github.io/illustrated-word2vec/)
- [https://wmathor.com/index.php/archives/1430/](https://wmathor.com/index.php/archives/1430/)

![w2c](../images/w2c.png)

## Supported features

* Skip-gram

* Batch update

* Sub-sampling of frequent word

* Negative Sampling

* GPU support

## Visualization

Save the embedding along with the words to TSV files as shown below, upload these two TSV files to [Embedding Projector](https://projector.tensorflow.org/) for better visualization.

```
def save_embedding(self, outdir, idx2word):
    embeds = self.in_embed.weight.data.cpu().numpy()        
    f1 = open(os.path.join(outdir, 'vec.tsv'), 'w')
    f2 = open(os.path.join(outdir, 'word.tsv'), 'w')        
    for idx in range(len(embeds)):
        word = idx2word[idx]
        embed = '\t'.join([str(x) for x in embeds[idx]])
        f1.write(embed+'\n')
        f2.write(word+'\n')
```
embed = '\t'.join([str(x) for x in embeds[idx]])
embeds[idx]	第 idx 个词的向量，比如 [0.23, -0.45, 0.67, ...]（长度 = embed_size）
[str(x) for x in embeds[idx]]	
把每个浮点数转成字符串，如 ['0.23', '-0.45', '0.67', ...]
'\t'.join(...)	用制表符 \t 连接所有数字，变成一行字符串：<br>"0.23	-0.45	0.67	..."
