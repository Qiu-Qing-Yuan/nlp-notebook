# Pytorch implementation of Glove
GloVe（**Global Vectors for Word Representation**）是斯坦福大学于 2014 年提出的一种**词嵌入（Word Embedding）**技术，和 Word2Vec 齐名，是自然语言处理（NLP）领域中生成词向量的另一种经典且强大的方法。

你可以把 GloVe 看作是 Word2Vec 的“**互补者**”或“**竞争者**”，它们的目标相同——生成能捕捉语义的词向量，但实现思路完全不同。

---

### GloVe 的核心思想：基于全局词共现统计

与 Word2Vec（通过预测上下文来学习）不同，**GloVe 的核心思想是直接分析整个语料库中词语的共现（co-occurrence）统计信息**。

*   **什么是共现？**  
    简单说，就是统计两个词在文本中一起出现的频率。比如，在一个句子 “I love natural language processing” 中，“natural” 和 “language” 就共现了，“love” 和 “processing” 也共现了（虽然语义上不相关，但统计上存在）。
*   **GloVe 做了什么？**  
    它构建一个巨大的“**词-词共现矩阵**”，矩阵中的每个元素 `X_ij` 表示词 `j` 在词 `i` 的上下文窗口中出现的次数。然后，GloVe 设计了一个巧妙的**损失函数**，目标是让两个词的向量差（或向量之间的点积）能够反映它们的共现概率比（ratio of co-occurrence probabilities）。

> **关键洞察**：  
> GloVe 认为，**词语之间的语义关系，可以由它们与其它词语的共现概率比来揭示**。  
> 例如，“ice” 和 “steam” 都与“water”相关，但“ice”更可能和“solid”共现，“steam”更可能和“gas”共现。通过分析这些概率比，模型可以学习到“ice”和“steam”都与“water”相关，但一个代表固态，一个代表气态。

---

### GloVe 的主要用途（和 Word2Vec 类似，但有其特点）

GloVe 生成的词向量同样可以用于各种 NLP 任务，作为基础特征输入：

1.  **文本表示与语义理解**：
    *   将词语转换为稠密向量，替代传统的 one-hot 编码。
    *   语义相似的词在向量空间中距离相近（如 “猫” 和 “狗” 接近，“国王” 和 “王后” 接近）。

2.  **下游 NLP 任务**：
    *   **情感分析**：判断评论是正面还是负面。
    *   **文本分类**：新闻分类、主题识别等。
    *   **机器翻译**：作为翻译模型的输入。
    *   **问答系统**、**信息检索**：提升语义匹配能力。

3.  **语义关系计算**：
    *   支持类比推理（`国王 - 男人 + 女人 ≈ 王后`）。
    *   找近义词、反义词。

---

### GloVe vs. Word2Vec：主要区别

| 特性 | GloVe | Word2Vec |
| :--- | :--- | :--- |
| **学习方式** | **全局统计**：基于整个语料的共现矩阵，进行矩阵分解式的优化。 | **局部上下文预测**：通过滑动窗口，预测目标词的上下文（Skip-gram）或根据上下文预测目标词（CBOW）。 |
| **训练速度** | 通常更快，因为可以利用全局统计并行计算。 | 相对较慢，尤其是 Skip-gram 模型。 |
| **数据利用** | 显式利用了**所有词对的共现信息**，对低频词可能更友好。 | 主要利用局部上下文窗口内的信息。 |
| **向量质量** | 在某些语义任务（如同义词识别）上表现优异，尤其在大规模语料上。 | 在句法任务（如词性标注）上有时表现更好。 |
| **可解释性** | 基于统计的数学模型，理论清晰。 | 基于神经网络预测，更像“黑箱”。 |

---

### 总结

**GloVe 的用处是：通过分析词语在整个语料库中的共现统计规律，生成高质量的词向量，从而让机器理解词语的语义和关系。**

它的优势在于：
*   **理论清晰**：直接基于词共现统计。
*   **高效**：利用全局信息，训练相对高效。
*   **效果好**：在许多任务上与 Word2Vec 相当甚至更优，尤其在大规模语料上。

虽然现在有 BERT、RoBERTa 等更先进的上下文相关（Contextual）词向量模型，但 GloVe 作为一种**静态词向量**（每个词只有一个固定向量），因其**简单、高效、开源且效果出色**，至今仍被广泛用于各种 NLP 应用中，是学习和实践 NLP 不可或缺的工具之一。

## 原理

- [https://blog.csdn.net/coderTC/article/details/73864097](https://blog.csdn.net/coderTC/article/details/73864097)
- [https://juejin.cn/post/6844903923279642638](https://juejin.cn/post/6844903923279642638)
- [https://nlpython.com/implementing-glove-model-with-pytorch/](https://nlpython.com/implementing-glove-model-with-pytorch/)

![glove1](../images/glove1.png)
![glove2](../images/glove2.png)
![glove3](../images/glove3.png)
![glove4](../images/glove4.png)
![glove5](../images/glove5.png)
![glove6](../images/glove6.png)
![glove7](../images/glove7.png)

### 附SVD及应用
SVD分解将任意矩阵分解成一个正交矩阵和一个对角矩阵以及另一个正交矩阵的乘积。
![glove8](../images/svd1.png)
![glove9](../images/svd2.png)
![glove10](../images/svd3.png)

## 可视化

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
