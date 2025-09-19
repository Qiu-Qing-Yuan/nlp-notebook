# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn # 神经网络模块
import torch.nn.functional as F # 函数式接口

class EmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(EmbeddingModel, self).__init__()
        
        self.vocab_size = vocab_size # 词汇表大小
        self.embed_size = embed_size # 嵌入维度

        # shape = [vocab_size.embed_size]
        # ① 输入嵌入层——中心词（input word）
        # 返回一个可学习的嵌入矩阵，形状为[vocab_size,embed_size]
        # 即 词的向量字典，每个词（索引表示）对应一个 embed_size 维的向量
        self.in_embed = nn.Embedding(self.vocab_size, self.embed_size)  # (10000, 200)
        # ② 输出嵌入层——上下文词（context word）
        self.out_embed = nn.Embedding(self.vocab_size, self.embed_size)
        self.init_embed()

    # 自定义的权重初始化函数
    def init_embed(self):
        init = 0.5 / self.embed_size
        self.in_embed.weight.data.uniform_(-init, init)  # 从[-a,a]之间均匀采样
        self.out_embed.weight.data.uniform_(-init, init)
        # self.out_embed.weight.data.uniform_(-0, 0)

    # 前向传播
    def forward(self, input_labels, pos_labels, neg_labels):
        ''' input_labels: center words, [batch_size] which is one dimentional vector of batch size
            pos_labels: positive words, [batch_size, (window_size * 2)]
            neg_labels：negative words, [batch_size, K]            
            return: loss, [batch_size]
        '''
        # 查找中心词向量：（512,200）
        input_embedding = self.in_embed(input_labels)# [batch_size, embed_size]
        # 查找正样本上下文向量：（512,4,200）   因为 pos_labels 是 2D，输出是 3D：[B, P, embed_size]
        pos_embedding = self.out_embed(pos_labels)# [batch_size, (window * 2), embed_size]
        # 查找负样本上下文向量：(512,15,200)
        neg_embedding = self.out_embed(neg_labels)# [batch_size,  K, embed_size]
        # （512, 200） -> (512,200,1)
        input_embedding = input_embedding.unsqueeze(2)# [batch_size, embed_size, 1]

        # 批量矩阵乘法 （512，4，1）
        pos_dot = torch.bmm(pos_embedding, input_embedding)# [batch_size, (window * 2), 1]
        # （512,4,1） -> (512,4)
        # [batch_size, (window * 2)] 最后一个维度是 1，表示每个点积的结果是一个标量，但还“包”着一个维度。移除指定维度中大小为 1 的维度
        pos_dot = pos_dot.squeeze(2)
        
        neg_dot = torch.bmm(neg_embedding, -input_embedding)# [batch_size, K, 1]
        neg_dot = neg_dot.squeeze(2)# [batch_size, K]
        
        log_pos = F.logsigmoid(pos_dot).sum(1)# [batch_size]  .sum(1) 是按 第二维（即上下文词维度） 求和。
        log_neg = F.logsigmoid(neg_dot).sum(1)# [batch_size]
        
        return -log_pos-log_neg # [batch_size]

    def save_embedding(self, outdir, idx2word):
        # 如果模型在 GPU 上训练，需要把张量移到 CPU 上，否则无法转为 NumPy
        # [vocab_size, embed_size]
        embeds = self.in_embed.weight.data.cpu().numpy()        
        f1 = open(os.path.join(outdir, 'vec.tsv'), 'w')
        f2 = open(os.path.join(outdir, 'word.tsv'), 'w')        
        for idx in range(len(embeds)):
            word = idx2word[idx]
            embed = '\t'.join([str(x) for x in embeds[idx]])
            f1.write(embed+'\n')
            f2.write(word+'\n')


