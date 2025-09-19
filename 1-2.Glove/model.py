# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn

""" 实现 GloVe 模型结构（双嵌入 + 偏置） """
class GloveModel(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(GloveModel, self).__init__()
        self.wi = nn.Embedding(num_embeddings, embedding_dim)
        self.wj = nn.Embedding(num_embeddings, embedding_dim)
        self.bi = nn.Embedding(num_embeddings, 1)
        self.bj = nn.Embedding(num_embeddings, 1)

        self.wi.weight.data.uniform_(-1, 1)
        self.wj.weight.data.uniform_(-1, 1)
        self.bi.weight.data.zero_()
        self.bj.weight.data.zero_()

    def forward(self, i_indices, j_indices):
        w_i = self.wi(i_indices) #[batch_size, embedding_dim]
        w_j = self.wj(j_indices) #[batch_size, embedding_dim]
        b_i = self.bi(i_indices).squeeze() #[batch_size]
        b_j = self.bj(j_indices).squeeze() #[batch_size]
        x = torch.sum(w_i * w_j, dim=1) + b_i + b_j #[batch_size]
        return x
    
    # def save_embedding(self, outdir, idx2word):
    #     embeds = self.wi.weight.data.cpu().numpy() + self.wj.weight.data.cpu().numpy()
    #     f1 = open(os.path.join(outdir, 'vec.tsv'), 'w')
    #     f2 = open(os.path.join(outdir, 'word.tsv'), 'w')
    #     for idx in range(len(embeds)):
    #         word = idx2word[idx]
    #         embed = '\t'.join([str(x) for x in embeds[idx]])
    #         f1.write(embed+'\n')
    #         f2.write(word+'\n')
    def save_embedding(self, outdir, idx2word):
        # embeds 是一个 连续的二维数组（C-contiguous）
        embeds = self.wi.weight.data.cpu().numpy() + self.wj.weight.data.cpu().numpy()
        with open(os.path.join(outdir, 'vec.tsv'), 'w') as f1, \
                open(os.path.join(outdir, 'word.tsv'), 'w') as f2:
            for idx in range(len(embeds)):
                word = idx2word[idx]
                # embeds[i] 是第 i 个词的最终词向量
                embed = '\t'.join([str(x) for x in embeds[idx]])
                f1.write(embed + '\n')
                f2.write(word + '\n')
