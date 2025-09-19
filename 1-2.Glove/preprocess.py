# -*- coding: utf-8 -*-
import torch
import torch.utils.data as tud
from collections import Counter, defaultdict

BATCH_SIZE = 512
# 只取前 20 万词（截断）
class GloveDataset(tud.Dataset):
    def __init__(self, text, n_words=200000, window_size=5):
        super(GloveDataset, self).__init__()
        self.window_size = window_size
        self.tokens = text.split(" ")[:n_words] # 分词后的列表
        vocab = set(self.tokens) # 去重后的 全 词汇表
        self.word2id = {w: i for i, w in enumerate(vocab)}  # 构建词与 ID 的映射
        self.id2word = {i: w for w, i in self.word2id.items()}
        self.vocab_size = len(vocab)
        self.id_tokens = [self.word2id[w] for w in self.tokens] # 将原文转换为 ID 序列

        """
        构建共现矩阵,考虑距离衰减
            以 w 为中心词，在其前后 window_size=5 范围内找上下文词 c
            共现计数加上 1 / |j-i|，越近的词贡献越大
            使用 defaultdict(Counter) 存储稀疏共现关系
        """
        cooc_mat = defaultdict(Counter)
        for i, w in enumerate(self.id_tokens):
            start_i = max(i - self.window_size, 0)
            end_i = min(i + self.window_size + 1, len(self.id_tokens))
            for j in range(start_i, end_i):
                if i != j:
                    c = self.id_tokens[j]
                    cooc_mat[w][c] += 1 / abs(j-i)

        self.i_idx = list() # 中心词
        self.j_idx = list() # 上下文词
        self.xij = list()  # 加权共现频率(已除以距离)

        #Create indexes and x values tensors
        for w, cnt in cooc_mat.items():
            for c, v in cnt.items():  # c:上下文词 v:共现计数
                self.i_idx.append(w)  # 中心词
                self.j_idx.append(c)  # 上下文词
                self.xij.append(v)    # 加权共现频率

        self.i_idx = torch.LongTensor(self.i_idx)
        self.j_idx = torch.LongTensor(self.j_idx)
        self.xij = torch.FloatTensor(self.xij)
        
    def __len__(self):
        # 数据集长度 = 所有 共现对 的数量(非句子数)
        return len(self.xij)

    def __getitem__(self, idx):
        return self.xij[idx], self.i_idx[idx], self.j_idx[idx]

""" ③ 加载数据并创建 dataset/dataloader """
traindataset = GloveDataset(open("text8_toy.txt").read()) # 纯文本，空格分隔
id2word = traindataset.id2word
# 创建一个 DataLoader ，每次返回一个 batch 的数据
traindataloader = tud.DataLoader(traindataset, BATCH_SIZE, shuffle=True)

# for xij, i_idx, j_idx in traindataloader:
#    print(xij, i_idx, j_idx)
#    print('-----------')
#     print("xij:{}".format(xij))
#     print("xij.shape:{}".format(xij.shape))
#     print("i_idx:{}".format(i_idx))
#     print("i_idx.shape:{}".format(i_idx.shape))
#     print("j_idx:{}".format(j_idx))
#     print("j_idx,shape:{}".format(j_idx.shape))
#     break

# 手动创建 dataset
# traindataset = GloveDataset(open("text8_toy.txt").read())

# 手动调用 __getitem()
# sample = traindataset[0] # 此处调用
# __getitem__(0)

# print(sample)




            