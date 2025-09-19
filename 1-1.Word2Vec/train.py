# -*- coding: utf-8 -*-
# 训练基于负采样的词嵌入模型（Word2Vec Skip-gram）
import re
import math
import numpy as np
import random
import torch
import torch.utils.data as tud
import torch.optim as optim
from tqdm import tqdm
from collections import Counter
from model import EmbeddingModel
from preprocess import WordEmbeddingDataset

# device = "cuda" if torch.cuda.is_available() else 'cpu'
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
EPOCHS = 100
MAX_VOCAB_SIZE = 10000
EMBEDDING_SIZE = 200
BATCH_SIZE = 512
LR = 0.001
TRAIN_DATA_PATH = 'text8_toy.txt'
OUT_DIR = './result_example'

with open(TRAIN_DATA_PATH) as f:
    text = f.read()

# 保留字母、数字、空格、撇号、连字符
text = re.sub(r"[^a-z0-9\s'-]", '', text.lower())
words = text.split()

# 子采样（Subsampling Frequent Words）
vocab_count_ = dict(Counter(words))
total_count = sum(vocab_count_.values())

t = 1e-5
subsampling = []
for word in words:
    # 丢弃高频词
    f = vocab_count_[word] / total_count # 词频
    P_drop = 1 - math.sqrt(t / f) if f > t else 0
    if random.random() > P_drop:
        subsampling.append(word)

# 🌿🌿🌿 词向量训练中构建词汇表（vocabulary）
# ①return:  Counter({'the': 1000, 'cat': 500, 'dog': 450, ..., 'rare_word': 1})
# ②返回一个列表
# ③将列表转为 字典
vocab_count = dict(Counter(subsampling).most_common(MAX_VOCAB_SIZE - 1))
# unknown：代表所有未登录词 （9999：真实词；1：未登录词）：低频词或者训练时没见过的词或者拼写错误
vocab_count['<UNK>'] = 1

idx2word = [word for word in vocab_count.keys()]
word2idx = {word: i for i, word in enumerate(idx2word)}

# 负采样分布（Unigram ^ 3/4）：用词频的 3/4 次方归一化作为  采样概率。幂次平滑以降低高频词的采样概率。
nc = np.array([count for count in vocab_count.values()], dtype=np.float32)** (3./4.)
word_freqs = nc / np.sum(nc)

dataset = WordEmbeddingDataset(subsampling, word2idx, word_freqs)

dataloader = tud.DataLoader(dataset, BATCH_SIZE, shuffle=True)

model = EmbeddingModel(len(idx2word), EMBEDDING_SIZE)
model.to(device)
model.train()
optimizer = optim.Adam(model.parameters(), lr=LR)


# 通过多轮训练（epochs），让模型学会将 语义相近的词在向量空间中拉近，不相关的词推远。
# 使用的技术是：Skip-gram + 负采样（Negative Sampling）
for epoch in range(EPOCHS):
    pbar = tqdm(dataloader) # DataLoader 对象，每次返回一个batch的数据
    pbar.set_description("[Epoch {}]".format(epoch))
    # print(len(dataloader))
    for i, (input_labels, pos_labels, neg_labels) in enumerate(pbar):
        # 每次取出一个 batch 的三元组
        input_labels = input_labels.to(device) # 输入词（中心词） [512,]
        pos_labels = pos_labels.to(device)  # 正样本上下文词（真实上下文词）[512,4]
        neg_labels = neg_labels.to(device)  # 负样本词 （随机采样的非上下文词）[512,15]
        # optimizer.zero_grad() # 等价
        model.zero_grad()
        # 前向传播（设batch_size = 64,那么 loss 是一个包含 64个值的张量，.mean()得到一个标量），将一批样本的损失“压缩”成一个数，以便 PyTorch 能正确计算梯度并更新模型参数。
        loss = model(input_labels, pos_labels, neg_labels).mean()
        loss.backward()
        optimizer.step()
        # 将 Tensor 中的标量转换为 Python 数值
        pbar.set_postfix(loss=loss.item())

model.save_embedding(OUT_DIR, idx2word) 