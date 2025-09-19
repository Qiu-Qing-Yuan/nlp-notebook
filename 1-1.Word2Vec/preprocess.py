# -*- coding: utf-8 -*-8
# Skip-gram 负采样（Negative Sampling） 的 PyTorch **自定义数据集实现**
import torch
import torch.utils.data as tud

C = 2 # context window size （上下文窗口大小）
K = 15 # number of negative samples, K is approximate to C*2*5 for middle size corpus, thst is to pick 5 negative samples for each consubsampling word selected

class WordEmbeddingDataset(tud.Dataset):
    def __init__(self, subsampling, word2idx, word_freqs):
        '''
        Args:
            subsampling: list of  ** all the subsampling **  from the training dataset
            word2idx: the mapping from word to index
            word_freqs: normalized frequency of each word
        '''
        super(WordEmbeddingDataset, self).__init__()
        # 把每一个词转换成对应的索引
        # 如果某个词不在word2idx中（低频词或拼写错误），就用 <UNK> 的索引代替
        # return [0,1,3,0] （整数列表） -> 所有词转为索引后的列表
        self.subsampling_encoded = [word2idx.get(word, word2idx['<UNK>']) for word in subsampling]
        # 将归一化的词频（用于负采样）转为 Pytorch 张量
        self.word_freqs = torch.Tensor(word_freqs) # [10000,]
        self.word2idx = word2idx
               
    def __len__(self):
        # 有多少个词，就可以作为多少次“中心词”进行训练（尽管边缘词上下文不完整）
        return len(self.subsampling_encoded)

    # 给定一个位置 idx，返回一个训练样本。
    def __getitem__(self, idx):
        ''' 
        return:
            - center word index
            - C indices of positive subsampling
            - K indices of negative subsampling
        '''

        center_subsampling = torch.tensor(self.subsampling_encoded[idx], dtype=torch.long)

        # 提取左右上下文
        left = self.subsampling_encoded[max(idx - C, 0) : idx]
        right = self.subsampling_encoded[idx + 1 : idx + 1 + C]

        # 构造正样本上下文
        # 补齐左边到 C 个（前面补）
        padded_left = [self.word2idx['<UNK>']] * (C - len(left)) + left
        # 补齐右边到 C 个（后面补）
        padded_right = right + [self.word2idx['<UNK>']] * (C - len(right))
        # 合并：左C个 + 右C个 = 总共 2*C 个
        pos_subsampling = padded_left + padded_right
        # 负采样：从词汇表中按频率随机选出 K=15 个“负样本”词（即不是当前中心词的上下文词），用于负采样训练。
        neg_subsampling = torch.multinomial(self.word_freqs, K, True)
        # 返回一个用于 Skip-gram 模型训练的样
        return (
            center_subsampling,
            torch.LongTensor(pos_subsampling),
            neg_subsampling
        )

    # eg:
    # (
    #     tensor(5),  # center word index
    #     tensor([0, 3, 7, 0]),  # pos: [<UNK>, 'cat', 'on', <UNK>]
    #     tensor([2, 1, 9, 0, 4, ...])  # neg: 15 个随机采样词索引
    # )
    # 将被 DataLoader 自动组合成batch