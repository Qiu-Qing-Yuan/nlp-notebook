# -*- coding: utf-8 -*-
import re
import math
import pandas as pd
import string
from collections import OrderedDict


class NewWord(object):
    def __init__(self, max_len_word, radio):
        self.max_len_word = max_len_word
        self.radio = radio
        self.words = {}

    # 提取候选词
    # 枚举 + 频率统计
    def find_words(self, doc):
        '''
        找出所有可能出现的词:从位置 i 开始，生成长度为 1直到 max_len_word 的所有子串
        :param doc: 传入的文本
        :param max_len_word:
        :return:
        '''
        len_doc = len(doc)
        for i in range(len_doc):
            end = min(i + self.max_len_word + 1, len_doc + 1)
            for j in range(i + 1, end):
                # 从文本中切出从i到j-1的子串。作为一个候选词
                # self.words: 是一个字典(dict)，用来存储所有出现过的“词”及其信息（词频）
                word = doc[i:j]
                if word in self.words:
                    # in: 检查是否已经作为“词”被记录过
                    # eg: self.words["天气"] = {'freq': 3}
                    self.words[word]['freq'] += 1
                else:
                    # 这个词第一次出现
                    self.words[word] = {} # 为这个词创建一个空字典
                    self.words[word]['freq'] = 1 # 设置初始频率为 1

    # 计算聚合度
    # 1、首先计算所有候选词出现的概率
    # 2、通过概率计算每一个词的聚合度
    def dop(self):
        '''
        计算聚合度（dop）:用于判断一个字符串是否是一个“紧密组合”的词：真正的词 而非 随机组合
        :param words:
        :return:
        '''
        len_words = len(self.words)  # 候选词总数
        # 计算每一个词频
        # k:   当前词
        # v:   {'freq': 5}

        # old
        # for k, v in self.words.items():
        # self.words[k]['freq_radio']：这个词出现的次数
        # 5 * len_words：分母是一个 经验性归一化因子
        # self.words[k]['freq_radio'] = self.words[k]['freq']/(5 * len_words)

        # 计算所有候选词的总频率
        # 计算所有词出现的总次数 （self.words 中每个词的 freq的总和）

        # new 总候选词个数
        total_candidate_freq = sum(info['freq'] for info in self.words.values())
        for k, v in self.words.items():
            self.words[k]['freq_radio'] = v['freq'] / total_candidate_freq

        for k, v in self.words.items():
            # 初始化空列表dop，用于存储所有可能切分方式下，左右部分频率比的乘积
            l = len(k)
            if l == 1:
                # 单字无法再分割，无“内部聚合”
                self.words[k]['dop'] = 0
                continue

            total_product = 0.0
            freq_radio_w = self.words[k]['freq_radio']  # 当前词自身的归一化频率

            for i in range(1, l):  # 遍历该词
                left = k[0: i]
                right = k[i: l]
                # 确保左右部分都在字典中
                if left in self.words and right in self.words:
                    p_left = self.words[left]['freq_radio']
                    p_right = self.words[right]['freq_radio']
                    # ????
                    total_product += p_left * p_right

            # 分子：该词自身的归一化频率（freq_radio）
            # 分母：所有切分方式下左右部分频率乘积之和（dop）
            # 取 log:将比值转换为对数尺度，便于比较和排序

            if total_product > 0:
                self.words[k]['dop'] = math.log(freq_radio_w / total_product)
            else:
                self.words[k]['dop'] = float('inf')  # “正无穷大”（infinity）

    def left_free(self, doc):
        '''
        计算左自由度（Left Freedom）
        这个词的左边能接哪些 不同的字？越多样，自由度越高，越像一个独立的词
        :param doc:
        :return:
        '''
        for k, v in self.words.items():
            # 用于找出 某词 k 在文本 doc 中所有可以提取左邻字的位置
            left_list = [m.start() for m in re.finditer(k, doc) if m.start() > 0]
            # 计算有多少个位置可以提取左邻字
            len_left_list = len(left_list)
            left_item = {}  # 创建一个空字典，用于统计每个 左邻字 出现的概率
            for li in left_list:
                if doc[li-1] in left_item:
                    left_item[doc[li-1]] += 1
                else:
                    left_item[doc[li-1]] = 1
            left = 0  # 用于累加“左自由度”
            for _k, _v in left_item.items():
                p = _v / len_left_list
                left -= p * math.log(p)
            self.words[k]['left_free'] = left

    def right_free(self, doc):
        '''
        计算右自由度
        :param doc:
        :return:
        '''
        for k, v in self.words.items():
            # 计算有多少位置可以提取右邻字
            right_list = [m.start() for m in re.finditer(k, doc) if m.start() + len(k) < len(doc)]
            len_right_list = len(right_list)
            right_item = {}  # 用于统计每个右邻字的出现次数
            for li in right_list:
                if doc[li+len(k)] in right_item:
                    right_item[doc[li+len(k)]] += 1
                else:
                    right_item[doc[li+len(k)]] = 1
            right = 0
            for _k, _v in right_item.items():
                p = _v / len_right_list
                right -= p * math.log(p) # 标准熵
            self.words[k]['right_free'] = right

    # self.words:
    # {
    #     '天气': {'freq': 5, 'dop': 3.2, 'left_free': 1.5, 'right_free': 1.8},
    #     '今天': {'freq': 4, 'dop': 2.8, 'left_free': 1.2, 'right_free': 2.0},
    #     ...
    # }

    # pd.DataFrame(self.words)
    #         天气   今天
    # freq     5      4
    # dop     3.2    2.8
    def get_df(self):
        # 将 self.words 字典转换为一个 Pandas DataFrame，计算综合得分，排序并筛选，最终返回一个结构化的结果表，用于输出或分析
        df = pd.DataFrame(self.words)
        df = df.T # 转置 Transpose
    #     freq dop left_free right_free
    # 天气 5   3.2    1.5         1.8
        df['score'] = df['dop'] + df['left_free'] + df['right_free']
        df = df.sort_values(by='score', ascending=False)

        # 增强筛选
        df = df[
            (df['score'] > self.radio) &
            (df.index.str.len() > 1) &
            (df['freq'] >= 2)]

        return df

    # 主控函数（pipeline）
    # self 是当前 NewWord 类的实例
    def run(self, doc):
        doc = re.sub('[,，.。"“”‘’\';；:：、？?！!\n\[\]\(\)（）\\/a-zA-Z0-9\s ]', '', doc) # 清洗文本
        self.find_words(doc) # 提取候选词
        self.dop()
        self.left_free(doc)
        self.right_free(doc)
        df = self.get_df()
        return df

if __name__ == '__main__':
    doc = open('./model/data.txt', 'r', encoding='utf-8').read()
    nw = NewWord(max_len_word=5, radio=6.6)
    df = nw.run(doc)
    df.to_csv('./model/text.txt', sep='|', encoding='utf-8')