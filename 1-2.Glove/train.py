# -*- coding: utf-8 -*-
import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from model import GloveModel
from preprocess import traindataloader, id2word

""" ② 准备设备 """
# device = "cuda" if torch.cuda.is_available() else 'cpu'
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
# device = torch.device("cuda:2") # 主CPU

""" ① 定义超参数 """
EPOCHS = 100
EMBEDDING_SIZE = 200
# 权重函数中的阈值，用于限制共现频率对损失的影响。当共现次数 xij 超过 X_MAX 时，权重不再增加。
X_MAX = 100
# 权重函数中的指数参数，控制权重随共现频率增长的速度。较小的 α 表示高频共现词的权重增长更慢。
ALPHA = 0.75
# 学习率（Learning Rate），用于 Adam 优化器。
LR = 0.0001
OUT_DIR = './result_example'

""" ④ 创建模型 """
model = GloveModel(len(id2word), EMBEDDING_SIZE)
model.to(device)
# model = torch.nn.DataParallel(model, device_ids=[2, 3])
model.train()

""" ⑤ 定义优化器 """
# 使用 Adam 优化器，学习率为 LR=0.0001，优化所有可学习参数（包括词向量和偏置项）。
optimizer = optim.Adam(model.parameters(), lr=LR)

# 自定义函数：权重函数与损失函数
"""
    输入：
    x: 共现频率 xij（张量）
    x_max: 阈值（100）
    alpha: 指数（0.75）
"""

""" 6️⃣ 定义 loss/weight 函数 """
def weight_func(x, x_max, alpha):
    wx = (x/x_max)**alpha
    # 确保权重不会超过1。
    wx = torch.min(wx, torch.ones_like(wx))
    return wx.to(device)

# 加权均方误差损失（Weighted MSE Loss）
def wmse_loss(weights, inputs, targets):
    # 计算当前 batch 的权重 f(xij)
    loss = weights * F.mse_loss(inputs, targets, reduction='none')
    return torch.mean(loss).to(device)

# 512 ✖️ 1572 = 804352
""" ⑦ 训练循环 """
for epoch in range(EPOCHS):
    pbar = tqdm(traindataloader)
    pbar.set_description("[Epoch {}]".format(epoch))
    # DataLoader 决定要生成一个 batch (512)
    # 随机选出 512 个索引(shuffle=True)
    for i, (xij, i_idx, j_idx) in enumerate(pbar):

        xij = xij.to(device)
        i_idx = i_idx.to(device)
        j_idx = j_idx.to(device)
        
        model.zero_grad() # 清除上一步的梯度,防止累积
        outputs = model(i_idx, j_idx) # 前向传播 GlobalModel.forward()
        weights_x = weight_func(xij, X_MAX, ALPHA)
        # outputs: 模型预测值
        # torch.log(xij): 目标值(对数共现概率)
        # weight_x:权重
        loss = wmse_loss(weights_x, outputs, torch.log(xij))
        
        loss.backward() # 反向传播:计算梯度
        optimizer.step() # 更新参数(Adam)
        # 在进度条上实时显示当前 batch 的损失值。
        pbar.set_postfix(loss=loss.item())

""" ⑧ 保存结果 """
model.save_embedding(OUT_DIR, id2word) 