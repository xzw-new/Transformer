import torch
import torch.nn as nn
import math
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
from unittest import result

# todo 1. 定义类(模拟词嵌入层),实现输入部分 -> 词嵌入层的功能
class Embedding(nn.Module):
    # 1. 初始化函数
    def __init__(self,vocab_size,d_model):
        """
        :param vocab_size: 词汇表大小
        :param d_model: 词嵌入维度
        """
        # 1. 初始化父类
        super().__init__()
        # 2. 接受参数
        self.vocab_size = vocab_size
        self.d_model = d_model
        # 3. 定义词嵌入层,将单词索引映射为对应的词嵌入向量
        self.embed = nn.Embedding(vocab_size,d_model)
    # 2. 反向传播
    def forward(self,x):
        #  是为了让词嵌入和位置编码在数值尺度上对齐，防止某一方信息被淹没，同时保持梯度稳定，避免训练时梯度消失或爆炸。本质是"方差归一化"的一种变体。
        return self.embed(x) * math.sqrt(self.d_model)


# todo 2. 测试 Embedding(词嵌入层)
def dm01_embedding():
    vocab_size, d_model = 1000, 512

    my_embed = Embedding(vocab_size, d_model)

    x = torch.tensor(
        [
            [100, 2, 421, 300],
            [500, 888, 306, 509]
        ]
    )

    result = my_embed(x)

    print(f'result: {result}, result.shape: {result.shape}') # [2, 4, 512],共有2个句子,每个句子4个词,每个词512维向量表示


# todo 3. 定义类(模拟位置编码层),实现输入部分 -> 位置编码的功能
class PositionalEncoding(nn.Module):
    # 1. 初始化函数
    def __init__(self, d_model, dropout, max_len = 60):
        """
        :param d_model: 词向量维度
        :param dropout:  随机失活概率
        :param max_len: 最大句子长度
        """
        # 1. 初始化父类
        super().__init__()
        # 2. 定义dropout层, 防止过拟合
        self.dropout = nn.Dropout(p = dropout)
        # 3. 定义pe(Positional Encoding), 用于保存位置编码结果
        pe = torch.zeros(max_len,d_model)  # shape: [60, 512]
        # 4. 定义一个位置列向量, 范围: 0 ~ max_len - 1
        position = torch.arange(0,max_len).unsqueeze(1) # shape: [60, 1]
        print(f'position: {position},position.shape: {position.shape}')
        # 5. 定义1个变化矩阵,本质是: 公式里的 1/10000^(2i/d_model)
        div_term = torch.exp(torch.arange(0,d_model,2) * (-math.log(10000.0)/d_model))
        print(f'div_term: {div_term.shape}') # [256]
        # 6. 计算三角函数的值
        # position [max_len, 1] div_term [1, 256], position * div_term [max_len, 256]
        position_value = position * div_term
        # 7. 进行pe的赋值, 偶数位置使用sin()
        pe[:, 0::2] = torch.sin(position_value)
        # 8. 进行pe的赋值, 偶数位置使用cos()
        pe[:, 1::2] = torch.cos(position_value)
        # 9. 将pe进行升维,形状: [1, 60, 512]
        pe = pe.unsqueeze(0)
        # 10. 把pe注册到模型的缓冲区,利用它,不断的更新参数
        self.register_buffer('pe',pe)

    # 2. 前向传播
    def forward(self, x):
        # x: 词向量, 形状: [batch_size, seq_len, d_model] -> [1, 60, 512]
        # 这个代码的核心是: 把'词向量'和'位置编码'进行相加
        x = x + self.pe[:, :x.size(1)]

        # 随机失活,不改变形状
        return self.dropout(x)


# todo 4. 测试PositionalEncoding(位置编码层)
def dm02_position():
    # 1. 定义词汇表大小 和 词嵌入维度
    vocab_size, d_model = 1000, 512

    # 2. 实例化Embedding层:
    my_embed = Embedding(vocab_size,d_model)

    # 3. 创建输入张量, 形状是: [batch_size, seq_len]
    x = torch.tensor(
        [
            [100, 2, 421, 300],
            [500, 888, 306, 509]
        ]
    )

    # 4. 计算词向量结果
    embed_x = my_embed(x)

    # 5. 创建位置编码层对象
    my_position = PositionalEncoding(d_model = d_model,dropout = 0.1)

    # 6. 计算位置编码层结果
    position_x = my_position(embed_x)

    # 7. 返回结果
    return position_x

# todo n. 测试代码
if __name__ ==  '__main__':
    # 1. 测试词嵌入
    # dm01_embedding()

    # 2. 测试位置编码
    result = dm02_position()
    print(f'result: {result},result.shape: {result.shape}')



