"""
案例:
    演示 Transformer的输出部分.
"""

# 导包
from dm07_decoder import *


# todo 1. 定义输出部分, 把解码器的特征 转成 最终预测结果(即: 词汇表中每个词的概率)
class Generator(nn.Module):
    # 1. 初始化函数.
    def __init__(self, d_model, vocab_size):
        """
        输出部分的 初始化函数
        :param d_model: 词向量维度, 例如: 512
        :param vocab_size: 词汇表大小, 例如: 1000
        """
        # 1. 初始化父类成员.
        super().__init__()
        # 2. 定义全连接层, 封装输入维度(词向量512) -> 输出维度(1000)
        self.linear = nn.Linear(d_model, vocab_size)

    # 2. 前向传播函数.
    def forward(self, x):
        # 1. 线性层处理: [2, 4, 512] -> [2, 4, 1000]
        x = self.linear(x)

        # 2. log_softmax()函数: 将分数 转成 对数概率分布.
        # dim=-1 表示最最后一维(词汇表维度)进行计算, 确保概率和为: 1
        return F.log_softmax(x, dim=-1)


# todo 2. 测试输出部分.
def use_generator():
    """
    获取解码器的特征 -> 通过生成器转成目标词汇的概率分布 -> 验证输出维度
    :return:
    """
    # 1. 获取解码器的输出结果.
    result = use_decoder()

    # 2. 初始化 输出生成器, 将512维的解码器输出, 转成1000维的概率分布.
    generator = Generator(512, 1000)
    # 3. 通过生成器, 获取概率分布.
    output = generator(result)

    # 4. 验证输出维度.
    print(f'output 模型最终输出结果: {output.shape}, {output}')     # [2, 4, 1000]

    # 5. 验证概率和为1.
    # 5.1 提取第1个样本, 第1个词的 对数概率
    log_probs = output[0, 0]                  # 形状: torch.Size([1000]), 第1个样本, 第1个词的概率分布
    # log_probs = output[1, 3]                # 形状: torch.Size([1000]), 第2个样本, 第4个词的概率分布
    # print(f'log_probs: {log_probs.shape}')    # torch.Size([1000])

    # 5.2 把对数概率 -> 普通概率
    probs = torch.exp(log_probs)
    # print(f'probs: {probs}')            # 具体的1000个词的概率分布

    # 5.3 验证概率和为1
    print(f'第1个词的概率和为: {torch.sum(probs)}') # 1.0
    print(f'第1个词的概率和为: {probs.sum()}')      # 1.0
    

# todo 3. 测试代码
if __name__ == '__main__':
    use_generator()