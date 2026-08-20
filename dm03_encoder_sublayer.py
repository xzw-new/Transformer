from dm01_input import *
from dm02_encoder_element import *

# todo 1. 初始化子层连接结构,核心是:残差连接 + 规范化层,让模型训练更稳定,避免梯度消失或者梯度爆炸
class SublayerConnection(nn.Module):
    # 1. 初始化函数
    def __init__(self,d_model,dropout = 0.1):
        """
        函数功能: 初始化子层连接结构(参数的)
        :param d_model: 输入的维度, 即: 词向量维度
        :param dropout: 随机失活的概率
        """
        # 1. 初始化父类成员
        super().__init__()
        # 2. 定义规范化层(LayerNorm)
        self.norm = LayerNorm(d_model)
        # 3. 定义随机失活层(Dropout)
        self.dropout = nn.Dropout(dropout)

    # 2. 前向传播函数
    def forward(self,x,sublayer):
        """
        函数功能: 完整具体的 子层的 计算动作
        :param x: 输入张量, 形状一般是: [batch_size, seq_len, d_model] -> [2, 4, 512]
        :param sublayer: 输入的子层对象, 例如: 多头注意力对象,前馈传播对象
        :return:
        """
        # 子层的核心逻辑: 有两种常见的方式
        # 方式1(大多数场景都用这个): 先子层处理, 在残差连接, 层规范化
        #           随机失活       层规范化    子层处理        残差连接
        my_result = self.dropout(self.norm(sublayer(x))) + x

        # 方式2: 先规范化层, 再子层处理, 残差连接

        # 返回结果
        return my_result

# todo 2. 测试子层连接结构
def use_sublayer():
    """
    测试子层连接结构的完整流程:
        1. 准备输入数据(词嵌入 + 位置编码)
        2. 实例化子层连接结构
        3. 定义子层函数(这里用多头注意力机制)
        4. 通过子层连接处理输入, 验证残差连接和规范化的效果
    :return:
    """
    # 1. 准备输入数据(词嵌入 + 位置编码)
    x = dm02_position()

    # 2. 实例化子层连接结构
    sublayer_conn = SublayerConnection(d_model=512)

    # 3. 定义子层函数(这里用 多头注意力 作为示例)
    # 多头注意力子层
    # result = sublayer_conn(x,lambda x:MultiHeadAttention(512,8)(x,x,x))

    # 前馈全连接子层
    result = sublayer_conn(x,lambda x: FeedForward(d_model=512,d_ff=2048)(x))

    print(f'子层处理结果: {result.shape}') # 子层处理结果: torch.size([2,4,512])

# todo 3. 测试代码
if __name__ == '__main__':
    use_sublayer()














