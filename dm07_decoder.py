"""
案例:
    代码演示 解码器.

解码器介绍:
    由 解码器层组成, 论文中默认是: 6个.
"""

# 导包
from dm06_decoder_layer import *

# todo 1.定义解码器 -> 把多个解码器层(默认: 6个)进行堆叠, 最后加一层规范化层, 让输出更稳定.
class Decoder(nn.Module):
    # 1. 初始化函数.
    def __init__(self, layer, N):
        """
        初始化参数的
        :param layer: 单个解码器层(DecoderLayer)对象, 要被复制N次
        :param N: 解码器层的堆叠数量
        """
        # 1. 初始化父类成员.
        super().__init__()
        # 2. 克隆N个解码器层.
        self.layers = clone(layer, N)
        # 3. 定义最终的规范化层.
        self.norm = LayerNorm(layer.d_model)

    # 2. 前向传播函数.
    def forward(self, x, encoder_output, source_mask, target_mask):
        """
        解码器的前向传播动作.
        :param x: 解码器的输入序列(即: 词向量 + 位置编码)
        :param encoder_output: 编码器的输出序列(词嵌入 + 源序列位置编码)
        :param source_mask: 源序列的填充掩码, 用于: 编码器-解码器 注意力
        :param target_mask: 目标序列的填充掩码, 用于 自注意力.
        :return:
        """
        # 1. 数据x经过 多个 解码器层的处理即可.
        for layer in self.layers:
            # layer变量: 表示某一个具体的 解码器层.
            x = layer(x, encoder_output, source_mask, target_mask)

        # 2. 全局规范化, 把所有层的输出再统一标准化, 然后返回.
        return self.norm(x)



# todo 2. 测试解码器, 从输入到输出走一遍流程, 验证每层的维度是否正常.
def use_decoder():
    # 1. 定义变量, 记录: 解码器的输入 -> Q
    y = torch.LongTensor([
        [1, 2, 3, 4],
        [5, 6, 7, 8]
    ])

    # 2. 将值传入到Embedding层(词嵌入层)
    # 2.1 创建词嵌入层对象.
    my_embed = Embedding(1000, 512)     # 词表大小: 1000, 词嵌入维度: 512
    # 2.2 把上述的y(输入) -> 词向量形式.
    embed_y = my_embed(y)
    print(f'词嵌入层: {embed_y.shape}')     # [2, 4, 512]

    # 3. 将 embed_y(词嵌入处理结果) -> 位置编码.
    my_position = PositionalEncoding(512, 0.1)
    position_y = my_position(embed_y)
    print(f'位置编码层: {position_y}')       # [2, 4, 512]

    # 4.创建多头注意力层
    multi_attn = MultiHeadAttention(512, 8)
    self_attn = copy.deepcopy(multi_attn)       # 深拷贝 -> 确保参数不会共享.
    src_attn = copy.deepcopy(multi_attn)

    # 5. 创建前向传播层(前馈全连接层)
    ff = FeedForward(512, 2048)
    # 6. 获取编码器的输出结果.
    encoder_output = use_encoder()

    # 7. 定义mask(掩码矩阵): source_mask 和 target_mask真实作用不同(要根据你真正的业务来判断), 这里我就随便举例了.
    source_mask = torch.zeros(8, 4, 4)
    target_mask = torch.zeros(8, 4, 4)

    # 8. 实例化 解码器层对象
    my_decoder_layer = DecoderLayer(512, self_attn, src_attn, ff)

    # 9. 堆叠6层解码器层 -> 组成解码器.
    my_decoder = Decoder(my_decoder_layer, 6)

    # 10. 跑一遍解码流程: 输入 -> 6个解码器层 -> 输出
    result = my_decoder(position_y, encoder_output, source_mask, target_mask)
    print(f'解码器的形状: {result.shape}')      # [2, 4, 512]
    # print(f'解码器对象: \n{my_decoder}')

    # 11. 返回解码器处理后的结果.
    return result


# todo 3. 测试代码.
if __name__ == '__main__':
    use_decoder()