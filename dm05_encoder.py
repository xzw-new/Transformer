from dm04_encoder_layer import *

# todo 1. 定义编码器类
class Encoder(nn.Module):
    # 1. 初始化函数
    def __init__(self,layer,N):
        """
        参数初始化函数
        :param layer: 单个编码器层
        :param N: 编码器层数量
        """
        # 1. 初始化父类成员
        super().__init__()
        # 2. 克隆N个 编码器层对象
        self.layers = clone(layer,N)
        # 3. 定义最终的规范化层
        self.norm = LayerNorm(layer.d_model)   # 512维

    # 2. 定义前向传播函数
    def forward(self,x,mask):
        """
        前向传播函数
        :param x: 输入数据
        :param mask: 掩码矩阵
        :return:
        """
        # 1. x(待处理的数据) 依次经过 N个 编码器层的实现
        for layer in self.layers:
            # x通过每一个 编码器层的处理
            x = layer(x, mask)
        # 2. 最终规范化,提升模型的稳定性
        return self.norm(x)


# todo 1. 测试编码器类
def use_encoder():
    # 1. 获取数据 -> 词向量 + 位置编码
    x = dm02_position()

    # 2. 构建单个编码器层(作为基础单元)
    # 2.1 多头注意力对象
    multi_head = MultiHeadAttention(512,8)

    # 2.2 前馈全连接对象
    feed_forward = FeedForward(512,2048)

    # 2.3 把 多头注意力层 和 前馈全连接层 组合起来 -> 编码器层对象
    encoder_layer = EncoderLayer(512, multi_head, feed_forward)

    # 3. 实例化编码器(堆叠 3个 编码器层), 论文默认是 6个
    encoder = Encoder(encoder_layer,3)

    # 4. 构建掩码张量, 形状: [batch_size, seq_len, seq_len] -> [2, 4, 4]
    mask = torch.zeros(8, 4, 4)

    # 5. 执行编码过程
    encoder_output = encoder(x, mask)

    # 6. 打印结果
    print(f'编码器的输出形状: {encoder_output.shape}')  # [2, 4, 512]

    # 7. 可选, 观察输入 和 输出结果
    print(f'输入示例: \n {x[0, 0, :5]}')                 # 第1个句子的第一个单词的 前5个词向量
    print(f'输出示例: \n {encoder_output[0, 0, :5]}')    # # 第1个句子的第一个单词的 前5个词向量


# todo 3. 测试代码
if __name__ == '__main__':
    use_encoder()






