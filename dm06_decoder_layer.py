from dm05_encoder import  *


# todo 1. 定义解码器层 -> 由3个子层结果
class DecoderLayer(nn.Module):
    # 1. 初始化函数
    def __init__(self,d_model,self_attn,src_attn,feed_forward, dropout = 0.1):
        """
        初始化函数, 给变量赋值
        :param d_model: 词向量维度
        :param self_attn: 注意力机制,处理(解码器输入序列内部关系), 即: 解码器的输入
        :param src_attn: 源序列(编码器-解码器)注意力机制
        :param feed_forward: 前向传播层, 强化特征的
        :param dropout: 随机失活概率
        """
        # 1. 初始化父类成员
        super().__init__()
        # 2. 定义属性
        self.d_model = d_model
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        # 3. 定义3个子层连接结构
        self.layers = clone(SublayerConnection(d_model,dropout),3)


    # 2. 解码器层的前向传播
    def forward(self,x,encoder_output,source_mask,target_mask):
        """
        解码器层 前向传播动作
        :param x: 解码器的输入序列(即: 词嵌入 + 位置编码)
        :param encoder_output: 编码器的输出序列 (词嵌入 + 源序列位置编码)
        :param source_mask: 源序列的填充掩码, 用于 编码器-解码器 注意力
        :param target_mask: 目标序列的填充掩码,用于自注意力
        :return:
        """
        # 1. 经过第1个子层 -> 多头自注意力机制层(掩码多头注意力机制)
        x = self.layers[0](x, lambda x: self.self_attn(x,x,x,target_mask))
        # 2. 经过第2个子层 -> 多头自注意力机制层(编码器-解码器)
        x = self.layers[1](x, lambda x: self.src_attn(x, encoder_output,encoder_output,source_mask))
        # 3. 经过第3个子层 -> 前向传播层(前馈全连接层)
        x = self.layers[2](x,self.feed_forward)
        # 4. 返回结果
        return x