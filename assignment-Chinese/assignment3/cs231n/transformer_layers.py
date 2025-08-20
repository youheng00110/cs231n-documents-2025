import torch
import torch.nn as nn
from torch.nn import functional as F
import math

"""
该文件定义了常用于Transformer的层类型。
"""

class PositionalEncoding(nn.Module):
    """
    对序列中标记的位置信息进行编码。在这种情况下，
    该层没有可学习的参数，因为它是正弦和余弦函数的简单组合。
    """
    def __init__(self, embed_dim, dropout=0.1, max_len=5000):
        """
        构建位置编码层。

        输入：
         - embed_dim：嵌入维度的大小
         - dropout：dropout概率
         - max_len：输入序列可能的最大长度
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        assert embed_dim % 2 == 0
        # 创建一个带有“批量维度”为1的数组（将在批次中的所有样本上广播）
        pe = torch.zeros(1, max_len, embed_dim)
        ############################################################################
        # 任务：按照Transformer_Captioning.ipynb中描述的方式构建位置编码数组。
        # 目标是让每一行交替出现正弦和余弦，指数为0、0、2、2、4、4等，直到达到
        # embed_dim。当然，这种精确的规定在一定程度上是任意的，但这是自动评分器所期望的。
        # 参考：我们的解决方案少于5行代码。
        ############################################################################

        ############################################################################
        #                             代码结束部分                                  #
        ############################################################################

        # 确保位置编码会随模型参数一起保存（主要是为了完整性）
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        对输入序列逐元素添加位置嵌入。

        输入：
         - x：输入到位置编码器的序列，形状为(N, S, D)，其中N是批量大小，S是序列长度，D是嵌入维度
        返回：
         - output：输入序列+位置编码，形状为(N, S, D)
        """
        N, S, D = x.shape
        # 创建一个占位符，将在下面的代码中覆盖
        output = torch.empty((N, S, D))
        ############################################################################
        # 任务：索引到位置编码数组中，并将相应的编码添加到输入序列。别忘了之后应用dropout。
        # 这只需要几行代码。
        ############################################################################

        ############################################################################
        #                             代码结束部分                                  #
        ############################################################################
        return output


class MultiHeadAttention(nn.Module):
    """
    实现简化版掩码注意力的模型层，如“Attention Is All You Need”
    （https://arxiv.org/abs/1706.03762）中所介绍。

    使用示例：
      attn = MultiHeadAttention(embed_dim, num_heads=2)

      # 自注意力
      data = torch.randn(batch_size, sequence_length, embed_dim)
      self_attn_output = attn(query=data, key=data, value=data)

      # 使用两个输入的注意力
      other_data = torch.randn(batch_size, sequence_length, embed_dim)
      attn_output = attn(query=data, key=other_data, value=other_data)
    """

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        """
        构建一个新的MultiHeadAttention层。

        输入：
         - embed_dim：标记嵌入的维度
         - num_heads：注意力头的数量
         - dropout：dropout概率
        """
        super().__init__()
        assert embed_dim % num_heads == 0

        # 我们将为你初始化这些层，因为改变顺序会影响随机数生成
        #（进而影响你与自动评分器的输出一致性）。注意这些层使用偏置项，
        # 但这并非严格必要（不同实现可能不同）。
        self.key = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        
        self.attn_drop = nn.Dropout(dropout)

        self.n_head = num_heads
        self.emd_dim = embed_dim
        self.head_dim = self.emd_dim // self.n_head

    def forward(self, query, key, value, attn_mask=None):
        """
        计算给定数据的掩码注意力输出，并行计算所有注意力头。

        在下面的形状定义中，N是批量大小，S是源序列长度，T是目标序列长度，E是嵌入维度。

        输入：
        - query：用作查询的输入数据，形状为(N, S, E)
        - key：用作键的输入数据，形状为(N, T, E)
        - value：用作值的输入数据，形状为(N, T, E)
        - attn_mask：形状为(S, T)的数组，其中mask[i,j]==0表示源中的标记i不应影响目标中的标记j。

        返回：
        - output：形状为(N, S, E)的张量，根据key和query计算的注意力权重对value进行加权组合的结果。
        """
        N, S, E = query.shape
        N, T, E = value.shape
        # 创建一个占位符，将在下面的代码中覆盖
        output = torch.empty((N, S, E))
        ############################################################################
        # 任务：使用Transformer_Captioning.ipynb中给出的公式实现多头注意力。
        # 提示：
        #  1) 你需要将形状从(N, T, E)拆分为(N, T, H, E/H)，其中H是头数。
        #  2) torch.matmul允许批量矩阵乘法。例如，(N, H, T, E/H)与(N, H, E/H, T)相乘
        #     会得到形状为(N, H, T, T)的结果。更多示例见
        #     https://pytorch.org/docs/stable/generated/torch.matmul.html
        #  3) 对于应用attn_mask，思考如何修改分数以防止某个值影响输出。具体来说，PyTorch的
        #     masked_fill函数可能会有用。
        ############################################################################

        ############################################################################
        #                             代码结束部分                                  #
        ############################################################################
        return output


class FeedForwardNetwork(nn.Module):
    """
    简单的两层前馈网络，带有dropout和ReLU激活函数。
    """
    def __init__(self, embed_dim, ffn_dim, dropout=0.1):
        """
        输入：
         - embed_dim：输入和输出嵌入的维度
         - ffn_dim：前馈网络的隐藏维度
         - dropout：dropout概率
        """
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x):
        """
        前馈网络的前向传播。

        输入：
        - x：形状为(N, T, D)的输入张量

        返回：
        - out：与输入形状相同的输出张量
        """
        out = torch.empty_like(x)

        out = self.fc1(x)
        out = self.gelu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out


class TransformerDecoderLayer(nn.Module):
    """
    Transformer解码器的单个层，用于TransformerDecoder。
    """
    def __init__(self, input_dim, num_heads, dim_feedforward=2048, dropout=0.1):
        """
        构建TransformerDecoderLayer实例。

        输入：
         - input_dim：输入中预期的特征数量。
         - num_heads：注意力头的数量
         - dim_feedforward：前馈网络模型的维度。
         - dropout：dropout概率值。
        """
        super().__init__()
        self.self_attn = MultiHeadAttention(input_dim, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(input_dim, num_heads, dropout)
        self.ffn = FeedForwardNetwork(input_dim, dim_feedforward, dropout)

        self.norm_self = nn.LayerNorm(input_dim)
        self.norm_cross = nn.LayerNorm(input_dim)
        self.norm_ffn = nn.LayerNorm(input_dim)

        self.dropout_self = nn.Dropout(dropout)
        self.dropout_cross = nn.Dropout(dropout)
        self.dropout_ffn = nn.Dropout(dropout)


    def forward(self, tgt, memory, tgt_mask=None):
        """
        将输入（和掩码）通过解码器层。

        输入：
        - tgt：解码器层的输入序列，形状为(N, T, D)
        - memory：编码器最后一层的输出序列，形状为(N, S, D)
        - tgt_mask：目标序列中需要掩码的部分，形状为(T, T)

        返回：
        - out：Transformer特征，形状为(N, T, W)
        """

        # 自注意力块（参考实现）
        shortcut = tgt
        tgt = self.self_attn(query=tgt, key=tgt, value=tgt, attn_mask=tgt_mask)
        tgt = self.dropout_self(tgt)
        tgt = tgt + shortcut
        tgt = self.norm_self(tgt)

        ############################################################################
        # 任务：完成解码器层，实现剩余两个子层：(1)使用编码器输出作为memory的交叉注意力块，
        # (2)前馈块。每个块应遵循上面自注意力实现的相同结构。
        ############################################################################

        ############################################################################
        #                             代码结束部分                                  #
        ############################################################################

        return tgt


class PatchEmbedding(nn.Module):
    """
    将图像分割为补丁并将每个补丁投影到嵌入向量的层。
    用作视觉Transformer（ViT）的输入层。

    输入：
    - img_size：输入图像的高/宽（假设是正方形图像）。
    - patch_size：每个补丁的高/宽（正方形补丁）。
    - in_channels：输入图像的通道数（例如，RGB为3）。
    - embed_dim：线性嵌入空间的维度。
    """
    def __init__(self, img_size, patch_size, in_channels=3, embed_dim=128):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        assert img_size % patch_size == 0, "图像尺寸必须能被补丁尺寸整除。"

        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = patch_size * patch_size * in_channels

        # 补丁展平后的线性投影（到嵌入维度）
        self.proj = nn.Linear(self.patch_dim, embed_dim)


    def forward(self, x):
        """
        补丁嵌入的前向传播。

        输入：
        - x：形状为(N, C, H, W)的输入图像张量

        返回：
        - out：带位置编码的补丁嵌入，形状为(N, num_patches, embed_dim)
        """
        N, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, \
            f"预期图像尺寸为({self.img_size}, {self.img_size})，但得到({H}, {W})"
        out = torch.zeros(N, self.embed_dim)

        ############################################################################
        # 任务：将图像分割为形状为(patch_size x patch_size x C)的非重叠补丁，并将它们重排
        # 为形状为(N, num_patches, patch_dim)的张量。不要使用for循环。
        # 相反，你可能会发现torch.reshape和torch.permute对这一步有帮助。一旦补丁被展平，
        # 使用投影层将它们嵌入到潜在向量中。
        ############################################################################

        ############################################################################
        #                             代码结束部分                                  #
        ############################################################################
        return out


class TransformerEncoderLayer(nn.Module):
    """
    Transformer编码器的单个层，用于TransformerEncoder。
    """
    def __init__(self, input_dim, num_heads, dim_feedforward=2048, dropout=0.1):
        """
        构建TransformerEncoderLayer实例。

        输入：
         - input_dim：输入中预期的特征数量。
         - num_heads：注意力头的数量。
         - dim_feedforward：前馈网络模型的维度。
         - dropout：dropout概率值。
        """
        super().__init__()
        self.self_attn = MultiHeadAttention(input_dim, num_heads, dropout)
        self.ffn = FeedForwardNetwork(input_dim, dim_feedforward, dropout)

        self.norm_self = nn.LayerNorm(input_dim)
        self.norm_ffn = nn.LayerNorm(input_dim)

        self.dropout_self = nn.Dropout(dropout)
        self.dropout_ffn = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        """
        将输入（和掩码）通过编码器层。

        输入：
        - src：编码器层的输入序列，形状为(N, S, D)
        - src_mask：源序列中需要掩码的部分，形状为(S, S)

        返回：
        - out：Transformer特征，形状为(N, S, D)
        """
        ############################################################################
        # 任务：通过应用自注意力，然后是前馈块来实现编码器层。代码将与解码器层非常相似。
        ############################################################################

        ############################################################################
        #                             代码结束部分                                  #
        ############################################################################
        return src