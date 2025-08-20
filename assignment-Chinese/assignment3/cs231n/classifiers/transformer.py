import numpy as np
import copy

import torch
import torch.nn as nn

from ..transformer_layers import *


class CaptioningTransformer(nn.Module):
    """
    CaptioningTransformer通过Transformer解码器从图像特征生成字幕。

    该Transformer接收大小为D的输入向量，词汇表大小为V，
    处理长度为T的序列，使用维度为W的词向量，
    并在大小为N的小批量上操作。
    """
    def __init__(self, word_to_idx, input_dim, wordvec_dim, num_heads=4,
                 num_layers=2, max_length=50):
        """
        构建一个新的CaptioningTransformer实例。

        输入：
        - word_to_idx: 提供词汇表的字典。它包含V个条目，
          并将每个字符串映射到[0, V)范围内的唯一整数。
        - input_dim: 输入图像特征向量的维度D。
        - wordvec_dim: 词向量的维度W。
        - num_heads: 注意力头的数量。
        - num_layers: transformer层的数量。
        - max_length: 最大可能的序列长度。
        """
        super().__init__()

        vocab_size = len(word_to_idx)
        self.vocab_size = vocab_size
        self._null = word_to_idx["<NULL>"]
        self._start = word_to_idx.get("<START>", None)
        self._end = word_to_idx.get("<END>", None)

        self.visual_projection = nn.Linear(input_dim, wordvec_dim)  # 将图像特征投影到词向量维度
        self.embedding = nn.Embedding(vocab_size, wordvec_dim, padding_idx=self._null)  # 词嵌入层
        self.positional_encoding = PositionalEncoding(wordvec_dim, max_len=max_length)  # 位置编码

        decoder_layer = TransformerDecoderLayer(input_dim=wordvec_dim, num_heads=num_heads)  # 解码器层
        self.transformer = TransformerDecoder(decoder_layer, num_layers=num_layers)  # Transformer解码器
        self.apply(self._init_weights)  # 初始化权重

        self.output = nn.Linear(wordvec_dim, vocab_size)  # 输出层，映射到词汇表大小

    def _init_weights(self, module):
        """
        初始化网络权重。
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)  # 正态分布初始化
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()  # 偏置初始化为0
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()  # 层归一化偏置初始化为0
            module.weight.data.fill_(1.0)  # 层归一化权重初始化为1

    def forward(self, features, captions):
        """
        给定图像特征和字幕标记，返回每个时间步可能标记的分布。注意，由于
        整个字幕序列同时提供，我们会屏蔽未来的时间步。

        输入：
         - features: 图像特征，形状为(N, D)
         - captions: 真实字幕，形状为(N, T)

        返回：
         - scores: 每个时间步每个标记的分数，形状为(N, T, V)
        """
        N, T = captions.shape
        # 创建一个占位符，将在下面的代码中覆盖
        scores = torch.empty((N, T, self.vocab_size))
        ############################################################################
        # 任务：实现CaptionTransformer的forward函数。                              #
        # 提示：                                                                    #
        #  1) 首先需要嵌入字幕并添加位置编码。然后需要将图像特征投影到相同的维度。    #
        #  2) 需要准备一个掩码(tgt_mask)来屏蔽字幕中未来的时间步。torch.tril()函数可能#
        #     有助于准备这个掩码。                                                  #
        #  3) 最后，在文本和图像嵌入以及tgt_mask上应用解码器特征。将输出投影到每个标记的分数#
        ############################################################################

        ############################################################################
        #                             代码结束部分                                  #
        ############################################################################

        return scores

    def sample(self, features, max_length=30):
        """
        给定图像特征，使用贪心解码预测图像字幕。

        输入：
         - features: 图像特征，形状为(N, D)
         - max_length: 最大可能的字幕长度

        返回：
         - captions: 每个示例的字幕，形状为(N, max_length)
        """
        with torch.no_grad():
            features = torch.Tensor(features)
            N = features.shape[0]

            # 创建一个空的字幕张量（所有标记都是NULL）
            captions = self._null * np.ones((N, max_length), dtype=np.int32)

            # 创建部分字幕，只包含开始标记
            partial_caption = self._start * np.ones(N, dtype=np.int32)
            partial_caption = torch.LongTensor(partial_caption)
            # [N] -> [N, 1]
            partial_caption = partial_caption.unsqueeze(1)

            for t in range(max_length):

                # 预测下一个标记（忽略所有其他时间步）
                output_logits = self.forward(features, partial_caption)
                output_logits = output_logits[:, -1, :]

                # 从词汇表中选择最可能的词ID
                # [N, V] -> [N]
                word = torch.argmax(output_logits, axis=1)

                # 更新我们的整体字幕和当前的部分字幕
                captions[:, t] = word.numpy()
                word = word.unsqueeze(1)
                partial_caption = torch.cat([partial_caption, word], dim=1)

            return captions


def clones(module, N):
    "生成N个相同的层。"
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class TransformerDecoder(nn.Module):
    """Transformer解码器"""
    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = clones(decoder_layer, num_layers)  # 克隆多个解码器层
        self.num_layers = num_layers

    def forward(self, tgt, memory, tgt_mask=None):
        output = tgt

        for mod in self.layers:
            output = mod(output, memory, tgt_mask=tgt_mask)  # 逐层处理

        return output


class TransformerEncoder(nn.Module):
    """Transformer编码器"""
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = clones(encoder_layer, num_layers)  # 克隆多个编码器层
        self.num_layers = num_layers

    def forward(self, src, src_mask=None):
        output = src

        for mod in self.layers:
            output = mod(output, src_mask=src_mask)  # 逐层处理

        return output


class VisionTransformer(nn.Module):
    """
    视觉Transformer（ViT）实现。
    """
    def __init__(self, img_size=32, patch_size=8, in_channels=3,
                 embed_dim=128, num_layers=6, num_heads=4,
                 dim_feedforward=256, num_classes=10, dropout=0.1):
        """
        输入：
         - img_size: 输入图像的大小（假设为正方形）。
         - patch_size: 每个补丁的大小（假设为正方形）。
         - in_channels: 图像通道数。
         - embed_dim: 每个补丁的嵌入维度。
         - num_layers: Transformer编码器层的数量。
         - num_heads: 注意力头的数量。
         - dim_feedforward: 前馈网络的隐藏大小。
         - num_classes: 分类标签的数量。
         - dropout: Dropout概率。
        """
        super().__init__()
        self.num_classes = num_classes
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)  # 补丁嵌入
        self.positional_encoding = PositionalEncoding(embed_dim, dropout=dropout)  # 位置编码

        encoder_layer = TransformerEncoderLayer(embed_dim, num_heads, dim_feedforward, dropout)  # 编码器层
        self.transformer = TransformerEncoder(encoder_layer, num_layers=num_layers)  # Transformer编码器

        # 从池化的标记预测类别分数的最终分类层
        self.head = nn.Linear(embed_dim, num_classes)

        self.apply(self._init_weights)  # 初始化权重


    def _init_weights(self, module):
        """
        初始化网络权重。
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)  # 正态分布初始化
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()  # 偏置初始化为0
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()  # 层归一化偏置初始化为0
            module.weight.data.fill_(1.0)  # 层归一化权重初始化为1

    def forward(self, x):
        """
        视觉Transformer的前向传播。

        输入：
         - x: 输入图像张量，形状为(N, C, H, W)

        返回：
         - logits: 输出分类logits，形状为(N, num_classes)
        """
        N = x.size(0)
        logits = torch.zeros(N, self.num_classes, device=x.device)
        
        ############################################################################
        # 任务：实现视觉Transformer的前向传播。                                    #
        # 1. 将输入图像转换为补丁向量序列。                                         #
        # 2. 添加位置编码以保留空间信息。                                           #
        # 3. 将序列通过Transformer编码器。                                          #
        # 4. 对补丁向量进行平均池化，为每个图像获取一个特征向量。可以使用torch.mean。  #
        # 5. 将其通过线性层生成类别logits。                                         #
        ############################################################################

        ############################################################################
        #                             代码结束部分                                  #
        ############################################################################


        return logits
