import numpy as np
import torch
from ..rnn_layers_pytorch import *


class CaptioningRNN:
    """
    CaptioningRNN（图像描述循环神经网络）使用循环神经网络从图像特征生成描述文字。

    该RNN接收维度为D的输入向量，词汇表大小为V，处理长度为T的序列，RNN隐藏层维度为H，
    使用维度为W的词向量，并在大小为N的小批量数据上运行。

    注意：CaptioningRNN不使用任何正则化方法。
    """

    def __init__(
        self,
        word_to_idx,
        input_dim=512,
        wordvec_dim=128,
        hidden_dim=128,
        cell_type="rnn",
        dtype=torch.float32,
    ):
        """
        构造一个新的CaptioningRNN实例。

        输入：
        - word_to_idx: 词汇表字典。包含V个条目，将每个字符串映射到[0, V)范围内的唯一整数。
        - input_dim: 输入图像特征向量的维度D。
        - wordvec_dim: 词向量的维度W。
        - hidden_dim: RNN隐藏状态的维度H。
        - cell_type: 使用的RNN类型；'rnn'或'lstm'。
        - dtype: 使用的numpy数据类型；训练时用float32，数值梯度检查时用float64。
        """
        if cell_type not in {"rnn", "lstm"}:
            raise ValueError('无效的cell_type "%s"' % cell_type)

        self.cell_type = cell_type  # RNN类型（rnn或lstm）
        self.dtype = dtype  # 数据类型
        self.word_to_idx = word_to_idx  # 词到索引的映射
        self.idx_to_word = {i: w for w, i in word_to_idx.items()}  # 索引到词的映射
        self.params = {}  # 存储模型参数的字典

        vocab_size = len(word_to_idx)  # 词汇表大小V

        # 特殊标记的索引
        self._null = word_to_idx["<NULL>"]  # 空标记的索引
        self._start = word_to_idx.get("<START>", None)  # 起始标记的索引（可能不存在）
        self._end = word_to_idx.get("<END>", None)  # 结束标记的索引（可能不存在）

        # 初始化词向量矩阵
        self.params["W_embed"] = torch.randn(vocab_size, wordvec_dim)
        self.params["W_embed"] /= 100  # 缩放初始化值

        # 初始化CNN特征到隐藏状态的投影参数
        self.params["W_proj"] = torch.randn(input_dim, hidden_dim)
        self.params["W_proj"] /= np.sqrt(input_dim)  # 标准化初始化
        self.params["b_proj"] = torch.zeros(hidden_dim)  # 偏置初始化为0

        # 初始化RNN参数
        dim_mul = {"lstm": 4, "rnn": 1}[cell_type]  # LSTM的参数维度是RNN的4倍（4个门）
        self.params["Wx"] = torch.randn(wordvec_dim, dim_mul * hidden_dim)
        self.params["Wx"] /= np.sqrt(wordvec_dim)  # 标准化初始化
        self.params["Wh"] = torch.randn(hidden_dim, dim_mul * hidden_dim)
        self.params["Wh"] /= np.sqrt(hidden_dim)  # 标准化初始化
        self.params["b"] = torch.zeros(dim_mul * hidden_dim)  # RNN偏置初始化为0

        # 初始化隐藏状态到词汇表的输出参数
        self.params["W_vocab"] = torch.randn(hidden_dim, vocab_size)
        self.params["W_vocab"] /= np.sqrt(hidden_dim)  # 标准化初始化
        self.params["b_vocab"] = torch.zeros(vocab_size)  # 偏置初始化为0

        # 将所有参数转换为指定的数据类型
        for k, v in self.params.items():
            self.params[k] = v.to(self.dtype)

    def loss(self, features, captions):
        """
        计算RNN的训练时损失。输入图像特征和对应图像的真实描述，
        使用RNN（或LSTM）计算所有参数的损失和梯度。

        输入：
        - features: 输入图像特征，形状为(N, D)
        - captions: 真实描述；整数数组，形状为(N, T + 1)，其中每个元素在[0, V)范围内

        返回：
        - loss: 标量损失值
        """
        # 将描述分割为两部分：captions_in包含除最后一个词外的所有词，作为RNN的输入；
        # captions_out包含除第一个词外的所有词，是RNN期望生成的目标。
        # 两者相对偏移一个位置，因为RNN应在接收词t后生成词t+1。
        # captions_in的第一个元素是START标记，captions_out的第一个元素是第一个词。
        captions_in = captions[:, :-1]
        captions_out = captions[:, 1:]

        # 需要用这个掩码忽略<NULL>标记的位置
        mask = captions_out != self._null

        # 从图像特征到初始隐藏状态的仿射变换的权重和偏置
        W_proj, b_proj = self.params["W_proj"], self.params["b_proj"]

        # 词嵌入矩阵
        W_embed = self.params["W_embed"]

        # RNN的输入到隐藏、隐藏到隐藏的权重和偏置
        Wx, Wh, b = self.params["Wx"], self.params["Wh"], self.params["b"]

        # 隐藏状态到词汇表的变换的权重和偏置
        W_vocab, b_vocab = self.params["W_vocab"], self.params["b_vocab"]

        loss = 0.0
        ###########################################################################################
        # 代办: 实现CaptioningRNN的前向传播。                                                       #
        # 前向传播需要完成以下步骤：                                                                #
        # (1) 使用仿射变换从图像特征计算初始隐藏状态，形状应为(N, H)                                 #
        # (2) 使用词嵌入层将captions_in中的词从索引转换为向量，形状为(N, T, W)                       #
        # (3) 根据self.cell_type，使用 vanilla RNN 或 LSTM 处理输入词向量序列，                     #
        #     生成所有时间步的隐藏状态，形状为(N, T, H)                                             #
        # (4) 使用（时间）仿射变换，通过隐藏状态计算每个时间步的词汇表得分，                          #
        #     形状为(N, T, V)                                                                     #
        # (5) 使用（时间）softmax，结合captions_out计算损失，忽略输出词为<NULL>的位置（通过上述mask）#
        #                                                                                        #
        # 不需要考虑权重的正则化及其梯度！                                                         #
        #                                                                                        #
        # 也不需要实现反向传播。                                                                  #
        #########################################################################################
        # 
        ############################################################################
        #                             你的代码结束                                 #
        ############################################################################

        return loss

    def sample(self, features, max_length=30):
        """
        运行模型的测试时前向传播，为输入的特征向量生成描述。

        在每个时间步，我们嵌入当前词，将其和前一个隐藏状态传入RNN得到下一个隐藏状态，
        使用隐藏状态计算所有词汇的得分，选择得分最高的词作为下一个词。初始隐藏状态通过
        对输入图像特征应用仿射变换得到，初始词是<START>标记。

        对于LSTM，还需要跟踪细胞状态；此时初始细胞状态应设为0。

        输入：
        - features: 输入图像特征数组，形状为(N, D)
        - max_length: 生成描述的最大长度T

        返回：
        - captions: 形状为(N, max_length)的数组，给出生成的描述，
          其中每个元素是[0, V)范围内的整数。captions的第一个元素是第一个生成的词，不是<START>标记。
        """
        N = features.shape[0]  # 小批量大小
        captions = self._null * torch.ones((N, max_length), dtype=torch.long)  # 初始化生成的描述为<NULL>

        # 解包参数
        W_proj, b_proj = self.params["W_proj"], self.params["b_proj"]
        W_embed = self.params["W_embed"]
        Wx, Wh, b = self.params["Wx"], self.params["Wh"], self.params["b"]
        W_vocab, b_vocab = self.params["W_vocab"], self.params["b_vocab"]

        #################################################################################################
        # 代办: 实现模型的测试时采样。需要通过对输入图像特征应用学习到的仿射变换来初始化RNN的隐藏状态。       #
        # 传入RNN的第一个词应为<START>标记；其值存储在self._start中。每个时间步需要：                       #
        # (1) 使用学习到的词嵌入对前一个词进行嵌入                                                        #
        # (2) 使用前一个隐藏状态和嵌入的当前词进行RNN步骤，得到下一个隐藏状态                               #
        # (3) 对下一个隐藏状态应用学习到的仿射变换，得到所有词汇的得分                                      #
        # (4) 选择得分最高的词作为下一个词，将其（词索引）写入captions变量的对应位置                        #
        #                                                                                               #
        # 为简单起见，不需要在采样到<END>标记后停止生成，但如果想实现也可以。                               #
        #                                                                                               #
        # 提示：不能使用rnn_forward或lstm_forward函数；需要在循环中调用rnn_step_forward或lstm_step_forward #
        #                                                                                               #
        # 注意：此函数仍在小批量上运行。如果使用LSTM，初始细胞状态应设为0。                                 #
        #################################################################################################
        # 
        ############################################################################
        #                             你的代码结束                                 #
        ############################################################################
        return captions
