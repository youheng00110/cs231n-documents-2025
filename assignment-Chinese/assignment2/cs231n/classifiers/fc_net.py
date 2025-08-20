from builtins import range
from builtins import object
import numpy as np

from ..layers import *
from ..layer_utils import *


class FullyConnectedNet(object):
    """多层全连接神经网络类。

    该网络包含任意数量的隐藏层、ReLU非线性激活函数和softmax损失函数。
    它还可以选择实现dropout（丢弃法）和batch/layer normalization（批归一化/层归一化）。
    对于一个有L层的网络，其架构为：

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    其中批归一化/层归一化和dropout是可选的，且{...}块会重复(L - 1)次。

    可学习参数存储在self.params字典中，将通过Solver类进行学习。
    """

    def __init__(
        self,
        hidden_dims,
        input_dim=3 * 32 * 32,
        num_classes=10,
        dropout_keep_ratio=1,
        normalization=None,
        reg=0.0,
        weight_scale=1e-2,
        dtype=np.float32,
        seed=None,
    ):
        """初始化一个新的全连接网络。

        输入：
        - hidden_dims: 整数列表，给出每个隐藏层的大小。
        - input_dim: 整数，给出输入的大小。
        - num_classes: 整数，给出要分类的类别数量。
        - dropout_keep_ratio: 0到1之间的标量，给出dropout的强度。
          如果dropout_keep_ratio=1，则网络不使用dropout。
        - normalization: 网络应使用的归一化类型。有效值为"batchnorm"、"layernorm"，
          或None（默认值，不使用归一化）。
        - reg: 标量，给出L2正则化强度。
        - weight_scale: 标量，给出权重随机初始化的标准差。
        - dtype: numpy数据类型对象；所有计算将使用此数据类型。float32更快但精度较低，
          因此数值梯度检查应使用float64。
        - seed: 如果不为None，则将此随机种子传递给dropout层。这将使dropout层具有确定性，
          以便我们可以对模型进行梯度检查。
        """
        self.normalization = normalization
        self.use_dropout = dropout_keep_ratio != 1  # 是否使用dropout
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)  # 总层数 = 1（输出层） + 隐藏层数量
        self.dtype = dtype
        self.params = {}  # 存储网络参数的字典

        #######################################################################################
        # 代办: 初始化网络参数，将所有值存储在self.params字典中。第一层的权重和偏置存储在W1和b1中  #
        # 第二层使用W2和b2，依此类推。权重应从以0为中心、标准差等于weight_scale的正态分布初始化。  #
        # 偏置应初始化为零。                                                                   #
        #                                                                                     #
        # 当使用批归一化时，第一层的缩放和偏移参数存储在gamma1和beta1中；第二层使用gamma2和beta2，#
        # 依此类推。缩放参数应初始化为1，偏移参数应初始化为0。                                   #
        ######################################################################################
        # 
        ############################################################################
        #                             你的代码结束                                 #
        ############################################################################

        # 当使用dropout时，我们需要向每个dropout层传递一个dropout_param字典，
        # 以便该层知道dropout概率和模式（训练/测试）。可以向每个dropout层传递相同的dropout_param。
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {"mode": "train", "p": dropout_keep_ratio}
            if seed is not None:
                self.dropout_param["seed"] = seed  # 设置随机种子以保证确定性

        # 使用批归一化时，我们需要跟踪运行均值和方差，
        # 因此需要向每个批归一化层传递一个特殊的bn_param对象。
        # 应将self.bn_params[0]传递给第一个批归一化层的前向传播，
        # self.bn_params[1]传递给第二个批归一化层的前向传播，依此类推。
        self.bn_params = []
        if self.normalization == "batchnorm":
            self.bn_params = [{"mode": "train"} for i in range(self.num_layers - 1)]
        if self.normalization == "layernorm":
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # 将所有参数转换为正确的数据类型
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """计算全连接网络的损失和梯度。
        
        输入：
        - X: 输入数据数组，形状为(N, d_1, ..., d_k)
        - y: 标签数组，形状为(N,)。y[i]给出X[i]的标签。

        返回：
        如果y为None，则运行模型的测试时前向传播并返回：
        - scores: 形状为(N, C)的数组，给出分类得分，其中scores[i, c]是X[i]对类别c的分类得分。

        如果y不为None，则运行训练时的前向和反向传播，并返回一个元组：
        - loss: 标量损失值
        - grads: 与self.params具有相同键的字典，将参数名称映射到损失相对于这些参数的梯度。
        """
        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"  # 根据y是否为None判断是测试还是训练模式

        # 为批归一化参数和dropout参数设置训练/测试模式，因为它们在训练和测试时的行为不同
        if self.use_dropout:
            self.dropout_param["mode"] = mode
        if self.normalization == "batchnorm":
            for bn_param in self.bn_params:
                bn_param["mode"] = mode
        scores = None
        ##############################################################################
        # 代办: 实现全连接网络的前向传播，计算X的类别得分并存储在scores变量中。         #
        #                                                                           #
        # 使用dropout时，需要向每个dropout前向传播传递self.dropout_param。            #
        #                                                                           #
        # 使用批归一化时，需要向第一个批归一化层的前向传播传递self.bn_params[0]，向第二个#
        # 批归一化层的前向传播传递self.bn_params[1]，依此类推。                       #
        ############################################################################
        # 
        ############################################################################
        #                             你的代码结束                                 #
        ############################################################################

        # 如果是测试模式，提前返回
        if mode == "test":
            return scores

        loss, grads = 0.0, {}
        ######################################################################################
        # 代办: 实现全连接网络的反向传播。将损失存储在loss变量中，梯度存储在grads字典中。        #
        # 使用softmax计算数据损失，并确保grads[k]存储self.params[k]的梯度。不要忘记添加L2正则化！#
        #                                                                                    #
        # 使用批归一化/层归一化时，不需要正则化缩放和偏移参数。                                 #
        #                                                                                    #
        # 注意：为确保你的实现与我们的一致并通过自动测试，请确保你的L2正则化包含0.5的因子，      #
        # 以简化梯度的表达式。                                                               #
        ####################################################################################
        # 
        ############################################################################
        #                             你的代码结束                                 #
        ############################################################################

        return loss, grads