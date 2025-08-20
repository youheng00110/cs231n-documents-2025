from builtins import range
from builtins import object
import os
import numpy as np

from ..layers import *
from ..layer_utils import *


class TwoLayerNet(object):
    """
    一个两层全连接神经网络，使用 ReLU 非线性激活函数和 Softmax 损失函数，采用模块化层设计。
    假设输入维度为 D，隐藏层维度为 H，对 C 个类别进行分类。

    网络架构应为：仿射 - ReLU - 仿射 - Softmax。

    注意：该类不实现梯度下降；相反，它将与一个单独的 Solver 对象交互，
    Solver 负责运行优化。

    模型的可学习参数存储在 self.params 字典中，该字典将参数名称映射到 NumPy 数组。
    """

    def __init__(
        self,
        input_dim=3 * 32 * 32,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
    ):
        """
        初始化一个新的网络。

        输入：
        - input_dim: 一个整数，给出输入的大小
        - hidden_dim: 一个整数，给出隐藏层的大小
        - num_classes: 一个整数，给出要分类的类别数
        - weight_scale: 标量，给出随机初始化权重的标准差
        - reg: 标量，给出 L2 正则化强度
        """
        self.params = {}
        self.reg = reg

        #############################################################################
        # 代办: 初始化两层网络的权重和偏置。权重应从均值为 0.0、标准差等于 weight_scale #
        # 的高斯分布中初始化，偏置应初始化为零。所有权重和偏置都应存储在 self.params     #
        # 字典中，第一层权重和偏置使用键 'W1' 和 'b1'，第二层权重和偏置使用键 'W2' 和    #
        # 'b2'。                                                                      #
        ###############################################################################
        
        ############################################################################
        #                             你的代码结束                                  #
        ############################################################################

    def loss(self, X, y=None):
        """
        计算一个小批量数据的损失和梯度。

        输入：
        - X: 输入数据的数组，形状为 (N, d_1, ..., d_k)
        - y: 标签数组，形状为 (N,)。y[i] 给出 X[i] 的标签。

        返回：
        如果 y 为 None，则运行测试时前向传播并返回：
        - scores: 形状为 (N, C) 的数组，给出分类分数，其中 scores[i, c] 是 X[i] 和类别 c
          的分类分数。

        如果 y 不为 None，则运行训练时前向和反向传播并返回一个元组：
        - loss: 给出损失的标量值
        - grads: 与 self.params 有相同键的字典，将参数名称映射到损失相对于这些参数的梯度。
        """
        scores = None
        ##############################################################################
        # 代办: 实现两层网络的前向传播，计算 X 的类别分数，并将它们存储在 scores 变量中。#
        ##############################################################################

        ############################################################################
        #                             你的代码结束                               #
        ############################################################################

        # 如果 y 为 None，则我们处于测试模式，直接返回分数
        if y is None:
            return scores

        loss, grads = 0, {}
        ##################################################################################
        # 代办: 实现两层网络的反向传播。将损失存储在 loss 变量中，梯度存储在 grads 字典中。  #
        # 计算数据损失时使用 Softmax，并确保 grads[k] 持有相对于 self.params[k] 的损失梯度。#
        # 不要忘记添加 L2 正则化！                                                        #
        #                                                                               #
        # 注意：为了确保你的实现与我们的匹配，并且你通过了自动化测试，确保你的 L2 正则化     #
        # 包含一个 0.5 的因子，以简化梯度表达式。                                         #
        #################################################################################

        ############################################################################
        #                             你的代码结束                                  #
        ############################################################################

        return loss, grads

    def save(self, fname):
        """
        保存模型参数。
        """
        fpath = os.path.join(os.path.dirname(__file__), "../saved/", fname)
        params = self.params
        np.save(fpath, params)
        print(fname, "已保存。")
    
    def load(self, fname):
        """
        加载模型参数。
        """
        fpath = os.path.join(os.path.dirname(__file__), "../saved/", fname)
        if not os.path.exists(fpath):
            print(fname, "不可用。")
            return False
        else:
            params = np.load(fpath, allow_pickle=True).item()
            self.params = params
            print(fname, "已加载。")
            return True



class FullyConnectedNet(object):
    """
    多层全连接神经网络类。

    网络包含任意数量的隐藏层、ReLU 非线性激活函数和 Softmax 损失函数。此外，还提供了 Dropout 和批量/层归一化作为选项。
    对于一个有 L 层的网络，架构为：

    {仿射 - [批量/层归一化] - ReLU - [Dropout]} x (L - 1) - 仿射 - Softmax

    其中，批量/层归一化和 Dropout 是可选的，{...} 块重复 L - 1 次。

    可学习参数存储在 self.params 字典中，将通过 Solver 类进行学习。
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
        """
        初始化一个新的 FullyConnectedNet。

        输入：
        - hidden_dims: 一个整数列表，给出每个隐藏层的大小。
        - input_dim: 一个整数，给出输入的大小。
        - num_classes: 一个整数，给出要分类的类别数。
        - dropout_keep_ratio: 介于 0 和 1 之间的标量，给出 Dropout 强度。
            如果 dropout_keep_ratio=1，则网络不应使用 Dropout。
        - normalization: 网络应使用的归一化类型。有效值为 "batchnorm"、"layernorm" 或 None（默认值，不使用归一化）。
        - reg: 标量，给出 L2 正则化强度。
        - weight_scale: 标量，给出随机初始化权重的标准差。
        - dtype: 一个 NumPy 数据类型对象；所有计算都将使用此数据类型。float32 速度更快，但精度较低，
            因此在进行数值梯度检查时应使用 float64。
        - seed: 如果不为 None，则将此随机种子传递给 Dropout 层。
            这将使 Dropout 层具有确定性，以便我们可以对模型进行梯度检查。
        """
        self.normalization = normalization
        self.use_dropout = dropout_keep_ratio != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ##############################################################################
        # 代办: 初始化网络的参数，将所有值存储在 self.params 字典中。将第一层的权重和偏置#
        # 存储在 W1 和 b1 中；第二层存储在 W2 和 b2 中，依此类推。权重应从均值为 0 的正态#
        # 分布中初始化，标准差等于 weight_scale。偏置应初始化为零。                     #
        #                                                                            #
        # 当使用批量归一化时，应将比例和偏移参数存储在第一层的 gamma1 和 beta1 中；第二  #
        # 层存储在 gamma2 和 beta2 中，依此类推。比例参数应初始化为一，偏移参数应初始化  #
        # 为零。                                                                     #
        ##############################################################################

        ############################################################################
        #                             你的代码结束                                  #
        ############################################################################

        # 当使用 Dropout 时，我们需要将一个 dropout_param 字典传递给每个 Dropout 层，
        # 以便 Dropout 层知道 Dropout 概率和模式（训练 / 测试）。
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {"mode": "train", "p": dropout_keep_ratio}
            if seed is not None:
                self.dropout_param["seed"] = seed

        # 当使用批量归一化时，我们需要跟踪运行时均值和方差，
        # 因此我们需要为每个批量归一化层传递一个特殊的 bn_param 对象。
        # 你应该将 self.bn_params[0] 传递给第一个批量归一化层的前向传播，
        # self.bn_params[1] 传递给第二个批量归一化层的前向传播，依此类推。
        self.bn_params = []
        if self.normalization == "batchnorm":
            self.bn_params = [{"mode": "train"} for i in range(self.num_layers - 1)]
        if self.normalization == "layernorm":
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # 将所有参数转换为正确的数据类型。
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        计算全连接网络的损失和梯度。
        
        输入：
        - X: 输入数据的数组，形状为 (N, d_1, ..., d_k)
        - y: 标签数组，形状为 (N,)。y[i] 给出 X[i] 的标签。

        返回：
        如果 y 为 None，则运行测试时前向传播并返回：
        - scores: 形状为 (N, C) 的数组，给出分类分数，其中 scores[i, c] 是 X[i] 和类别 c
          的分类分数。

        如果 y 不为 None，则运行训练时前向和反向传播并返回一个元组：
        - loss: 给出损失的标量值
        - grads: 与 self.params 有相同键的字典，将参数名称映射到损失相对于这些参数的梯度。
        """
        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"

        # 设置批量归一化参数和 Dropout 参数的训练 / 测试模式，因为它们在训练和测试时的行为不同。
        if self.use_dropout:
            self.dropout_param["mode"] = mode
        if self.normalization == "batchnorm":
            for bn_param in self.bn_params:
                bn_param["mode"] = mode
        scores = None
        ################################################################################
        # 代办: 实现全连接网络的前向传播，计算 X 的类别分数，并将它们存储在 scores 变量中。#
        #                                                                              #
        # 当使用 Dropout 时，你需要将 self.dropout_param 传递给每个 Dropout 前向传播。    #
        #                                                                              # 
        # 当使用批量归一化时，你需要将 self.bn_params[0] 传递给第一个批量归一化层的前向    #
        # 传播，self.bn_params[1] 传递给第二个批量归一化层的前向传播，依此类推。          #
        ###############################################################################

        ############################################################################
        #                             你的代码结束                                  #
        ############################################################################

        # 如果测试模式则提前返回。
        if mode == "test":
            return scores

        loss, grads = 0.0, {}
        ##################################################################################
        # 代办: 实现全连接网络的反向传播。将损失存储在 loss 变量中，梯度存储在 grads 字典中。#
        # 计算数据损失时使用 Softmax，并确保 grads[k] 持有相对于 self.params[k] 的损失梯度。#
        # 不要忘记添加 L2 正则化！                                                        #
        #                                                                                #
        # 当使用批量 / 层归一化时，你不需要正则化比例和偏移参数。                           #
        #                                                                               #
        # 注意：为了确保你的实现与我们的匹配，并且你通过了自动化测试，确保你的 L2 正则化     #
        # 包含一个 0.5 的因子，以简化梯度表达式。                                         #
        #################################################################################

        ############################################################################
        #                             你的代码结束                                  #
        ############################################################################

        return loss, grads


    def save(self, fname):
        """
        保存模型参数。
        """
        fpath = os.path.join(os.path.dirname(__file__), "../saved/", fname)
        params = self.params
        np.save(fpath, params)
        print(fname, "已保存。")
    
    def load(self, fname):
        """
        加载模型参数。
        """
        fpath = os.path.join(os.path.dirname(__file__), "../saved/", fname)
        if not os.path.exists(fpath):
            print(fname, "不可用。")
            return False
        else:
            params = np.load(fpath, allow_pickle=True).item()
            self.params = params
            print(fname, "已加载。")
            return True
        





      

    
        
        
        
       
        
        
        
        
        
        
