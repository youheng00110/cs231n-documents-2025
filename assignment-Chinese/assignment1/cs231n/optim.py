import numpy as np

"""
该文件实现了各种一阶更新规则，这些规则通常用于训练神经网络。
每个更新规则接收当前权重和损失相对于这些权重的梯度，并生成下一组权重。
每个更新规则具有相同的接口：

def update(w, dw, config=None):

输入：
  - w: 给出当前权重的numpy数组。
  - dw: 与w形状相同的numpy数组，给出损失相对于w的梯度。
  - config: 包含超参数值的字典，如学习率、动量等。如果更新规则需要在多次迭代中缓存值，
    则config也将保存这些缓存值。

返回：
  - next_w: 更新后的点。
  - config: 要传递给更新规则下一次迭代的配置字典。

注意：对于大多数更新规则，默认学习率可能表现不佳；然而其他超参数的默认值
应该适用于各种不同的问题。

为提高效率，更新规则可能执行原地更新，修改w并将next_w设置为w。
"""


def sgd(w, dw, config=None):
    """
    执行标准随机梯度下降（vanilla stochastic gradient descent）。

    配置格式：
    - learning_rate: 标量学习率。
    """
    if config is None:
        config = {}
    # 设置默认学习率为0.01
    config.setdefault("learning_rate", 1e-2)

    # 原地更新权重：w = w - 学习率 * 梯度
    w -= config["learning_rate"] * dw
    return w, config


def sgd_momentum(w, dw, config=None):
    """
    执行带动量的随机梯度下降。

    配置格式：
    - learning_rate: 标量学习率。
    - momentum: 0到1之间的标量，给出动量值。将momentum设置为0则退化为标准SGD。
    - velocity: 与w和dw形状相同的numpy数组，用于存储梯度的移动平均值。
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-2)  # 默认学习率
    config.setdefault("momentum", 0.9)  # 默认动量值
    # 初始化速度v（若未提供则为与w同形状的零数组）
    v = config.get("velocity", np.zeros_like(w))

    next_w = None
    ###############################################################################
    # 待办：实现动量更新公式。将更新后的值存储在next_w变量中。你还需要使用并更新速度v。#
    ###############################################################################

    ###########################################################################
    #                             你的代码结束                                #
    ###########################################################################
    # 更新配置中的速度
    config["velocity"] = v

    return next_w, config


def rmsprop(w, dw, config=None):
    """
    使用RMSProp更新规则，该规则使用梯度平方的移动平均值来设置自适应的每个参数的学习率。

    配置格式：
    - learning_rate: 标量学习率。
    - decay_rate: 0到1之间的标量，给出平方梯度缓存的衰减率。
    - epsilon: 用于平滑以避免除零的小标量。
    - cache: 梯度二阶矩的移动平均值。
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-2)  # 默认学习率
    config.setdefault("decay_rate", 0.99)  # 默认衰减率
    config.setdefault("epsilon", 1e-8)  # 默认epsilon
    # 初始化缓存（若未提供则为与w同形状的零数组）
    config.setdefault("cache", np.zeros_like(w))

    next_w = None
    #############################################################################
    # 待办：实现RMSprop更新公式，将w的下一个值存储在next_w变量中。不要忘记更新存储在#
    # config['cache']中的缓存值。                                                #
    #############################################################################

    ###########################################################################
    #                             你的代码结束                                #
    ###########################################################################

    return next_w, config


def adam(w, dw, config=None):
    """
    使用Adam更新规则，该规则结合了梯度及其平方的移动平均值和偏差校正项。

    配置格式：
    - learning_rate: 标量学习率。
    - beta1: 梯度一阶矩移动平均值的衰减率。
    - beta2: 梯度二阶矩移动平均值的衰减率。
    - epsilon: 用于平滑以避免除零的小标量。
    - m: 梯度的移动平均值。
    - v: 梯度平方的移动平均值。
    - t: 迭代次数。
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-3)  # 默认学习率
    config.setdefault("beta1", 0.9)  # 默认beta1
    config.setdefault("beta2", 0.999)  # 默认beta2
    config.setdefault("epsilon", 1e-8)  # 默认epsilon
    # 初始化m、v（若未提供则为与w同形状的零数组）
    config.setdefault("m", np.zeros_like(w))
    config.setdefault("v", np.zeros_like(w))
    config.setdefault("t", 0)  # 初始化迭代次数

    next_w = None
    ###########################################################################
    # 待办：实现Adam更新公式，将w的下一个值存储在next_w变量中。不要忘记更新存储在 #
    # config中的m、v和t变量。                                                  #
    #                                                                         #
    # 注意：为了与参考输出匹配，请在任何计算中使用t之前修改t。                   #
    ###########################################################################

    ###########################################################################
    #                             你的代码结束                                #
    ###########################################################################

    return next_w, config
