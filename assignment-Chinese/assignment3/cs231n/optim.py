"""
该文件实现了各种常用于训练神经网络的一阶更新规则。
每个更新规则接收当前权重和损失相对于这些权重的梯度，
并生成下一组权重。每个更新规则具有相同的接口：

def update(w, dw, config=None):

输入：
  - w: 给出当前权重的numpy数组。
  - dw: 与w形状相同的numpy数组，给出损失相对于w的梯度。
  - config: 包含超参数值的字典，如学习率、动量等。
    如果更新规则需要在多次迭代中缓存值，那么config也会保存这些缓存值。

返回：
  - next_w: 更新后的点。
  - config: 要传递给更新规则下一次迭代的配置字典。

注意：对于大多数更新规则，默认学习率可能表现不佳；
然而，其他超参数的默认值应该适用于各种不同的问题。

为了提高效率，更新规则可能会执行原地更新，修改w并将next_w设置为w。
"""

import numpy as np


def sgd(w, dw, config=None):
    """
    执行标准随机梯度下降。

    配置格式：
    - learning_rate: 标量学习率。
    """
    if config is None:
        config = {}
    # 设置默认学习率为0.01
    config.setdefault("learning_rate", 1e-2)

    # 执行SGD更新：w = w - learning_rate * dw
    w -= config["learning_rate"] * dw
    return w, config


def sgd_momentum(w, dw, config=None):
    """
    执行带动量的随机梯度下降。

    配置格式：
    - learning_rate: 标量学习率。
    - momentum: 0到1之间的标量，给出动量值。
      设置momentum = 0则退化为标准SGD。
    - velocity: 与w和dw形状相同的numpy数组，用于存储梯度的移动平均值。
    """
    if config is None:
        config = {}
    # 设置默认超参数
    config.setdefault("learning_rate", 1e-2)
    config.setdefault("momentum", 0.9)
    # 获取速度缓存，默认为与w相同形状的零数组
    v = config.get("velocity", np.zeros_like(w))

    next_w = None
    # 更新速度：v = momentum * v - learning_rate * dw
    v *= config["momentum"]
    v -= config["learning_rate"] * dw
    # 更新权重：w = w + v
    w += v
    next_w = w
    # 保存当前速度到配置中
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
    # 设置默认超参数
    config.setdefault("learning_rate", 1e-2)
    config.setdefault("decay_rate", 0.99)
    config.setdefault("epsilon", 1e-8)
    # 获取缓存，默认为与w相同形状的零数组
    config.setdefault("cache", np.zeros_like(w))

    next_w = None
    rho = config["decay_rate"]  # 衰减率
    lr = config["learning_rate"]  # 学习率
    eps = config["epsilon"]  # 平滑项
    # 更新缓存：cache = rho * cache + (1 - rho) * dw^2
    config["cache"] *= rho
    config["cache"] += (1.0 - rho) * dw ** 2
    # 计算更新步长：-learning_rate * dw / (sqrt(cache) + epsilon)
    step = -(lr * dw) / (np.sqrt(config["cache"]) + eps)
    # 更新权重
    w += step
    next_w = w

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
    # 设置默认超参数
    config.setdefault("learning_rate", 1e-3)
    config.setdefault("beta1", 0.9)
    config.setdefault("beta2", 0.999)
    config.setdefault("epsilon", 1e-8)
    # 初始化一阶矩、二阶矩和迭代次数
    config.setdefault("m", np.zeros_like(w))
    config.setdefault("v", np.zeros_like(w))
    config.setdefault("t", 0)

    next_w = None
    beta1, beta2, eps = config["beta1"], config["beta2"], config["epsilon"]
    t, m, v = config["t"], config["m"], config["v"]
    # 更新一阶矩估计：m = beta1 * m + (1 - beta1) * dw
    m = beta1 * m + (1 - beta1) * dw
    # 更新二阶矩估计：v = beta2 * v + (1 - beta2) * dw^2
    v = beta2 * v + (1 - beta2) * (dw * dw)
    # 迭代次数加1
    t += 1
    # 计算偏差校正后的学习率
    alpha = config["learning_rate"] * np.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)
    # 更新权重：w = w - alpha * (m / (sqrt(v) + epsilon))
    w -= alpha * (m / (np.sqrt(v) + eps))
    # 保存更新后的参数到配置中
    config["t"] = t
    config["m"] = m
    config["v"] = v
    next_w = w

    return next_w, config