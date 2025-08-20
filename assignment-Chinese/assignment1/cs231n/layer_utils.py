from .layers import *


def affine_relu_forward(x, w, b):
    """
    仿射层后接 ReLU 的前向传播。

    参数
    ----
    x : ndarray
        输入数据。
    w, b : ndarray
        仿射层的权重和偏置。

    返回
    ----
    tuple
        - out : ReLU 的输出。
        - cache : 包含前向传播中间结果的对象，用于反向传播。
    """
    a, fc_cache = affine_forward(x, w, b)  # 仿射层前向传播
    out, relu_cache = relu_forward(a)      # ReLU 层前向传播
    cache = (fc_cache, relu_cache)
    return out, cache


def affine_relu_backward(dout, cache):
    """
    仿射层后接 ReLU 的反向传播。

    参数
    ----
    dout : ndarray
        上游梯度。
    cache : tuple
        包含前向传播中间结果的对象。

    返回
    ----
    tuple
        - dx : 输入 x 的梯度。
        - dw : 权重 w 的梯度。
        - db : 偏置 b 的梯度。
    """
    fc_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)  # ReLU 层反向传播
    dx, dw, db = affine_backward(da, fc_cache)  # 仿射层反向传播
    return dx, dw, db

