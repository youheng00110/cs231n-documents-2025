from .layers import *
from .fast_layers import *  # 导入快速实现的层（如快速卷积、池化等）


def affine_relu_forward(x, w, b):
    """
    一个便捷层，先执行仿射变换（全连接），再执行ReLU激活。

    输入：
    - x: 仿射层的输入
    - w, b: 仿射层的权重和偏置

    返回：
    - out: ReLU的输出
    - cache: 用于反向传播的缓存数据
    """
    a, fc_cache = affine_forward(x, w, b)  # 仿射变换：a = x·w + b
    out, relu_cache = relu_forward(a)  # ReLU激活：out = max(0, a)
    cache = (fc_cache, relu_cache)  # 缓存仿射层和ReLU层的中间数据
    return out, cache


def affine_relu_backward(dout, cache):
    """
    affine-relu便捷层的反向传播
    """
    fc_cache, relu_cache = cache  # 从缓存中提取仿射层和ReLU层的数据
    da = relu_backward(dout, relu_cache)  # ReLU反向传播：计算损失对a的梯度
    dx, dw, db = affine_backward(da, fc_cache)  # 仿射层反向传播：计算损失对x、w、b的梯度
    return dx, dw, db


def conv_relu_forward(x, w, b, conv_param):
    """
    一个便捷层，先执行卷积操作，再执行ReLU激活。

    输入：
    - x: 卷积层的输入
    - w, b, conv_param: 卷积层的权重、偏置和参数（步长、填充等）

    返回：
    - out: ReLU的输出
    - cache: 用于反向传播的缓存数据
    """
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)  # 快速卷积：a = conv(x, w, b)
    out, relu_cache = relu_forward(a)  # ReLU激活：out = max(0, a)
    cache = (conv_cache, relu_cache)  # 缓存卷积层和ReLU层的中间数据
    return out, cache


def conv_relu_backward(dout, cache):
    """
    conv-relu便捷层的反向传播
    """
    conv_cache, relu_cache = cache  # 从缓存中提取卷积层和ReLU层的数据
    da = relu_backward(dout, relu_cache)  # ReLU反向传播：计算损失对a的梯度
    dx, dw, db = conv_backward_fast(da, conv_cache)  # 快速卷积反向传播：计算损失对x、w、b的梯度
    return dx, dw, db


def conv_bn_relu_forward(x, w, b, gamma, beta, conv_param, bn_param):
    """
    一个便捷层，依次执行卷积、空间批归一化（BN）和ReLU激活。
    
    输入：
    - x: 卷积层输入
    - w, b: 卷积层权重和偏置
    - gamma, beta: 批归一化的缩放和偏移参数
    - conv_param: 卷积参数
    - bn_param: 批归一化参数
    
    返回：
    - out: ReLU输出
    - cache: 缓存数据（卷积、BN、ReLU的中间结果）
    """
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)  # 卷积：a = conv(x, w, b)
    an, bn_cache = spatial_batchnorm_forward(a, gamma, beta, bn_param)  # 空间BN：an = BN(a, gamma, beta)
    out, relu_cache = relu_forward(an)  # ReLU激活：out = max(0, an)
    cache = (conv_cache, bn_cache, relu_cache)  # 缓存三层的中间数据
    return out, cache


def conv_bn_relu_backward(dout, cache):
    """
    conv-bn-relu便捷层的反向传播
    """
    conv_cache, bn_cache, relu_cache = cache  # 提取三层缓存数据
    dan = relu_backward(dout, relu_cache)  # ReLU反向传播：损失对an的梯度
    da, dgamma, dbeta = spatial_batchnorm_backward(dan, bn_cache)  # BN反向传播：损失对a、gamma、beta的梯度
    dx, dw, db = conv_backward_fast(da, conv_cache)  # 卷积反向传播：损失对x、w、b的梯度
    return dx, dw, db, dgamma, dbeta


def conv_relu_pool_forward(x, w, b, conv_param, pool_param):
    """
    一个便捷层，依次执行卷积、ReLU激活和池化操作。

    输入：
    - x: 卷积层的输入
    - w, b, conv_param: 卷积层的权重、偏置和参数
    - pool_param: 池化层的参数（池化核大小、步长等）

    返回：
    - out: 池化层的输出
    - cache: 用于反向传播的缓存数据
    """
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)  # 卷积：a = conv(x, w, b)
    s, relu_cache = relu_forward(a)  # ReLU激活：s = max(0, a)
    out, pool_cache = max_pool_forward_fast(s, pool_param)  # 快速最大池化：out = pool(s)
    cache = (conv_cache, relu_cache, pool_cache)  # 缓存卷积、ReLU、池化层的中间数据
    return out, cache


def conv_relu_pool_backward(dout, cache):
    """
    conv-relu-pool便捷层的反向传播
    """
    conv_cache, relu_cache, pool_cache = cache  # 从缓存中提取三层的数据
    ds = max_pool_backward_fast(dout, pool_cache)  # 池化反向传播：计算损失对s的梯度
    da = relu_backward(ds, relu_cache)  # ReLU反向传播：计算损失对a的梯度
    dx, dw, db = conv_backward_fast(da, conv_cache)  # 卷积反向传播：计算损失对x、w、b的梯度
    return dx, dw, db
