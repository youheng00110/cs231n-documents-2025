from __future__ import print_function
import numpy as np

try:
    # 尝试导入Cython编译的im2col和col2im函数（用于加速卷积操作）
    from .im2col_cython import col2im_cython, im2col_cython
    from .im2col_cython import col2im_6d_cython
except ImportError:
    pass
    # 导入失败时的提示（可忽略，若未编译Cython扩展会出现此信息）
    # print("""=========== 若未在ConvolutionalNetworks.ipynb中操作，可安全忽略以下信息 ===========""")
    # print("\t需要编译Cython扩展以完成本作业的部分内容。")
    # print("\t编译说明将在下方的notebook章节中给出。")

from .im2col import *  # 导入im2col相关的辅助函数


def conv_forward_im2col(x, w, b, conv_param):
    """
    基于im2col和col2im的卷积层前向传播快速实现。
    
    输入：
    - x: 输入数据，形状为(N, C, H, W)，其中N是批量大小，C是通道数，H/W是高/宽
    - w: 卷积核，形状为(F, C, HH, WW)，其中F是卷积核数量，HH/WW是卷积核高/宽
    - b: 偏置，形状为(F,)
    - conv_param: 字典，包含卷积参数：
      - 'stride': 步长
      - 'pad': 填充大小
    
    返回：
    - out: 输出特征图，形状为(N, F, H_out, W_out)，其中H_out/W_out是输出高/宽
    - cache: 缓存数据，用于反向传播
    """
    N, C, H, W = x.shape  # 解析输入数据形状
    num_filters, _, filter_height, filter_width = w.shape  # 解析卷积核形状
    stride, pad = conv_param["stride"], conv_param["pad"]  # 解析步长和填充

    # 检查维度是否匹配（确保输入尺寸经过填充和步长后能被卷积核整除）
    assert (W + 2 * pad - filter_width) % stride == 0, "宽度维度不匹配"
    assert (H + 2 * pad - filter_height) % stride == 0, "高度维度不匹配"

    # 计算输出特征图的尺寸
    out_height = (H + 2 * pad - filter_height) // stride + 1
    out_width = (W + 2 * pad - filter_width) // stride + 1
    out = np.zeros((N, num_filters, out_height, out_width), dtype=x.dtype)  # 初始化输出

    # 使用im2col将输入图像转换为列矩阵（加速卷积计算）
    x_cols = im2col_cython(x, w.shape[2], w.shape[3], pad, stride)
    # 卷积操作转化为矩阵乘法：(F, C*HH*WW) * (C*HH*WW, N*H_out*W_out) + 偏置 -> (F, N*H_out*W_out)
    res = w.reshape((w.shape[0], -1)).dot(x_cols) + b.reshape(-1, 1)

    # 调整输出形状为(N, F, H_out, W_out)
    out = res.reshape(w.shape[0], out.shape[2], out.shape[3], x.shape[0])
    out = out.transpose(3, 0, 1, 2)  # 转置维度：(F, H_out, W_out, N) -> (N, F, H_out, W_out)

    cache = (x, w, b, conv_param, x_cols)  # 缓存用于反向传播的数据
    return out, cache


def conv_forward_strides(x, w, b, conv_param):
    """
    基于步长技巧（stride tricks）的卷积层前向传播快速实现。
    通过巧妙设置数组步长，避免显式复制数据，加速im2col转换。
    """
    N, C, H, W = x.shape  # 输入数据形状
    F, _, HH, WW = w.shape  # 卷积核形状（F为卷积核数量）
    stride, pad = conv_param["stride"], conv_param["pad"]  # 步长和填充

    # 对输入进行填充
    p = pad
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode="constant")  # 仅在高和宽方向填充

    # 计算输出特征图尺寸
    H_padded, W_padded = H + 2 * pad, W + 2 * pad  # 填充后的高和宽
    out_h = (H_padded - HH) // stride + 1  # 输出高
    out_w = (W_padded - WW) // stride + 1  # 输出宽

    # 使用步长技巧实现im2col（创建视图而非复制数据）
    # 形状：(C, HH, WW, N, out_h, out_w)
    shape = (C, HH, WW, N, out_h, out_w)
    # 步长：每个维度的元素间隔（基于原始数据的步长计算）
    strides = (H_padded * W_padded, W_padded, 1, C * H_padded * W_padded, stride * W_padded, stride)
    strides = x.itemsize * np.array(strides)  # 转换为字节步长
    # 创建strided数组视图（不复制数据）
    x_stride = np.lib.stride_tricks.as_strided(x_padded, shape=shape, strides=strides)
    x_cols = np.ascontiguousarray(x_stride)  # 确保数组在内存中连续
    x_cols.shape = (C * HH * WW, N * out_h * out_w)  # 重塑为列矩阵

    # 卷积操作：矩阵乘法 + 偏置
    res = w.reshape(F, -1).dot(x_cols) + b.reshape(-1, 1)

    # 调整输出形状为(N, F, out_h, out_w)
    res.shape = (F, N, out_h, out_w)
    out = res.transpose(1, 0, 2, 3)  # 转置维度：(F, N, out_h, out_w) -> (N, F, out_h, out_w)

    # 确保输出在内存中连续
    out = np.ascontiguousarray(out)

    cache = (x, w, b, conv_param, x_cols)  # 缓存数据
    return out, cache


def conv_backward_strides(dout, cache):
    """
    基于步长技巧的卷积层反向传播实现，对应conv_forward_strides的反向计算。
    计算损失对输入x、权重w和偏置b的梯度。
    """
    x, w, b, conv_param, x_cols = cache  # 从缓存中恢复数据
    stride, pad = conv_param["stride"], conv_param["pad"]  # 步长和填充

    N, C, H, W = x.shape  # 输入形状
    F, _, HH, WW = w.shape  # 卷积核形状
    _, _, out_h, out_w = dout.shape  # 输出梯度形状

    # 计算偏置的梯度：对所有批量和空间维度求和
    db = np.sum(dout, axis=(0, 2, 3))

    # 计算权重的梯度：输出梯度与输入列矩阵的转置相乘
    dout_reshaped = dout.transpose(1, 0, 2, 3).reshape(F, -1)  # 重塑为(F, N*out_h*out_w)
    dw = dout_reshaped.dot(x_cols.T).reshape(w.shape)  # 结果重塑为卷积核形状

    # 计算输入的梯度：权重转置与输出梯度相乘，再通过col2im转换回图像形状
    dx_cols = w.reshape(F, -1).T.dot(dout_reshaped)  # (C*HH*WW, N*out_h*out_w)
    dx_cols.shape = (C, HH, WW, N, out_h, out_w)  # 重塑为原始列矩阵的形状
    # 使用6维col2im转换回图像
    dx = col2im_6d_cython(dx_cols, N, C, H, W, HH, WW, pad, stride)

    return dx, dw, db  # 返回输入、权重、偏置的梯度


def conv_backward_im2col(dout, cache):
    """
    基于im2col的卷积层反向传播快速实现，对应conv_forward_im2col的反向计算。
    """
    x, w, b, conv_param, x_cols = cache  # 恢复缓存数据
    stride, pad = conv_param["stride"], conv_param["pad"]  # 步长和填充

    # 计算偏置的梯度：对批量和空间维度求和
    db = np.sum(dout, axis=(0, 2, 3))

    num_filters, _, filter_height, filter_width = w.shape  # 卷积核形状
    # 重塑输出梯度为(F, N*out_h*out_w)
    dout_reshaped = dout.transpose(1, 2, 3, 0).reshape(num_filters, -1)
    # 计算权重梯度：输出梯度与输入列矩阵的转置相乘，再重塑为卷积核形状
    dw = dout_reshaped.dot(x_cols.T).reshape(w.shape)

    # 计算输入梯度：权重转置与输出梯度相乘，再通过col2im转换回图像
    dx_cols = w.reshape(num_filters, -1).T.dot(dout_reshaped)  # (C*HH*WW, N*out_h*out_w)
    # 使用cython的col2im转换回图像形状
    dx = col2im_cython(
        dx_cols,
        x.shape[0],  # N
        x.shape[1],  # C
        x.shape[2],  # H
        x.shape[3],  # W
        filter_height,  # HH
        filter_width,   # WW
        pad,
        stride,
    )

    return dx, dw, db  # 返回输入、权重、偏置的梯度


# 选择卷积前向/反向传播的快速实现（默认使用strides方法）
conv_forward_fast = conv_forward_strides
conv_backward_fast = conv_backward_strides


def max_pool_forward_fast(x, pool_param):
    """
    最大池化层前向传播的快速实现。
    根据池化区域是否为正方形且完整覆盖输入，选择reshape方法（更快）或im2col方法。
    """
    N, C, H, W = x.shape  # 输入形状
    pool_height, pool_width = pool_param["pool_height"], pool_param["pool_width"]  # 池化核尺寸
    stride = pool_param["stride"]  # 池化步长

    # 检查池化参数是否满足reshape方法的条件：池化核为正方形且步长等于核尺寸，且能完整覆盖输入
    same_size = pool_height == pool_width == stride
    tiles = H % pool_height == 0 and W % pool_width == 0
    if same_size and tiles:
        # 使用reshape方法（更快，无需复制数据）
        out, reshape_cache = max_pool_forward_reshape(x, pool_param)
        cache = ("reshape", reshape_cache)
    else:
        # 使用im2col方法（通用性更强）
        out, im2col_cache = max_pool_forward_im2col(x, pool_param)
        cache = ("im2col", im2col_cache)
    return out, cache


def max_pool_backward_fast(dout, cache):
    """
    最大池化层反向传播的快速实现。
    根据前向传播使用的方法（reshape或im2col）选择对应的反向计算。
    """
    method, real_cache = cache  # 从缓存中获取方法和实际缓存数据
    if method == "reshape":
        return max_pool_backward_reshape(dout, real_cache)
    elif method == "im2col":
        return max_pool_backward_im2col(dout, real_cache)
    else:
        raise ValueError('未识别的方法 "%s"' % method)


def max_pool_forward_reshape(x, pool_param):
    """
    基于reshape的最大池化前向传播实现。
    仅适用于池化核为正方形、步长等于核尺寸且完整覆盖输入的情况。
    """
    N, C, H, W = x.shape  # 输入形状
    pool_height, pool_width = pool_param["pool_height"], pool_param["pool_width"]  # 池化核尺寸
    stride = pool_param["stride"]  # 步长

    # 验证reshape方法的适用性
    assert pool_height == pool_width == stride, "池化参数不满足reshape方法要求"
    assert H % pool_height == 0
    assert W % pool_height == 0

    # 重塑输入为(N, C, H/pool_height, pool_height, W/pool_width, pool_width)
    x_reshaped = x.reshape(
        N, C, H // pool_height, pool_height, W // pool_width, pool_width
    )
    # 在池化核的高和宽维度取最大值，得到输出
    out = x_reshaped.max(axis=3).max(axis=4)  # 先沿axis=3（池化核高）取max，再沿axis=4（池化核宽）取max

    cache = (x, x_reshaped, out)  # 缓存数据
    return out, cache


def max_pool_backward_reshape(dout, cache):
    """
    基于reshape的最大池化反向传播实现。
    仅适用于前向传播使用max_pool_forward_reshape的情况。
    
    注意：若存在多个最大值（argmax不唯一），此方法会将梯度分配给所有最大值位置。
    这在实际中影响较小，若需严格分配可取消下方归一化注释，但会降低效率。
    """
    x, x_reshaped, out = cache  # 恢复缓存数据

    dx_reshaped = np.zeros_like(x_reshaped)  # 初始化池化核维度的梯度
    # 扩展out的维度，使其与x_reshaped形状匹配（用于广播）
    out_newaxis = out[:, :, :, np.newaxis, :, np.newaxis]
    # 生成掩码：标记x_reshaped中哪些元素是最大值（前向传播选中的元素）
    mask = x_reshaped == out_newaxis
    # 扩展dout的维度，使其与dx_reshaped形状匹配
    dout_newaxis = dout[:, :, :, np.newaxis, :, np.newaxis]
    # 广播dout到dx_reshaped的形状
    dout_broadcast, _ = np.broadcast_arrays(dout_newaxis, dx_reshaped)
    # 将上游梯度分配给掩码标记的位置
    dx_reshaped[mask] = dout_broadcast[mask]
    # 若存在多个最大值，将梯度平均分配（可选，默认注释以提高效率）
    dx_reshaped /= np.sum(mask, axis=(3, 5), keepdims=True)
    # 重塑回输入x的形状
    dx = dx_reshaped.reshape(x.shape)

    return dx


def max_pool_forward_im2col(x, pool_param):
    """
    基于im2col的最大池化前向传播实现。
    通用性强，但速度不如reshape方法，适用于非正方形池化核或不完整覆盖输入的情况。
    """
    N, C, H, W = x.shape  # 输入形状
    pool_height, pool_width = pool_param["pool_height"], pool_param["pool_width"]  # 池化核尺寸
    stride = pool_param["stride"]  # 步长

    # 检查维度是否匹配
    assert (H - pool_height) % stride == 0, "高度不匹配"
    assert (W - pool_width) % stride == 0, "宽度不匹配"

    # 计算输出尺寸
    out_height = (H - pool_height) // stride + 1
    out_width = (W - pool_width) // stride + 1

    # 将输入按通道拆分，转换为(N*C, 1, H, W)，便于im2col处理
    x_split = x.reshape(N * C, 1, H, W)
    # 使用im2col将池化区域转换为列矩阵
    x_cols = im2col(x_split, pool_height, pool_width, padding=0, stride=stride)
    # 找到每列的最大值索引和最大值
    x_cols_argmax = np.argmax(x_cols, axis=0)  # 最大值索引
    x_cols_max = x_cols[x_cols_argmax, np.arange(x_cols.shape[1])]  # 最大值
    # 调整输出形状为(N, C, out_height, out_width)
    out = x_cols_max.reshape(out_height, out_width, N, C).transpose(2, 3, 0, 1)

    cache = (x, x_cols, x_cols_argmax, pool_param)  # 缓存数据（含最大值索引，用于反向传播）
    return out, cache


def max_pool_backward_im2col(dout, cache):
    """
    基于im2col的最大池化反向传播实现。
    对应max_pool_forward_im2col的反向计算。
    """
    x, x_cols, x_cols_argmax, pool_param = cache  # 恢复缓存数据
    N, C, H, W = x.shape  # 输入形状
    pool_height, pool_width = pool_param["pool_height"], pool_param["pool_width"]  # 池化核尺寸
    stride = pool_param["stride"]  # 步长

    # 重塑输出梯度为(out_height*out_width*N*C,)
    dout_reshaped = dout.transpose(2, 3, 0, 1).flatten()
    # 初始化列矩阵的梯度
    dx_cols = np.zeros_like(x_cols)
    # 将上游梯度分配给前向传播中最大值的位置
    dx_cols[x_cols_argmax, np.arange(dx_cols.shape[1])] = dout_reshaped
    # 使用col2im将列矩阵的梯度转换回图像形状
    dx = col2im_indices(
        dx_cols, (N * C, 1, H, W), pool_height, pool_width, padding=0, stride=stride
    )
    # 重塑回输入x的形状
    dx = dx.reshape(x.shape)

    return dx
