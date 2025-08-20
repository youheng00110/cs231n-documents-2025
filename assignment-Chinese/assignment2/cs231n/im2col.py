from builtins import range
import numpy as np


def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):
    """
    计算im2col操作中用于索引的坐标。
    
    输入：
    - x_shape: 输入数据的形状，格式为(N, C, H, W)
    - field_height: 卷积核/感受野的高度
    - field_width: 卷积核/感受野的宽度
    - padding: 填充大小
    - stride: 步长
    
    返回：
    - 三个数组(k, i, j)，分别用于索引通道、高度和宽度维度
    """
    # 首先确定输出的尺寸
    N, C, H, W = x_shape
    # 验证维度是否匹配（确保填充和步长设置合理）
    assert (H + 2 * padding - field_height) % stride == 0
    assert (W + 2 * padding - field_height) % stride == 0  # 原文此处应为field_width，可能是笔误
    # 计算输出特征图的高度和宽度
    out_height = (H + 2 * padding - field_height) // stride + 1
    out_width = (W + 2 * padding - field_width) // stride + 1

    # 生成感受野在高度方向的基础索引（0到field_height-1，重复field_width次）
    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, C)  # 按通道数复制（每个通道的感受野索引相同）
    # 生成感受野在高度方向的偏移量（由步长和输出高度决定）
    i1 = stride * np.repeat(np.arange(out_height), out_width)

    # 生成感受野在宽度方向的基础索引（0到field_width-1，每个值重复field_height次，再按通道数复制）
    j0 = np.tile(np.arange(field_width), field_height * C)
    # 生成感受野在宽度方向的偏移量（由步长和输出宽度决定）
    j1 = stride * np.tile(np.arange(out_width), out_height)

    # 组合基础索引和偏移量，得到完整的高度和宽度索引
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    # 生成通道索引（每个通道重复field_height*field_width次）
    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

    return (k, i, j)


def im2col_indices(x, field_height, field_width, padding=1, stride=1):
    """ 基于高级索引的im2col实现（将图像转换为列矩阵） """
    # 对输入进行零填充
    p = padding
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode="constant")

    # 获取索引
    k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding, stride)

    # 使用索引提取感受野并转换为列矩阵
    cols = x_padded[:, k, i, j]
    C = x.shape[1]  # 通道数
    # 调整维度顺序并重塑为 (C*field_height*field_width, 输出元素总数)
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
    return cols


def col2im_indices(cols, x_shape, field_height=3, field_width=3, padding=1, stride=1):
    """ 基于高级索引和np.add.at的col2im实现（将列矩阵转换回图像） """
    N, C, H, W = x_shape  # 原始输入形状
    H_padded, W_padded = H + 2 * padding, W + 2 * padding  # 填充后的形状
    # 初始化填充后的图像
    x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
    # 获取索引
    k, i, j = get_im2col_indices(x_shape, field_height, field_width, padding, stride)

    # 重塑列矩阵并调整维度顺序
    cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)  # 转换为(N, C*field_height*field_width, 输出元素总数)

    # 将列矩阵的值添加回填充后的图像（重叠区域累加）
    np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)

    # 去除填充部分
    if padding == 0:
        return x_padded
    return x_padded[:, :, padding:-padding, padding:-padding]


# ~~START DELETE~~（以下为辅助函数和朴素实现，可能用于教学或对比）
def get_num_fields(x_shape, field_height, field_width, padding, stride):
    """
    辅助函数：计算水平和垂直方向上感受野的数量。

    输入：
    - x_shape: 输入形状的4元组 (N, C, H, W)
    - field_height: 感受野高度
    - field_width: 感受野宽度
    - padding: 填充大小
    - stride: 步长

    返回：
    - 元组 (HH, WW)，分别表示垂直和水平方向上的感受野数量
    """
    N, C, H, W = x_shape
    if (W + 2 * padding - field_width) % stride != 0:
        raise ValueError("im2col参数无效；宽度不匹配")
    if (H + 2 * padding - field_height) % stride != 0:
        raise ValueError("im2col参数无效；高度不匹配")

    # 计算水平和垂直方向的感受野数量
    WW = (W + 2 * padding - field_width) // stride + 1
    HH = (H + 2 * padding - field_height) // stride + 1

    return HH, WW


def field_coords(H, W, field_height, field_width, padding, stride):
    """
    生成器：按顺序生成感受野的坐标。
    遍历顺序为从左到右、从上到下（类似阅读顺序，左上角为(0,0)）。

    遍历方式示例：
    for y0, y1, x0, x1 in field_coords(*args):
      # 处理感受野坐标 (y0, y1, x0, x1)

    输入：
    - H: 输入高度
    - W: 输入宽度
    - field_height: 感受野高度
    - field_width: 感受野宽度
    - padding: 填充大小
    - stride: 步长

    生成：
    - 元组 (y0, y1, x0, x1)，表示感受野的坐标范围（左闭右开）
    """
    if (W + 2 * padding - field_width) % stride != 0:
        raise ValueError("field_coords参数无效；宽度不匹配")
    if (H + 2 * padding - field_height) % stride != 0:
        raise ValueError("field_coords参数无效；高度不匹配")
    # 遍历垂直方向的感受野
    yy = 0
    while stride * yy + field_height <= H + 2 * padding:
        y0 = yy * stride  # 感受野顶部坐标
        y1 = yy * stride + field_height  # 感受野底部坐标（开区间）
        # 遍历水平方向的感受野
        xx = 0
        while stride * xx + field_width <= W + 2 * padding:
            x0 = xx * stride  # 感受野左部坐标
            x1 = xx * stride + field_width  # 感受野右部坐标（开区间）
            yield (y0, y1, x0, x1)
            xx += 1
        yy += 1


def im2col_naive(x, field_height=3, field_width=3, padding=1, stride=1):
    """
    将独立的3D输入组成的4D数组转换为单列矩阵，其中每列是一个输入的感受野。

    输入x的形状为(N, C, H, W)，表示N个独立输入，每个输入有高度H、宽度W和C个通道。

    单个输入的感受野是一个覆盖所有通道的矩形块，其高度和宽度由field_height和field_width指定。
    感受野按步长stride在输入上滑动。提取感受野前，会在输入的四周进行零填充。

    此函数用于高效实现卷积层和池化层。

    简单示例：对以下矩阵使用field_height=field_width=2、padding=1、stride=1的im2col：
    [1 2]
    [3 4]

    首先填充得到：
    [0 0 0 0]
    [0 1 2 0]
    [0 3 4 0]
    [0 0 0 0]

    然后滑动2x2窗口，每个窗口重塑为一列，结果为：
    [0 0 0 0 1 2 0 3 4]
    [0 0 0 1 2 0 3 4 0]
    [0 1 2 0 3 4 0 0 0]
    [1 2 0 3 4 0 0 0 0]

    应使用field_coords生成器遍历感受野，为保证后续重塑正确，需先遍历感受野再遍历输入。

    输入：
    - x: 4D数组，形状为(N, C, H, W)
    - field_height: 感受野高度
    - field_width: 感受野宽度
    - padding: 零填充大小
    - stride: 相邻感受野的水平和垂直偏移

    返回：
    - 2D数组，每列是一个输入的感受野
    """
    # 确定输出尺寸
    N, C, H, W = x.shape
    HH, WW = get_num_fields(x.shape, field_height, field_width, padding, stride)

    # 对输入进行零填充
    p = padding
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode="constant")

    # 为结果分配空间：每行对应感受野的一个元素，每列对应一个感受野
    cols = np.zeros((C * field_height * field_width, N * HH * WW), dtype=x.dtype)

    # 遍历所有感受野和输入，将感受野复制到列矩阵
    next_col = 0  # 下一列的索引
    for y0, y1, x0, x1 in field_coords(
        H, W, field_height, field_width, padding, stride
    ):
        for i in range(N):  # 遍历每个输入
            # 提取感受野并展平为列
            cols[:, next_col] = x_padded[i, :, y0:y1, x0:x1].flatten()
            next_col += 1
    return cols


def col2im(cols, x_shape, field_height=3, field_width=3, padding=1, stride=1):
    """
    执行与im2col不完全相反的操作，用于高效计算卷积层的反向传播。

    cols是一个矩阵，每列包含一个感受野的数据；x_shape是原始4D数据的形状。
    将cols中的数据重塑回x_shape的形状，重叠的感受野对应元素需累加。

    简单示例：使用相同参数（filter_height=filter_width=2, padding=0, stride=2），
    im2col的转换如下：
    [1 2 3]               [1 2 4 5]
    [4 5 6]  --im2col-->  [2 3 5 6]
    [7 8 9]               [4 5 7 8]
                          [5 6 8 9]

    而col2im的转换如下：
    [a b c d]               [ a      e+b      f ]
    [e f g h]  --col2im-->  [i+c  m+j+k+g+d  n+h]
    [i j k l]               [ k      o+l      p ]
    [m n o p]

    可重用field_coords生成器，遍历顺序与im2col一致（先感受野再输入）。

    若padding不为零，col2im应去除重塑数组中对应填充的部分，输出形状与x_shape一致。

    输入：
    - cols: 列矩阵，每列是一个感受野
    - x_shape: 元组(N, C, H, W)，表示目标形状
    - field_height: 感受野高度
    - field_width: 感受野宽度
    - padding: 填充大小
    - stride: 步长
    """
    x = np.empty(x_shape, dtype=cols.dtype)  # 最终输出
    N, C, H, W = x_shape
    # 计算感受野数量
    HH, WW = get_num_fields(x_shape, field_height, field_width, padding, stride)

    # 初始化填充后的图像
    x_padded = np.zeros((N, C, H + 2 * padding, W + 2 * padding), dtype=cols.dtype)
    next_col = 0  # 下一列的索引
    # 遍历所有感受野和输入，将列数据添加回图像（重叠区域累加）
    for y0, y1, x0, x1 in field_coords(
        H, W, field_height, field_width, padding, stride
    ):
        for i in range(N):  # 遍历每个输入
            col = cols[:, next_col]  # 当前列
            # 将列重塑为感受野形状并添加到填充后的图像
            x_padded[i, :, y0:y1, x0:x1] += col.reshape(C, field_height, field_width)
            next_col += 1

    # 去除填充部分
    if padding > 0:
        x = x_padded[:, :, padding:-padding, padding:-padding]
    else:
        x = x_padded
    # 验证输出形状是否正确
    assert x.shape == x_shape, "预期形状 %r 但得到 %r" % (x_shape, x.shape)
    return x


# ~~END DELETE~~
