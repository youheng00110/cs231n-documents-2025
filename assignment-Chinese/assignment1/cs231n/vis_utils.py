"""
图像可视化工具模块
"""
from builtins import range
from past.builtins import xrange  # 兼容旧版本Python的xrange

from math import sqrt, ceil
import numpy as np


def visualize_grid(Xs, ubound=255.0, padding=1):
    """
    将4D张量的图像数据重塑为网格，便于可视化。

    参数
    ----
    Xs : 多维数组，形状为(N, H, W, C)
        图像数据，其中：
        - N为图像数量
        - H为图像高度
        - W为图像宽度
        - C为通道数（如RGB为3通道）
    ubound : 浮点数
        输出网格的值将缩放到[0, ubound]范围。
    padding : 整数
        网格中元素之间的空白像素数。

    返回
    ----
    grid : 多维数组，形状为(grid_height, grid_width, C)
        可视化网格图像。
    """
    (N, H, W, C) = Xs.shape
    grid_size = int(ceil(sqrt(N)))  # 计算网格大小（向上取整的平方根）
    # 计算网格的高度和宽度，考虑 padding
    grid_height = H * grid_size + padding * (grid_size - 1)
    grid_width = W * grid_size + padding * (grid_size - 1)
    grid = np.zeros((grid_height, grid_width, C))  # 初始化网格
    next_idx = 0  # 下一个要放置的图像索引
    y0, y1 = 0, H  # 当前行的起始和结束高度
    for y in range(grid_size):
        x0, x1 = 0, W  # 当前列的起始和结束宽度
        for x in range(grid_size):
            if next_idx < N:  # 如果还有图像要放置
                img = Xs[next_idx]
                # 归一化图像值到[0, ubound]
                low, high = np.min(img), np.max(img)
                grid[y0:y1, x0:x1] = ubound * (img - low) / (high - low)
                next_idx += 1
            # 更新下一列的起始和结束宽度
            x0 += W + padding
            x1 += W + padding
        # 更新下一行的起始和结束高度
        y0 += H + padding
        y1 += H + padding
    return grid


def vis_grid(Xs):
    """
    可视化图像网格（简化版）。
    """
    (N, H, W, C) = Xs.shape
    A = int(ceil(sqrt(N)))  # 网格的行列数
    # 初始化网格，背景为所有图像的最小值
    G = np.ones((A * H + A, A * W + A, C), Xs.dtype)
    G *= np.min(Xs)
    n = 0  # 当前图像索引
    for y in range(A):
        for x in range(A):
            if n < N:  # 放置图像到网格
                G[y * H + y : (y + 1) * H + y, x * W + x : (x + 1) * W + x, :] = Xs[
                    n, :, :, :
                ]
                n += 1
    # 归一化到[0, 1]范围
    maxg = G.max()
    ming = G.min()
    G = (G - ming) / (maxg - ming)
    return G


def vis_nn(rows):
    """
    可视化多行图像（适用于神经网络层的可视化）。
    """
    N = len(rows)  # 行数
    D = len(rows[0])  # 每行的图像数
    H, W, C = rows[0][0].shape  # 单个图像的尺寸
    Xs = rows[0][0]
    # 初始化网格，考虑行与行、列与列之间的间隔
    G = np.ones((N * H + N, D * W + D, C), Xs.dtype)
    for y in range(N):
        for x in range(D):
            # 放置每个位置的图像
            G[y * H + y : (y + 1) * H + y, x * W + x : (x + 1) * W + x, :] = rows[y][x]
    # 归一化到[0, 1]范围
    maxg = G.max()
    ming = G.min()
    G = (G - ming) / (maxg - ming)
    return G