from builtins import range
from past.builtins import xrange

from math import sqrt, ceil
import numpy as np


def visualize_grid(Xs, ubound=255.0, padding=1):
    """
    将4D图像数据张量重塑为网格，以便于可视化。

    输入：
    - Xs: 形状为(N, H, W, C)的数据，其中N是图像数量，H/W是高/宽，C是通道数
    - ubound: 输出网格的值将缩放到[0, ubound]范围
    - padding: 网格元素之间的空白像素数
    """
    (N, H, W, C) = Xs.shape  # 解析输入数据的形状
    grid_size = int(ceil(sqrt(N)))  # 计算网格的大小（取平方根并向上取整）
    # 计算网格的总高度和宽度（包含空白）
    grid_height = H * grid_size + padding * (grid_size - 1)
    grid_width = W * grid_size + padding * (grid_size - 1)
    grid = np.zeros((grid_height, grid_width, C))  # 初始化网格

    next_idx = 0  # 下一个要放置的图像索引
    y0, y1 = 0, H  # 当前行的起始和结束高度
    for y in range(grid_size):
        x0, x1 = 0, W  # 当前列的起始和结束宽度
        for x in range(grid_size):
            if next_idx < N:  # 如果还有图像未放置
                img = Xs[next_idx]
                low, high = np.min(img), np.max(img)  # 获取图像的最小值和最大值
                # 将图像归一化到[0, ubound]并放入网格
                grid[y0:y1, x0:x1] = ubound * (img - low) / (high - low)
                next_idx += 1  # 索引递增
            # 更新下一列的起始和结束宽度
            x0 += W + padding
            x1 += W + padding
        # 更新下一行的起始和结束高度
        y0 += H + padding
        y1 += H + padding
    return grid


def vis_grid(Xs):
    """ 可视化图像网格 """
    (N, H, W, C) = Xs.shape  # 解析输入形状
    A = int(ceil(sqrt(N)))  # 网格大小（方阵）
    # 初始化网格，背景为输入图像的最小值
    G = np.ones((A * H + A, A * W + A, C), Xs.dtype)
    G *= np.min(Xs)
    n = 0  # 当前图像索引
    for y in range(A):
        for x in range(A):
            if n < N:  # 放置图像到网格的对应位置（包含空白）
                G[y * H + y : (y + 1) * H + y, x * W + x : (x + 1) * W + x, :] = Xs[
                    n, :, :, :
                ]
                n += 1
    # 将网格归一化到[0, 1]范围
    maxg = G.max()
    ming = G.min()
    G = (G - ming) / (maxg - ming)
    return G


def vis_nn(rows):
    """ 可视化图像的数组的数组（即二维数组的图像） """
    N = len(rows)  # 行数
    D = len(rows[0])  # 列数
    H, W, C = rows[0][0].shape  # 单个图像的形状
    Xs = rows[0][0]  # 用于获取数据类型
    # 初始化网格，背景为1（后续会归一化）
    G = np.ones((N * H + N, D * W + D, C), Xs.dtype)
    # 放置每个位置的图像（行和列之间有空白）
    for y in range(N):
        for x in range(D):
            G[y * H + y : (y + 1) * H + y, x * W + x : (x + 1) * W + x, :] = rows[y][x]
    # 归一化到[0, 1]范围
    maxg = G.max()
    ming = G.min()
    G = (G - ming) / (maxg - ming)
    return G
