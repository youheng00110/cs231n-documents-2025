from __future__ import print_function
from builtins import zip
from builtins import range

import matplotlib
import numpy as np
from scipy.ndimage import uniform_filter


def extract_features(imgs, feature_fns, verbose=False):
    """
    给定图像的像素数据和几个可对单张图像进行操作的特征函数，
    对所有图像应用所有特征函数，拼接每张图像的特征向量，
    并将所有图像的特征存储在一个矩阵中。

    输入：
    - imgs：N x H x W x C的数组，包含N张图像的像素数据。
    - feature_fns：特征函数列表，共k个。第i个特征函数应以
      H x W x D的数组为输入，并返回长度为F_i的（一维）数组。
    - verbose：布尔值；若为True，则打印进度信息。

    返回：
    一个形状为(N, F_1 + ... + F_k)的数组，其中每一行是单张图像
    所有特征的拼接结果。
    """
    num_images = imgs.shape[0]
    if num_images == 0:
        return np.array([])

    # 用第一张图像确定特征维度
    feature_dims = []
    first_image_features = []
    for feature_fn in feature_fns:
        feats = feature_fn(imgs[0].squeeze())  # 移除单维度条目
        assert len(feats.shape) == 1, "特征函数的输出必须是一维的"
        feature_dims.append(feats.size)
        first_image_features.append(feats)

    # 已知特征维度后，可分配一个大数组来存储所有特征（每行对应一张图像）
    total_feature_dim = sum(feature_dims)
    imgs_features = np.zeros((num_images, total_feature_dim))
    imgs_features[0] = np.hstack(first_image_features).T  # 拼接第一张图像的所有特征

    # 提取其余图像的特征
    for i in range(1, num_images):
        idx = 0
        for feature_fn, feature_dim in zip(feature_fns, feature_dims):
            next_idx = idx + feature_dim
            imgs_features[i, idx:next_idx] = feature_fn(imgs[i].squeeze())
            idx = next_idx
        if verbose and i % 1000 == 999:
            print("已完成 %d / %d 张图像的特征提取" % (i + 1, num_images))

    return imgs_features


def rgb2gray(rgb):
    """将RGB图像转换为灰度图像

      参数：
        rgb : RGB图像

      返回：
        gray : 灰度图像

    """
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.144])  # 利用RGB转灰度的加权公式


def hog_feature(im):
    """计算图像的梯度方向直方图（HOG）特征

         基于skimage.feature.hog修改而来
         http://pydoc.net/Python/scikits-image/0.4.2/skimage.feature.hog

       参考：
         Histograms of Oriented Gradients for Human Detection
         Navneet Dalal and Bill Triggs, CVPR 2005

      参数：
        im : 输入的灰度图或RGB图像

      返回：
        feat: 梯度方向直方图（HOG）特征

    """

    # 若为RGB图像则转换为灰度图
    if im.ndim == 3:
        image = rgb2gray(im)
    else:
        image = np.at_least_2d(im)  # 确保至少为二维（灰度图）

    sx, sy = image.shape  # 图像尺寸
    orientations = 9  # 梯度 bins 数量
    cx, cy = (8, 8)  # 每个细胞（cell）的像素数

    gx = np.zeros(image.shape)
    gy = np.zeros(image.shape)
    gx[:, :-1] = np.diff(image, n=1, axis=1)  # 计算x方向梯度
    gy[:-1, :] = np.diff(image, n=1, axis=0)  # 计算y方向梯度
    grad_mag = np.sqrt(gx **2 + gy** 2)  # 梯度幅值
    # 梯度方向（转换为角度并调整至0-180度）
    grad_ori = np.arctan2(gy, (gx + 1e-15)) * (180 / np.pi) + 90

    n_cellsx = int(np.floor(sx / cx))  # x方向的细胞数量
    n_cellsy = int(np.floor(sy / cy))  # y方向的细胞数量
    # 计算方向积分图像
    orientation_histogram = np.zeros((n_cellsx, n_cellsy, orientations))
    for i in range(orientations):
        # 为该方向创建新的积分图像
        # 分离该范围内的方向
        temp_ori = np.where(grad_ori < 180 / orientations * (i + 1), grad_ori, 0)
        temp_ori = np.where(grad_ori >= 180 / orientations * i, temp_ori, 0)
        # 选择这些方向对应的幅值
        cond2 = temp_ori > 0
        temp_mag = np.where(cond2, grad_mag, 0)
        # 对每个细胞区域进行均值滤波并提取结果
        orientation_histogram[:, :, i] = uniform_filter(temp_mag, size=(cx, cy))[
            round(cx / 2) :: cx, round(cy / 2) :: cy
        ].T

    return orientation_histogram.ravel()  # 展平为一维数组


def color_histogram_hsv(im, nbin=10, xmin=0, xmax=255, normalized=True):
    """
    利用色调（hue）计算图像的颜色直方图。

    输入：
    - im：H x W x C的数组，包含RGB图像的像素数据。
    - nbin：直方图的分箱数量（默认：10）。
    - xmin：像素最小值（默认：0）。
    - xmax：像素最大值（默认：255）。
    - normalized：是否对直方图进行归一化（默认：True）。

    返回：
      长度为nbin的一维向量，代表输入图像基于色调的颜色直方图。
    """
    ndim = im.ndim
    bins = np.linspace(xmin, xmax, nbin + 1)  # 生成直方图的分箱边界
    # 将RGB转换为HSV并缩放至[0, xmax]范围
    hsv = matplotlib.colors.rgb_to_hsv(im / xmax) * xmax
    # 计算色调通道的直方图
    imhist, bin_edges = np.histogram(hsv[:, :, 0], bins=bins, density=normalized)
    imhist = imhist * np.diff(bin_edges)  # 考虑分箱宽度的校正

    # 返回直方图
    return imhist


# ~~START DELETE~~
# 这些是我们为了试验而实现的其他特征，不向学生开放。
def color_histogram(im, nbin=10, xmin=0, xmax=255, normalized=True):
    """计算图像的颜色直方图特征

      参数：
        im : 灰度图或RGB图像的numpy数组
        nbin : 直方图分箱数量（默认：10）
        xmin : 像素最小值（默认：0）
        xmax : 像素最大值（默认：255）
        normalized : 是否归一化直方图的布尔标志

      返回：
        feat : 颜色直方图特征

    """
    ndim = im.ndim
    bins = np.linspace(xmin, xmax, nbin + 1)
    # 灰度图像
    if ndim == 2:
        imhist, bin_edges = np.histogram(im, bins=bins, density=normalized)
        return imhist
    # RGB图像
    elif ndim == 3:
        color_hist = np.array([])
        # 遍历三个颜色通道
        for k in range(3):
            # 计算归一化直方图
            imhist, bin_edges = np.histogram(im[:, :, k], bins=bins, density=normalized)
            imhist = imhist * np.diff(bin_edges)
            # 拼接直方图
            color_hist = np.concatenate((color_hist, imhist))
        # 返回直方图
        return color_hist
    # 未知图像类型
    return np.array([])


def color_histogram_spatial(img, levels=3, nbin=4):
    """
    金字塔上的颜色直方图。
    """
    feats = []

    for level in range(1, levels + 1):
        chunks = np.array_split(img, level, axis=0)
        chunks = [np.array_split(chunk, level, axis=1) for chunk in chunks]
        for x in chunks:
            for chunk in x:
                feats.append(color_histogram_cross(chunk, nbin=nbin))

    return np.hstack(feats)


def color_histogram_cross(img, nbin=5, normalized=True):
    """
    三维分箱的RGB颜色直方图。
    """
    height, width, channels = img.shape
    new_size = (height * width, channels)
    colors = np.reshape(img, new_size)
    return np.histogramdd(colors, bins=nbin, normed=normalized)[0].flatten()


# ~~END DELETE~~
