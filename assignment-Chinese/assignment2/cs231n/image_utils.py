from __future__ import print_function
from future import standard_library

standard_library.install_aliases()
from builtins import range
import urllib.request, urllib.error, urllib.parse, os, tempfile  # 用于URL请求、文件操作和临时文件

import numpy as np
from imageio import imread  # 用于读取图像
from PIL import Image  # 用于图像处理

"""
用于图像查看和处理的工具函数。
"""


def blur_image(X):
    """
    非常轻微的图像模糊操作，用作图像生成的正则化手段。

    输入：
    - X: 图像数据，形状为(N, 3, H, W)，其中N是批量大小，3是通道数，H/W是高/宽

    返回：
    - X_blur: X的模糊版本，形状为(N, 3, H, W)
    """
    from .fast_layers import conv_forward_fast  # 导入快速卷积前向传播函数

    # 定义模糊卷积核（3个输入通道，3个输出通道，3x3大小）
    w_blur = np.zeros((3, 3, 3, 3))
    b_blur = np.zeros(3)  # 偏置为0
    blur_param = {"stride": 1, "pad": 1}  # 卷积参数：步长1，填充1（保持尺寸不变）
    # 初始化模糊核（每个通道使用相同的模糊权重）
    for i in range(3):
        w_blur[i, i] = np.asarray([[1, 2, 1], [2, 188, 2], [1, 2, 1]], dtype=np.float32)
    w_blur /= 200.0  # 归一化权重（总和为200）
    # 通过卷积操作实现模糊
    return conv_forward_fast(X, w_blur, b_blur, blur_param)[0]


# SqueezeNet模型的图像预处理均值和标准差（基于ImageNet数据集）
SQUEEZENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
SQUEEZENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_image(img):
    """对图像进行预处理以适应SqueezeNet模型。
    
    减去像素均值并除以标准差。
    """
    return (img.astype(np.float32) / 255.0 - SQUEEZENET_MEAN) / SQUEEZENET_STD


def deprocess_image(img, rescale=False):
    """撤销对图像的预处理并转换回uint8格式。"""
    img = img * SQUEEZENET_STD + SQUEEZENET_MEAN  # 恢复均值和标准差
    if rescale:
        # 重新缩放图像至[0, 1]范围
        vmin, vmax = img.min(), img.max()
        img = (img - vmin) / (vmax - vmin)
    # 裁剪到[0, 255]并转换为uint8
    return np.clip(255 * img, 0.0, 255.0).astype(np.uint8)


def image_from_url(url):
    """
    从URL读取图像。返回包含像素数据的numpy数组。
    实现方式：将图像写入临时文件，再读取回来。略显繁琐。
    """
    try:
        f = urllib.request.urlopen(url)  # 打开URL
        _, fname = tempfile.mkstemp()  # 创建临时文件
        with open(fname, "wb") as ff:
            ff.write(f.read())  # 将URL内容写入临时文件
        img = imread(fname)  # 读取临时文件中的图像
        os.remove(fname)  # 删除临时文件
        return img
    except urllib.error.URLError as e:
        print("URL错误: ", e.reason, url)
    except urllib.error.HTTPError as e:
        print("HTTP错误: ", e.code, url)


def load_image(filename, size=None):
    """从磁盘加载图像并调整大小。
    
    输入：
    - filename: 文件路径
    - size: 调整后最短边的长度
    """
    img = imread(filename)  # 读取图像
    if size is not None:
        orig_shape = np.array(img.shape[:2])  # 原始形状（高，宽）
        min_idx = np.argmin(orig_shape)  # 找到最短边的索引
        scale_factor = float(size) / orig_shape[min_idx]  # 缩放因子
        new_shape = (orig_shape * scale_factor).astype(int)  # 新形状
        # 调整图像大小（注意：当前宽高可能翻转，且应改为双线性插值以匹配PyTorch实现）
        img = np.array(Image.fromarray(img).resize(new_shape, resample=Image.NEAREST))
    return img
