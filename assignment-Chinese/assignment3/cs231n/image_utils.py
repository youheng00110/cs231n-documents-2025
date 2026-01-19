"""用于图像查看和处理的工具函数"""

import urllib.request, urllib.error, urllib.parse  # 用于URL请求和文件操作
from io import BytesIO  # 用于在内存中处理二进制数据

import numpy as np
from imageio import imread  # 用于读取图像
from PIL import Image  # 用于图像处理


def blur_image(X):
    """
    非常温和的图像模糊操作，用作图像生成的正则化器
    
    输入:
    - X: 形状为(N, 3, H, W)的图像数据
    
    返回:
    - X_blur: X的模糊版本，形状为(N, 3, H, W)
    """
    from .fast_layers import conv_forward_fast  # 导入快速卷积函数

    # 初始化模糊卷积核
    w_blur = np.zeros((3, 3, 3, 3))
    b_blur = np.zeros(3)
    blur_param = {"stride": 1, "pad": 1}  # 卷积参数：步长1，填充1
    # 定义模糊核（高斯模糊近似）
    for i in range(3):
        w_blur[i, i] = np.asarray([[1, 2, 1], [2, 188, 2], [1, 2, 1]], dtype=np.float32)
    w_blur /= 200.0  # 归一化卷积核
    # 进行卷积操作实现模糊
    return conv_forward_fast(X, w_blur, b_blur, blur_param)[0]


# SqueezeNet模型的图像均值和标准差（用于预处理）
SQUEEZENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
SQUEEZENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_image(img):
    """对图像进行预处理以适应SqueezeNet模型
    
    减去像素均值并除以标准差
    """
    return (img.astype(np.float32) / 255.0 - SQUEEZENET_MEAN) / SQUEEZENET_STD


def deprocess_image(img, rescale=False):
    """撤销图像的预处理并转换回uint8格式"""
    img = img * SQUEEZENET_STD + SQUEEZENET_MEAN  # 恢复均值和标准差
    if rescale:
        # 重新缩放图像到[0, 1]范围
        vmin, vmax = img.min(), img.max()
        img = (img - vmin) / (vmax - vmin)
    # 裁剪到[0, 255]并转换为uint8
    return np.clip(255 * img, 0.0, 255.0).astype(np.uint8)


def image_from_url(url):
    """
    从URL读取图像。返回包含像素数据的numpy数组。
    实现方式：直接将远程内容读入内存再解析，避免Windows句柄占用。
    """
    try:
        with urllib.request.urlopen(url) as f:  # 打开URL
            data = f.read()  # 读取全部图像字节
        # 直接从内存缓冲区解析图片，绕开Windows无法及时删除临时文件的问题
        img = imread(BytesIO(data))
        return img
    except urllib.error.URLError as e:
        print("URL错误: ", e.reason, url)
    except urllib.error.HTTPError as e:
        print("HTTP错误: ", e.code, url)


def load_image(filename, size=None):
    """从磁盘加载并调整图像大小
    
    输入:
    - filename: 文件路径
    - size: 调整后最短边的长度
    """
    img = imread(filename)  # 读取图像
    if size is not None:
        orig_shape = np.array(img.shape[:2])  # 获取原始图像的高度和宽度
        min_idx = np.argmin(orig_shape)  # 找到最短边的索引
        scale_factor = float(size) / orig_shape[min_idx]  # 计算缩放因子
        new_shape = (orig_shape * scale_factor).astype(int)  # 计算新的尺寸
        # 注意：当前宽度和高度值在这里是翻转的，我们应该
        # 将重采样方法改为BILINEAR以匹配torch实现
        img = np.array(Image.fromarray(img).resize(new_shape, resample=Image.NEAREST))
    return img
