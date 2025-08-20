"""
数据加载与预处理工具模块
"""
from __future__ import print_function

from builtins import range
from six.moves import cPickle as pickle
import numpy as np
import os
from imageio import imread
import platform


def load_pickle(f):
    """
    兼容 Python 2/3 的 pickle 加载函数。

    参数
    ----
    f : file-like object
        以二进制模式打开的文件句柄。

    返回
    ----
    object
        反序列化后的 Python 对象。
    """
    version = platform.python_version_tuple()
    if version[0] == "2":
        return pickle.load(f)
    elif version[0] == "3":
        return pickle.load(f, encoding="latin1")
    raise ValueError("invalid python version: {}".format(version))


def load_CIFAR_batch(filename):
    """
    加载单个 CIFAR-10/100 batch。

    参数
    ----
    filename : str
        batch 文件路径。

    返回
    ----
    X : ndarray, shape (10000, 32, 32, 3)
        图像数据，像素值已转为 float。
    Y : ndarray, shape (10000,)
        对应的标签。
    """
    with open(filename, "rb") as f:
        datadict = load_pickle(f)
        X = datadict["data"]
        Y = datadict["labels"]
        # 重塑为 (N, C, H, W) 再转置为 (N, H, W, C)
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        Y = np.array(Y)
        return X, Y


def load_CIFAR10(ROOT):
    """
    加载整个 CIFAR-10 数据集。

    参数
    ----
    ROOT : str
        CIFAR-10 根目录。

    返回
    ----
    X_train, y_train : ndarray
        训练集数据和标签。
    X_test, y_test   : ndarray
        测试集数据和标签。
    """
    xs, ys = [], []
    for b in range(1, 6):
        f = os.path.join(ROOT, "data_batch_%d" % b)
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del xs, ys  # 释放中间变量
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, "test_batch"))
    return Xtr, Ytr, Xte, Yte


def get_CIFAR10_data(
    num_training=49000, num_validation=1000, num_test=1000, subtract_mean=True
):
    """
    从磁盘加载 CIFAR-10 并完成预处理，方便后续分类器使用。

    参数
    ----
    num_training   : int
        训练集样本数。
    num_validation : int
        验证集样本数。
    num_test       : int
        测试集样本数。
    subtract_mean  : bool
        是否减去训练集均值。

    返回
    ----
    dict
        包含以下键值：
        'X_train', 'y_train' : 训练集
        'X_val',   'y_val'   : 验证集
        'X_test',  'y_test'  : 测试集
    """
    cifar10_dir = os.path.join(os.path.dirname(__file__), "datasets/cifar-10-batches-py")
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    # 划分验证集
    mask = list(range(num_training, num_training + num_validation))
    X_val = X_train[mask]
    y_val = y_train[mask]

    # 取训练集
    mask = list(range(num_training))
    X_train = X_train[mask]
    y_train = y_train[mask]

    # 取测试集
    mask = list(range(num_test))
    X_test = X_test[mask]
    y_test = y_test[mask]

    # 均值归一化
    if subtract_mean:
        mean_image = np.mean(X_train, axis=0)
        X_train -= mean_image
        X_val -= mean_image
        X_test -= mean_image

    # 调整维度顺序：NHWC -> NCHW
    X_train = X_train.transpose(0, 3, 1, 2).copy()
    X_val   = X_val.transpose(0, 3, 1, 2).copy()
    X_test  = X_test.transpose(0, 3, 1, 2).copy()

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }


def load_tiny_imagenet(path, dtype=np.float32, subtract_mean=True):
    """
    加载 TinyImageNet 数据集。

    参数
    ----
    path : str
        TinyImageNet 根目录。
    dtype : numpy dtype
        用于加载数据的类型。
    subtract_mean : bool
        是否减去训练集均值。

    返回
    ----
    dict
        包含以下键值：
        'class_names' : 类别名列表
        'X_train', 'y_train' : 训练集
        'X_val', 'y_val'     : 验证集
        'X_test', 'y_test'   : 测试集
        'mean_image'         : 训练集均值
    """
    # 读取类别 wnids
    with open(os.path.join(path, "wnids.txt"), "r") as f:
        wnids = [x.strip() for x in f]

    # 建立 wnid 到标签的映射
    wnid_to_label = {wnid: i for i, wnid in enumerate(wnids)}

    # 读取类别名称
    with open(os.path.join(path, "words.txt"), "r") as f:
        wnid_to_words = dict(line.split("\t") for line in f)
        for wnid, words in wnid_to_words.items():
            wnid_to_words[wnid] = [w.strip() for w in words.split(",")]
    class_names = [wnid_to_words[wnid] for wnid in wnids]

    # 加载训练数据
    X_train, y_train = [], []
    for i, wnid in enumerate(wnids):
        if (i + 1) % 20 == 0:
            print("加载训练类别 %d / %d" % (i + 1, len(wnids)))
        boxes_file = os.path.join(path, "train", wnid, "%s_boxes.txt" % wnid)
        with open(boxes_file, "r") as f:
            filenames = [x.split("\t")[0] for x in f]
        num_images = len(filenames)

        X_train_block = np.zeros((num_images, 3, 64, 64), dtype=dtype)
        y_train_block = wnid_to_label[wnid] * np.ones(num_images, dtype=np.int64)
        for j, img_file in enumerate(filenames):
            img_file = os.path.join(path, "train", wnid, "images", img_file)
            img = imread(img_file)
            if img.ndim == 2:
                img.shape = (64, 64, 1)
            X_train_block[j] = img.transpose(2, 0, 1)
        X_train.append(X_train_block)
        y_train.append(y_train_block)

    X_train = np.concatenate(X_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)

    # 加载验证数据
    with open(os.path.join(path, "val", "val_annotations.txt"), "r") as f:
        img_files, val_wnids = [], []
        for line in f:
            img_file, wnid = line.split("\t")[:2]
            img_files.append(img_file)
            val_wnids.append(wnid)
        num_val = len(img_files)
        y_val = np.array([wnid_to_label[wnid] for wnid in val_wnids])
        X_val = np.zeros((num_val, 3, 64, 64), dtype=dtype)
        for i, img_file in enumerate(img_files):
            img_file = os.path.join(path, "val", "images", img_file)
            img = imread(img_file)
            if img.ndim == 2:
                img.shape = (64, 64, 1)
            X_val[i] = img.transpose(2, 0, 1)

    # 加载测试图像
    img_files = os.listdir(os.path.join(path, "test", "images"))
    X_test = np.zeros((len(img_files), 3, 64, 64), dtype=dtype)
    for i, img_file in enumerate(img_files):
        img_file = os.path.join(path, "test", "images", img_file)
        img = imread(img_file)
        if img.ndim == 2:
            img.shape = (64, 64, 1)
        X_test[i] = img.transpose(2, 0, 1)
    y_test = None  # 学生代码通常无测试标签

    # 计算均值并归一化
    mean_image = X_train.mean(axis=0)
    if subtract_mean:
        X_train -= mean_image[None]
        X_val -= mean_image[None]
        X_test -= mean_image[None]

    return {
        "class_names": class_names,
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "mean_image": mean_image,
    }


def load_models(models_dir):
    """
    从磁盘加载已保存的模型。

    参数
    ----
    models_dir : str
        存放模型文件的目录路径。

    返回
    ----
    dict
        以模型文件名为键、对应模型为值的字典。
    """
    models = {}
    for model_file in os.listdir(models_dir):
        with open(os.path.join(models_dir, model_file), "rb") as f:
            try:
                models[model_file] = load_pickle(f)["model"]
            except pickle.UnpicklingError:
                continue
    return models


# 解决 Numpy 1.17+ 与 TensorFlow 2.x 的兼容性问题
# https://stackoverflow.com/questions/55890813/how-to-fix-object-arrays-cannot-be-loaded-when-allow-pickle-false-for-imdb-loa 
np_load_old = np.load
np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)


def load_imagenet_val(num=None):
    """
    从 ImageNet 验证集中加载少量图像。

    参数
    ----
    num : int or None
        要加载的图像数量（最多 25）。

    返回
    ----
    X : ndarray, shape [num, 224, 224, 3]
        图像数据。
    y : ndarray, shape [num]
        图像标签。
    class_names : dict
        将整数标签映射到类别名称的字典。
    """
    imagenet_fn = os.path.join(
        os.path.dirname(__file__), "datasets/imagenet_val_25.npz"
    )
    if not os.path.isfile(imagenet_fn):
        print("文件 %s 未找到" % imagenet_fn)
        print("请运行以下命令：")
        print("cd cs231n/datasets")
        print("bash get_imagenet_val.sh")
        assert False, "需要下载 imagenet_val_25.npz"

    # 临时修改 np.load 参数以解决 allow_pickle=False 的问题
    f = np.load(imagenet_fn)
    X = f["X"]
    y = f["y"]
    class_names = f["label_map"].item()
    if num is not None:
        X = X[:num]
        y = y[:num]
    return X, y, class_names
