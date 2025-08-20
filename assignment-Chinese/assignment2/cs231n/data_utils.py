from __future__ import print_function  # 确保Python 2和3的print函数兼容

from builtins import range
from six.moves import cPickle as pickle  # 导入pickle用于数据序列化
import numpy as np
import os
from imageio import imread  # 用于读取图像
import platform  # 用于获取Python版本信息

def load_pickle(f):
    """加载pickle文件，兼容Python 2和3"""
    version = platform.python_version_tuple()  # 获取Python版本
    if version[0] == "2":
        return pickle.load(f)  # Python 2的加载方式
    elif version[0] == "3":
        return pickle.load(f, encoding="latin1")  # Python 3的加载方式（指定编码）
    raise ValueError("无效的Python版本: {}".format(version))


def load_CIFAR_batch(filename):
    """加载CIFAR数据集的单个批次"""
    with open(filename, "rb") as f:
        datadict = load_pickle(f)  # 加载pickle数据
        X = datadict["data"]  # 图像数据
        Y = datadict["labels"]  # 标签数据
        # 重塑图像形状：(10000, 3, 32, 32) -> (10000, 32, 32, 3)，并转换为float类型
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        Y = np.array(Y)  # 转换标签为numpy数组
        return X, Y


def load_CIFAR10(ROOT):
    """加载整个CIFAR-10数据集"""
    xs = []  # 存储所有训练图像
    ys = []  # 存储所有训练标签
    # 加载5个训练批次
    for b in range(1, 6):
        f = os.path.join(ROOT, "data_batch_%d" % (b,))  # 批次文件路径
        X, Y = load_CIFAR_batch(f)  # 加载单个批次
        xs.append(X)
        ys.append(Y)
    # 合并所有训练数据
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y  # 释放内存
    # 加载测试集
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, "test_batch"))
    return Xtr, Ytr, Xte, Yte


def get_CIFAR10_data(
    num_training=49000, num_validation=1000, num_test=1000, subtract_mean=True
):
    """
    从磁盘加载CIFAR-10数据集并进行预处理，为分类器做准备。
    这些步骤与我们在SVM中使用的步骤相同，但浓缩为一个函数。
    """
    # 加载原始CIFAR-10数据
    cifar10_dir = os.path.join(
        os.path.dirname(__file__), "datasets/cifar-10-batches-py"
    )  # CIFAR-10数据目录
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    # 对数据进行子采样
    # 验证集：从训练集中选取num_validation个样本
    mask = list(range(num_training, num_training + num_validation))
    X_val = X_train[mask]
    y_val = y_train[mask]
    # 训练集：选取前num_training个样本
    mask = list(range(num_training))
    X_train = X_train[mask]
    y_train = y_train[mask]
    # 测试集：选取前num_test个样本
    mask = list(range(num_test))
    X_test = X_test[mask]
    y_test = y_test[mask]

    # 数据归一化：减去均值图像
    if subtract_mean:
        mean_image = np.mean(X_train, axis=0)  # 计算训练集的均值图像
        X_train -= mean_image
        X_val -= mean_image
        X_test -= mean_image

    # 转置使通道维度在前（适应PyTorch的格式：(N, C, H, W)）
    X_train = X_train.transpose(0, 3, 1, 2).copy()
    X_val = X_val.transpose(0, 3, 1, 2).copy()
    X_test = X_test.transpose(0, 3, 1, 2).copy()

    # 将数据打包成字典返回
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
    加载TinyImageNet数据集。TinyImageNet-100-A、TinyImageNet-100-B和
    TinyImageNet-200具有相同的目录结构，因此可用于加载其中任何一个。

    输入：
    - path: 加载目录的路径字符串。
    - dtype: 用于加载数据的numpy数据类型。
    - subtract_mean: 是否减去训练集的均值图像。

    返回：一个包含以下条目的字典：
    - class_names: 列表，其中class_names[i]是字符串列表，给出加载数据集中第i类的WordNet名称。
    - X_train: (N_tr, 3, 64, 64)的训练图像数组
    - y_train: (N_tr,)的训练标签数组
    - X_val: (N_val, 3, 64, 64)的验证图像数组
    - y_val: (N_val,)的验证标签数组
    - X_test: (N_test, 3, 64, 64)的测试图像数组。
    - y_test: (N_test,)的测试标签数组；如果测试标签不可用（如在学生代码中），则y_test为None。
    - mean_image: (3, 64, 64)的训练集均值图像数组
    """
    # 首先加载wnids（WordNet ID）
    with open(os.path.join(path, "wnids.txt"), "r") as f:
        wnids = [x.strip() for x in f]  # 读取所有wnid

    # 将wnids映射到整数标签
    wnid_to_label = {wnid: i for i, wnid in enumerate(wnids)}

    # 使用words.txt获取每个类的名称
    with open(os.path.join(path, "words.txt"), "r") as f:
        wnid_to_words = dict(line.split("\t") for line in f)  # 解析wnid到单词的映射
        for wnid, words in wnid_to_words.items():
            wnid_to_words[wnid] = [w.strip() for w in words.split(",")]  # 分割多个名称
    class_names = [wnid_to_words[wnid] for wnid in wnids]  # 按标签顺序存储类名

    # 接下来加载训练数据
    X_train = []
    y_train = []
    for i, wnid in enumerate(wnids):
        if (i + 1) % 20 == 0:
            print("加载第 %d / %d 个同义词集的训练数据" % (i + 1, len(wnids)))
        # 为了确定文件名，需要打开boxes文件
        boxes_file = os.path.join(path, "train", wnid, "%s_boxes.txt" % wnid)
        with open(boxes_file, "r") as f:
            filenames = [x.split("\t")[0] for x in f]  # 提取文件名
        num_images = len(filenames)  # 该类的图像数量

        # 初始化该类的图像和标签数组
        X_train_block = np.zeros((num_images, 3, 64, 64), dtype=dtype)
        y_train_block = wnid_to_label[wnid] * np.ones(num_images, dtype=np.int64)
        # 加载每个图像
        for j, img_file in enumerate(filenames):
            img_file = os.path.join(path, "train", wnid, "images", img_file)  # 图像路径
            img = imread(img_file)  # 读取图像
            if img.ndim == 2:
                ## 灰度图像，添加通道维度
                img.shape = (64, 64, 1)
            X_train_block[j] = img.transpose(2, 0, 1)  # 转置为(C, H, W)格式
        X_train.append(X_train_block)
        y_train.append(y_train_block)

    # 合并所有训练数据
    X_train = np.concatenate(X_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)

    # 接下来加载验证数据
    with open(os.path.join(path, "val", "val_annotations.txt"), "r") as f:
        img_files = []
        val_wnids = []
        for line in f:
            img_file, wnid = line.split("\t")[:2]  # 提取图像文件名和对应的wnid
            img_files.append(img_file)
            val_wnids.append(wnid)
        num_val = len(img_files)
        y_val = np.array([wnid_to_label[wnid] for wnid in val_wnids])  # 转换为标签
        X_val = np.zeros((num_val, 3, 64, 64), dtype=dtype)
        # 加载每个验证图像
        for i, img_file in enumerate(img_files):
            img_file = os.path.join(path, "val", "images", img_file)
            img = imread(img_file)
            if img.ndim == 2:
                img.shape = (64, 64, 1)  # 灰度图像处理
            X_val[i] = img.transpose(2, 0, 1)  # 转置为(C, H, W)

    # 接下来加载测试图像
    # 学生代码可能没有测试标签，因此需要遍历images目录中的文件
    img_files = os.listdir(os.path.join(path, "test", "images"))
    X_test = np.zeros((len(img_files), 3, 64, 64), dtype=dtype)
    for i, img_file in enumerate(img_files):
        img_file = os.path.join(path, "test", "images", img_file)
        img = imread(img_file)
        if img.ndim == 2:
            img.shape = (64, 64, 1)  # 灰度图像处理
        X_test[i] = img.transpose(2, 0, 1)  # 转置为(C, H, W)

    # 加载测试标签（如果存在）
    y_test = None
    y_test_file = os.path.join(path, "test", "test_annotations.txt")
    if os.path.isfile(y_test_file):
        with open(y_test_file, "r") as f:
            img_file_to_wnid = {}
            for line in f:
                line = line.split("\t")
                img_file_to_wnid[line[0]] = line[1]
        y_test = [wnid_to_label[img_file_to_wnid[img_file]] for img_file in img_files]
        y_test = np.array(y_test)

    # 计算均值图像并进行归一化
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
        "class_names": class_names,
        "mean_image": mean_image,
    }


def load_models(models_dir):
    """
    从磁盘加载保存的模型。这将尝试解 pickle 目录中的所有文件；
    任何解 pickle 时出错的文件（如README.txt）将被跳过。

    输入：
    - models_dir: 包含模型文件的目录路径字符串。每个模型文件是一个pickle字典，包含'model'字段。

    返回：
    一个将模型文件名映射到模型的字典。
    """
    models = {}
    for model_file in os.listdir(models_dir):
        with open(os.path.join(models_dir, model_file), "rb") as f:
            try:
                models[model_file] = load_pickle(f)["model"]  # 加载模型
            except pickle.UnpicklingError:
                continue  # 跳过解包错误的文件
    return models


def load_imagenet_val(num=None):
    """加载少量ImageNet的验证图像。

    输入：
    - num: 要加载的图像数量（最大25）

    返回：
    - X: 形状为[num, 224, 224, 3]的numpy数组
    - y: 整数图像标签的numpy数组，形状为[num]
    - class_names: 将整数标签映射到类名的字典
    """
    imagenet_fn = os.path.join(
        os.path.dirname(__file__), "datasets/imagenet_val_25.npz"
    )  # ImageNet验证集文件路径
    if not os.path.isfile(imagenet_fn):
        print("文件 %s 未找到" % imagenet_fn)
        print("请运行以下命令：")
        print("cd cs231n/datasets")
        print("bash get_imagenet_val.sh")
        assert False, "需要下载imagenet_val_25.npz"

    # 修改np.load的默认参数以支持加载对象数组
    # 参考：https://stackoverflow.com/questions/55890813/how-to-fix-object-arrays-cannot-be-loaded-when-allow-pickle-false-for-imdb-loa
    np_load_old = np.load
    np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True,** k)
    f = np.load(imagenet_fn)
    np.load = np_load_old  # 恢复默认参数
    X = f["X"]  # 图像数据
    y = f["y"]  # 标签数据
    class_names = f["label_map"].item()  # 类名映射
    if num is not None:
        X = X[:num]  # 截取前num个图像
        y = y[:num]  # 截取前num个标签
    return X, y, class_names
