import os, json
import numpy as np
import h5py  # 用于处理HDF5格式文件

# 获取当前文件所在目录的路径
dir_path = os.path.dirname(os.path.realpath(__file__))
# 定义COCO数据集的基础目录路径
BASE_DIR = os.path.join(dir_path, "datasets/coco_captioning")

def load_coco_data(base_dir=BASE_DIR, max_train=None, pca_features=True):
    """
    加载COCO图像描述数据集。

    参数:
        base_dir: 数据集的基础目录，默认为BASE_DIR
        max_train: 若不为None，则对训练数据进行子采样，保留最多max_train个样本
        pca_features: 布尔值，是否使用PCA降维后的特征

    返回:
        data: 包含数据集所有信息的字典
    """
    print('基础目录 ', base_dir)
    data = {}
    # 加载描述文件（HDF5格式）
    caption_file = os.path.join(base_dir, "coco2014_captions.h5")
    with h5py.File(caption_file, "r") as f:
        for k, v in f.items():
            data[k] = np.asarray(v)  # 将HDF5数据集转换为NumPy数组

    # 加载训练集图像特征（根据pca_features选择不同文件）
    if pca_features:
        train_feat_file = os.path.join(base_dir, "train2014_vgg16_fc7_pca.h5")
    else:
        train_feat_file = os.path.join(base_dir, "train2014_vgg16_fc7.h5")
    with h5py.File(train_feat_file, "r") as f:
        data["train_features"] = np.asarray(f["features"])  # 存储训练特征

    # 加载验证集图像特征（根据pca_features选择不同文件）
    if pca_features:
        val_feat_file = os.path.join(base_dir, "val2014_vgg16_fc7_pca.h5")
    else:
        val_feat_file = os.path.join(base_dir, "val2014_vgg16_fc7.h5")
    with h5py.File(val_feat_file, "r") as f:
        data["val_features"] = np.asarray(f["features"])  # 存储验证特征

    # 加载词汇表数据（JSON格式）
    dict_file = os.path.join(base_dir, "coco2014_vocab.json")
    with open(dict_file, "r") as f:
        dict_data = json.load(f)
        for k, v in dict_data.items():
            data[k] = v  # 将词汇表数据加入字典

    # 加载训练集图像的URL
    train_url_file = os.path.join(base_dir, "train2014_urls.txt")
    with open(train_url_file, "r") as f:
        train_urls = np.asarray([line.strip() for line in f])  # 读取并转换为NumPy数组
    data["train_urls"] = train_urls

    # 加载验证集图像的URL
    val_url_file = os.path.join(base_dir, "val2014_urls.txt")
    with open(val_url_file, "r") as f:
        val_urls = np.asarray([line.strip() for line in f])  # 读取并转换为NumPy数组
    data["val_urls"] = val_urls

    # 对训练数据进行子采样（如果需要）
    if max_train is not None:
        num_train = data["train_captions"].shape[0]
        mask = np.random.randint(num_train, size=max_train)  # 随机选择max_train个样本
        data["train_captions"] = data["train_captions"][mask]  # 子采样描述
        data["train_image_idxs"] = data["train_image_idxs"][mask]  # 子采样图像索引
#         data["train_features"] = data["train_features"][data["train_image_idxs"]]  # （注释掉的）子采样特征
    return data


def decode_captions(captions, idx_to_word):
    """
    将索引序列解码为文本描述。

    参数:
        captions: 形状为(N, T)或(T,)的数组，其中每个元素是词汇表索引
        idx_to_word: 从索引到单词的映射字典

    返回:
        decoded: 解码后的文本列表（若输入为单条则返回字符串）
    """
    singleton = False
    if captions.ndim == 1:  # 若输入是单条描述（1维数组）
        singleton = True
        captions = captions[None]  # 转为二维数组（添加批次维度）
    decoded = []
    N, T = captions.shape  # N为样本数，T为序列长度
    for i in range(N):
        words = []
        for t in range(T):
            word = idx_to_word[captions[i, t]]  # 将索引转换为单词
            if word != "<NULL>":  # 忽略填充符
                words.append(word)
            if word == "<END>":  # 遇到结束符则停止
                break
        decoded.append(" ".join(words))  # 拼接单词为句子
    if singleton:
        decoded = decoded[0]  # 若输入是单条，返回字符串而非列表
    return decoded


def sample_coco_minibatch(data, batch_size=100, split="train"):
    """
    从COCO数据集中采样一个小批量数据。

    参数:
        data: 由load_coco_data返回的数据集字典
        batch_size: 批量大小
        split: 数据集分割，"train"或"val"

    返回:
        captions: 形状为(batch_size, T)的描述数组
        image_features: 形状为(batch_size, D)的图像特征数组
        urls: 长度为batch_size的图像URL列表
    """
    split_size = data["%s_captions" % split].shape[0]  # 该分割的样本总数
    mask = np.random.choice(split_size, batch_size)  # 随机选择样本索引
    captions = data["%s_captions" % split][mask]  # 采样描述
    image_idxs = data["%s_image_idxs" % split][mask]  # 对应的图像索引
    image_features = data["%s_features" % split][image_idxs]  # 对应的图像特征
    urls = data["%s_urls" % split][image_idxs]  # 对应的图像URL
    return captions, image_features, urls
