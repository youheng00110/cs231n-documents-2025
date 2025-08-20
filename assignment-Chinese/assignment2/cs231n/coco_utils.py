import os, json
import numpy as np
import h5py  # 用于处理HDF5格式文件

# 获取当前文件所在目录路径
dir_path = os.path.dirname(os.path.realpath(__file__))
# 定义COCO数据集的基础目录路径
BASE_DIR = os.path.join(dir_path, "datasets/coco_captioning")

def load_coco_data(base_dir=BASE_DIR, max_train=None, pca_features=True):
    """
    加载COCO数据集的图像描述数据。
    
    参数：
    - base_dir：数据集的基础目录
    - max_train：训练样本的最大数量（用于子采样）
    - pca_features：是否使用PCA降维后的图像特征
    
    返回：
    - data：包含各种数据集信息的字典
    """
    print('基础目录 ', base_dir)
    data = {}  # 存储数据集的字典
    # 描述文件路径
    caption_file = os.path.join(base_dir, "coco2014_captions.h5")
    # 读取描述文件（HDF5格式）
    with h5py.File(caption_file, "r") as f:
        for k, v in f.items():
            data[k] = np.asarray(v)  # 将所有键值对存入data字典

    # 读取训练集图像特征
    if pca_features:
        # 使用PCA降维后的特征文件
        train_feat_file = os.path.join(base_dir, "train2014_vgg16_fc7_pca.h5")
    else:
        # 使用原始特征文件
        train_feat_file = os.path.join(base_dir, "train2014_vgg16_fc7.h5")
    with h5py.File(train_feat_file, "r") as f:
        data["train_features"] = np.asarray(f["features"])  # 存储训练集特征

    # 读取验证集图像特征
    if pca_features:
        val_feat_file = os.path.join(base_dir, "val2014_vgg16_fc7_pca.h5")
    else:
        val_feat_file = os.path.join(base_dir, "val2014_vgg16_fc7.h5")
    with h5py.File(val_feat_file, "r") as f:
        data["val_features"] = np.asarray(f["features"])  # 存储验证集特征

    # 读取词汇表文件（JSON格式）
    dict_file = os.path.join(base_dir, "coco2014_vocab.json")
    with open(dict_file, "r") as f:
        dict_data = json.load(f)
        for k, v in dict_data.items():
            data[k] = v  # 将词汇表数据存入data字典（如word_to_idx, idx_to_word）

    # 读取训练集图像的URL
    train_url_file = os.path.join(base_dir, "train2014_urls.txt")
    with open(train_url_file, "r") as f:
        # 读取每行并去除首尾空白，转换为numpy数组
        train_urls = np.asarray([line.strip() for line in f])
    data["train_urls"] = train_urls  # 存储训练集URL

    # 读取验证集图像的URL
    val_url_file = os.path.join(base_dir, "val2014_urls.txt")
    with open(val_url_file, "r") as f:
        val_urls = np.asarray([line.strip() for line in f])
    data["val_urls"] = val_urls  # 存储验证集URL

    # 可能对子采样训练数据（如果指定了max_train）
    if max_train is not None:
        num_train = data["train_captions"].shape[0]  # 训练描述的总数
        # 随机选择max_train个样本的索引
        mask = np.random.randint(num_train, size=max_train)
        # 对子采样的训练数据进行截断
        data["train_captions"] = data["train_captions"][mask]
        data["train_image_idxs"] = data["train_image_idxs"][mask]
#         data["train_features"] = data["train_features"][data["train_image_idxs"]]
    return data


def decode_captions(captions, idx_to_word):
    """
    将描述的索引序列转换为文字序列（解码）。
    
    参数：
    - captions：形状为(N, T)或(T,)的数组，存储单词索引
    - idx_to_word：从索引到单词的映射字典
    
    返回：
    - decoded：解码后的文字序列列表（或单个字符串，如果输入是单条描述）
    """
    singleton = False  # 标记是否为单条描述
    if captions.ndim == 1:
        singleton = True  # 如果输入是一维数组，标记为单条
        captions = captions[None]  # 增加一个维度，变为(N=1, T)
    
    decoded = []  # 存储解码结果
    N, T = captions.shape  # N为批量大小，T为序列长度
    for i in range(N):
        words = []  # 存储单条描述的单词
        for t in range(T):
            word = idx_to_word[captions[i, t]]  # 将索引转换为单词
            if word != "<NULL>":  # 忽略填充标记
                words.append(word)
            if word == "<END>":  # 遇到结束标记则停止
                break
        decoded.append(" ".join(words))  # 将单词拼接为字符串
    
    # 如果输入是单条描述，返回单个字符串
    if singleton:
        decoded = decoded[0]
    return decoded


def sample_coco_minibatch(data, batch_size=100, split="train"):
    """
    从指定的数据集中随机采样一个小批量样本。
    
    参数：
    - data：包含数据集的字典
    - batch_size：批量大小
    - split：数据集划分（"train"或"val"）
    
    返回：
    - captions：小批量的描述索引序列
    - image_features：对应的图像特征
    - urls：对应的图像URL
    """
    # 获取指定划分的描述数量
    split_size = data["%s_captions" % split].shape[0]
    # 随机选择batch_size个样本的索引
    mask = np.random.choice(split_size, batch_size)
    # 获取选中的描述
    captions = data["%s_captions" % split][mask]
    # 获取对应的图像索引
    image_idxs = data["%s_image_idxs" % split][mask]
    # 获取对应的图像特征
    image_features = data["%s_features" % split][image_idxs]
    # 获取对应的图像URL
    urls = data["%s_urls" % split][image_idxs]
    return captions, image_features, urls
