import os
from PIL import Image  # 用于图像处理
import numpy as np
import torch
from torch.utils.data import Dataset  # 用于创建自定义数据集
from torchvision import transforms as T  # 用于图像预处理
import joblib  # 用于加载和保存数据

import clip  # 导入CLIP模型库
from tqdm.auto import tqdm  # 用于显示进度条


def get_text_augs():
    """加载文本增强数据（释义）"""
    # 获取释义字典文件路径
    fpath = os.path.join(os.path.dirname(__file__), "datasets/paraphrases_dict.pkl")
    paraphrases_dict = joblib.load(fpath)  # 加载pkl文件
    augs = {}
    # 处理每个文本的释义
    for text, paraphrases in tqdm(paraphrases_dict.items()):
        paraphrases = paraphrases.split(",")  # 按逗号分割释义
        text_augs = []
        for p in paraphrases:
            p = p.strip()  # 去除首尾空格
            if "\n" in p:  # 处理包含换行符的情况
                p = p.split("\n")[-1]
            text_augs.append(p)
        augs[text] = text_augs  # 存储增强后的文本
    return augs


class ClipEmbed:
    """使用CLIP模型对文本进行编码"""
    
    def __init__(self, device):
        # 加载CLIP模型和预处理函数
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)
        self.model = self.model.eval()  # 设置为评估模式
        self.device = device  # 计算设备

    def embed(self, text):
        """对文本进行编码，返回CLIP特征向量"""
        with torch.inference_mode():  # 禁用梯度计算，加速推理
            # 对文本进行tokenize并移动到指定设备
            text = clip.tokenize(text).to(self.device)
            # 编码文本并将结果移回CPU
            text_emb = self.model.encode_text(text)[0].cpu()
        return text_emb


class TextEmbedder:
    """文本嵌入器，用于加载、保存和处理文本嵌入，支持PCA降维"""
    
    def __init__(self):
        self.loaded = None  # 用于存储加载的数据

    def load_processed(self, data_path):
        """加载预处理的文本嵌入数据"""
        self.loaded = torch.load(data_path)

    def save_processed(self, all_texts, path):
        """保存预处理的文本嵌入数据，包括PCA组件"""
        assert not os.path.exists(path), "路径已存在，避免覆盖"
        # 初始化CLIP文本编码器
        text_embedder = ClipEmbed(device="cuda")
        all_texts = list(set(all_texts))  # 去重

        # 编码所有文本
        idx_mapping = {}  # 文本到索引的映射
        text_embeddings = []  # 存储所有文本嵌入
        for i, text in tqdm(enumerate(all_texts)):
            idx_mapping[text] = i
            text_embeddings.append(text_embedder.embed(text))
        text_embeddings = torch.stack(text_embeddings)  # 堆叠成张量

        # 执行PCA降维
        data = text_embeddings.float().numpy()
        mean = np.mean(data, axis=0)  # 计算均值向量
        centered_data = data - mean  # 中心化数据
        # 奇异值分解
        U, S, Vt = np.linalg.svd(centered_data, full_matrices=False)
        components = Vt  # 所有主成分
        components = torch.from_numpy(components).float()
        mean = torch.from_numpy(mean).float()

        # 保存处理后的数据
        torch.save(
            {
                "idx_mapping": idx_mapping,  # 文本到索引的映射
                "embs": text_embeddings,  # 文本嵌入
                "pca_components": components,  # PCA主成分
                "mean": mean,  # 均值向量
            },
            path,
        )

    def embed(self, *, text=None, emb=None, num_pca=None):
        """获取文本嵌入，可以选择应用PCA降维
        
        参数:
            text: 要嵌入的文本（与emb二选一）
            emb: 原始嵌入向量（与text二选一）
            num_pca: 若不为None，则应用PCA降维到指定维度
        """
        # 确保text和emb中有且仅有一个被提供
        assert (text is None) ^ (emb is None)

        if emb is None:
            # 根据文本获取嵌入
            emb_idx = self.loaded["idx_mapping"][text]
            emb = self.loaded["embs"][emb_idx].float()

        if num_pca is not None:
            # 应用PCA降维
            emb = self.encode_pca(emb, num_pca)

        return emb

    def encode_pca(self, emb, num_pca):
        """对嵌入向量应用PCA编码（降维）"""
        emb = emb - self.loaded["mean"]  # 中心化
        # 投影到前num_pca个主成分
        emb = self.loaded["pca_components"][:num_pca] @ emb
        return emb

    def decode_pca(self, emb):
        """对PCA降维后的向量进行解码（恢复原始空间）"""
        num_pca = emb.shape[0]
        # 投影回原始空间
        emb = self.loaded["pca_components"][:num_pca].T @ emb
        emb = emb + self.loaded["mean"]  # 恢复均值
        return emb


class EmojiDataset(Dataset):
    """表情符号数据集，继承自PyTorch的Dataset类"""
    
    def __init__(
        self,
        image_size,  # 图像尺寸
        data_path="data/emoji_data.npz",  # 数据路径
        text_emb_path="data/text_embeddings.pt",  # 文本嵌入路径
        num_text_emb_pca=None,  # 文本嵌入的PCA维度
    ):
        # 构建完整的数据路径
        data_path = os.path.join(os.path.dirname(__file__), "datasets/emoji_data.npz")
        text_emb_path = os.path.join(os.path.dirname(__file__), "datasets/text_embeddings.pt")

        self.load_augs = False  # 是否加载文本增强数据
        if self.load_augs:
            print("加载文本增强数据")
            self.text_augs = get_text_augs()  # 获取增强文本
            text_emb_path = "data/text_embeddings_augs.pt"  # 使用增强版文本嵌入

        # 加载数据集
        loaded = np.load(data_path, allow_pickle=True)
        self.data = [loaded[key].item() for key in loaded]

        # 如果使用增强数据，扩展文本列表
        if self.load_augs:
            for d in self.data:
                texts = []
                for t in d["texts"]:
                    texts.extend(self.text_augs[t])  # 添加释义
                texts = d["texts"] + texts  # 合并原始文本和释义
                d["texts"] = texts

        # 定义图像预处理管道
        self.transform = T.Compose(
            [T.Resize(image_size), T.CenterCrop(image_size), T.ToTensor()]
        )
        self.num_text_emb_pca = num_text_emb_pca  # PCA维度
        self.text_embedder = TextEmbedder()  # 初始化文本嵌入器
        self.text_embedder.load_processed(text_emb_path)  # 加载预处理的文本嵌入

    def __len__(self):
        """返回数据集大小"""
        return len(self.data)

    def __getitem__(self, idx):
        """获取指定索引的样本
        
        返回:
            img: 预处理后的图像张量
            model_kwargs: 包含文本嵌入和文本的字典
        """
        imgs = self.data[idx]["images"]  # 获取该样本的所有图像
        texts = self.data[idx]["texts"]  # 获取该样本的所有文本

        # 随机选择一张图像
        img_idx = np.random.choice(len(imgs))
        img = imgs[img_idx]

        # 预处理图像
        img = Image.fromarray(img)  # 转换为PIL图像
        img = self.transform(img)  # 应用预处理

        # 随机选择一个文本
        text = np.random.choice(texts)
        # 获取文本嵌入（可能经过PCA）
        text_emb = self.text_embedder.embed(text=text, num_pca=self.num_text_emb_pca)
        model_kwargs = {"text_emb": text_emb, "text": text}  # 模型参数
        return img, model_kwargs

    def random_model_kwargs(self, n):
        """随机获取n个样本的模型参数（文本嵌入等）"""
        # 随机选择n个索引
        idxs = np.random.choice(len(self), n)
        # 获取这些样本
        samples = [self.__getitem__(idx) for idx in idxs]
        # 整理成批次
        imgs, model_kwargs = torch.utils.data.default_collate(samples)

        return model_kwargs

    def embed_new_text(self, text, clip_embed):
        """对新文本进行编码，应用相同的PCA处理"""
        # 使用CLIP编码新文本
        text_emb = clip_embed.embed(text).float().cpu()
        # 应用PCA（如果指定）
        if self.num_text_emb_pca is not None:
            text_emb = self.text_embedder.encode_pca(text_emb, self.num_text_emb_pca)
        return text_emb
