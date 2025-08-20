import torch
import torch.nn as nn
import numpy as np
import clip  # 导入CLIP模型库
from PIL import Image  # 导入PIL库，用于图像处理
import tensorflow_datasets as tfds  # 导入tensorflow_datasets，用于加载DAVIS数据集
from torchvision import transforms as T  # 导入torchvision的transforms，用于图像预处理
import cv2  # 导入OpenCV库，用于图像处理
from tqdm.auto import tqdm  # 导入tqdm，用于显示进度条


def get_similarity_no_loop(text_features, image_features):
    """
    计算文本特征向量和图像特征向量之间的成对余弦相似度。

    参数:
        text_features (torch.Tensor): 形状为(N, D)的张量，N为文本数量，D为特征维度。
        image_features (torch.Tensor): 形状为(M, D)的张量，M为图像数量，D为特征维度。

    返回:
        torch.Tensor: 形状为(N, M)的相似度矩阵，其中每个元素(i, j)表示
        text_features[i]与image_features[j]之间的余弦相似度。
    """
    similarity = None
    ############################################################################
    # 任务：计算余弦相似度。请勿使用for循环。                                 #
    ############################################################################

    ############################################################################
    #                             代码结束部分                                  #
    ############################################################################

    return similarity


@torch.no_grad()  # 禁用梯度计算，用于推理阶段
def clip_zero_shot_classifier(clip_model, clip_preprocess, images,
                              class_texts, device):
    """使用CLIP模型执行零样本图像分类。

    参数:
        clip_model (torch.nn.Module): 预训练的CLIP模型，用于编码图像和文本。
        clip_preprocess (Callable): 图像编码前应用于每个图像的预处理函数。
        images (List[np.ndarray]): 输入图像的列表，每个图像为NumPy数组（形状为H x W x C，uint8类型）。
        class_texts (List[str]): 零样本分类的类别标签字符串列表。
        device (torch.device): 计算设备（CPU或GPU）。将text_tokens传入clip_model前需移至该设备。

    返回:
        List[str]: 每个图像的预测类别标签，从给定的class_texts中选择。
    """
    
    pred_classes = []

    ############################################################################
    # 任务：为图像确定类别标签。                                              #
    ############################################################################

    ############################################################################
    #                             代码结束部分                                  #
    ############################################################################

    return pred_classes
  

class CLIPImageRetriever:
    """
    一个使用CLIP的简单图像检索系统。
    """
    
    @torch.no_grad()  # 禁用梯度计算
    def __init__(self, clip_model, clip_preprocess, images, device):
        """
        参数:
          clip_model (torch.nn.Module): 预训练的CLIP模型。
          clip_preprocess (Callable): 图像预处理函数。
          images (List[np.ndarray]): 图像列表，每个图像为NumPy数组（形状为H x W x C）。
          device (torch.device): 模型运行的设备。
        """
        ############################################################################
        # 任务：存储所有在retrieve方法中需要使用的对象变量。                        #
        # 注意：此处应一次性处理所有图像，避免对每个文本查询重复计算。               #
        # 对于计算最优的实现，可能不会使用上述的相似度函数。                         #
        ############################################################################

        ############################################################################
        #                             代码结束部分                                  #
        ############################################################################
        pass
    
    @torch.no_grad()  # 禁用梯度计算
    def retrieve(self, query: str, k: int = 2):
        """
        检索与输入文本最相似的前k张图像的索引。
        可使用torch.Tensor.topk方法。

        参数:
            query (str): 文本查询。
            k (int): 返回前k张图像。

        返回:
            List[int]: 前k张最相似图像的索引。
        """
        top_indices = []
        ############################################################################
        # 任务：检索前k张图像的索引。                                              #
        ############################################################################

        ############################################################################
        #                             代码结束部分                                  #
        ############################################################################
        return top_indices

  
class DavisDataset:
    """DAVIS数据集处理类，用于加载和处理DAVIS视频分割数据集。"""
    
    def __init__(self):
        # 加载DAVIS 480p验证集（不使用监督模式）
        self.davis = tfds.load('davis/480p', split='validation', as_supervised=False)
        # 定义图像预处理管道：调整大小、转换为Tensor、标准化
        self.img_tsfm = T.Compose([
            T.Resize((480, 480)), T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        
      
    def get_sample(self, index):
        """获取指定索引的视频样本（帧和掩码）。

        参数:
            index: 视频样本的索引。

        返回:
            frames: 视频帧的列表（每个帧为NumPy数组）。
            masks: 对应的分割掩码列表。
        """
        assert index < len(self.davis), "索引超出数据集范围"
        ds_iter = iter(tfds.as_numpy(self.davis))  # 将数据集转换为NumPy迭代器
        for i in range(index + 1):
            video = next(ds_iter)  # 获取第index个视频
        frames, masks = video['video']['frames'], video['video']['segmentations']
        print(f"视频 {video['metadata']['video_name'].decode()} 包含 {len(frames)} 帧")
        return frames, masks
    
    def process_frames(self, frames, dino_model, device):
        """使用DINO模型处理视频帧，提取特征。

        参数:
            frames: 视频帧列表。
            dino_model: 预训练的DINO模型。
            device: 计算设备。

        返回:
            res: 处理后的特征张量（形状为[帧数, 特征数, 特征维度]）。
        """
        res = []
        for f in frames:
            # 预处理图像并移至指定设备
            f = self.img_tsfm(Image.fromarray(f))[None].to(device)
            with torch.no_grad():  # 禁用梯度计算
                # 获取中间层特征（取第1层）
                tok = dino_model.get_intermediate_layers(f, n=1)[0]
            res.append(tok[0, 1:])  # 去除class token，保留补丁特征

        res = torch.stack(res)  # 堆叠所有帧的特征
        return res
    
    def process_masks(self, masks, device):
        """处理分割掩码，调整大小并展平。

        参数:
            masks: 掩码列表。
            device: 计算设备。

        返回:
            res: 处理后的掩码张量。
        """
        res = []
        for m in masks:
            # 调整掩码大小为60x60，展平为一维
            m = cv2.resize(m, (60, 60), cv2.INTER_NEAREST)
            res.append(torch.from_numpy(m).long().flatten(-2, -1))
        res = torch.stack(res).to(device)  # 堆叠并移至设备
        return res
    
    def mask_frame_overlay(self, processed_mask, frame):
        """将处理后的掩码叠加到原始帧上。

        参数:
            processed_mask: 处理后的掩码。
            frame: 原始图像帧。

        返回:
            overlay: 叠加后的图像。
        """
        H, W = frame.shape[:2]
        mask = processed_mask.detach().cpu().numpy()  #  detach并移至CPU转为NumPy
        mask = mask.reshape((60, 60))  # 重塑为60x60
        # 调整掩码大小与原始帧一致
        mask = cv2.resize(
            mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
        overlay = create_segmentation_overlay(mask, frame.copy())  # 创建叠加效果
        return overlay
        


def create_segmentation_overlay(segmentation_mask, image, alpha=0.5):
    """
    在RGB图像上生成带颜色的分割叠加层。

    参数:
        segmentation_mask (np.ndarray): 2D数组（形状为H, W），包含类别索引。
        image (np.ndarray): 3D数组（形状为H, W, 3），RGB图像。
        alpha (float): 叠加层的透明度（0=仅图像，1=仅掩码）。

    返回:
        np.ndarray: 带分割叠加层的图像（形状为H, W, 3，uint8类型）。
    """
    # 检查分割掩码和图像的尺寸是否匹配
    assert segmentation_mask.shape[:2] == image.shape[:2], "分割掩码与图像尺寸不匹配"
    assert image.dtype == np.uint8, "图像必须为uint8类型"

    # 为每个类别生成确定性颜色（使用固定随机种子）
    def generate_colormap(n):
        np.random.seed(42)  # 确保颜色生成可复现
        colormap = np.random.randint(0, 256, size=(n, 3), dtype=np.uint8)
        return colormap

    colormap = generate_colormap(10)  # 生成10个类别的颜色映射

    # 为分割掩码创建彩色图像
    seg_color = colormap[segmentation_mask]  # 形状为(H, W, 3)

    # 与原始图像混合
    overlay = cv2.addWeighted(image, 1 - alpha, seg_color, alpha, 0)

    return overlay


def compute_iou(pred, gt, num_classes):
    """计算平均交并比（IoU）。

    参数:
        pred: 预测的类别标签。
        gt: 真实的类别标签。
        num_classes: 类别数量。

    返回:
        iou: 平均IoU。
    """
    iou = 0
    for ci in range(num_classes):
        p = pred == ci  # 预测为类别ci的区域
        g = gt == ci    # 真实为类别ci的区域
        # 计算当前类别的IoU并累加（加1e-8避免除零）
        iou += (p & g).sum() / ((p | g).sum() + 1e-8)
    return iou / num_classes  # 返回平均IoU


class DINOSegmentation:
    """基于DINO特征的分割模型，用于将DINO特征向量分类为分割类别。"""
    
    def __init__(self, device, num_classes: int, inp_dim : int = 384):
        """
        初始化DINOSegmentation模型。

        定义一个轻量级神经网络，用于将DINO特征向量分类为分割类别。
        包含模型初始化、优化器和损失函数设置。

        参数:
            device (torch.device): 模型运行的设备（CPU或CUDA）。
            num_classes (int): 分割类别数量。
            inp_dim (int, optional): 输入DINO特征的维度。
        """

        ############################################################################
        # 任务：定义一个轻量级PyTorch模型、优化器和损失函数，用于将每个DINO特征向量  #
        # 分类为分割类别。可以是线性层或两层神经网络。                             #
        ############################################################################

        ############################################################################
        #                             代码结束部分                                  #
        ############################################################################
        pass

    def train(self, X_train, Y_train, num_iters=500):
        """使用提供的训练数据训练分割模型。

        参数:
            X_train (torch.Tensor): 输入特征向量（形状为[N, D]）。
            Y_train (torch.Tensor): 真实标签（形状为[N,]）。
            num_iters (int, optional): 优化步骤数。
        """
        ############################################################################
        # 任务：训练模型，执行`num_iters`步优化。                                  #
        ############################################################################

        ############################################################################
        #                             代码结束部分                                  #
        ############################################################################
        pass
    
    @torch.no_grad()  # 禁用梯度计算
    def inference(self, X_test):
        """对给定的测试DINO特征向量执行推理。

        参数:
            X_test (torch.Tensor): 输入特征向量（形状为[N, D]）。

        返回:
            torch.Tensor of shape (N,): 预测的类别索引。
        """
        pred_classes = None
        ############################################################################
        # 任务：对测试特征进行推理，返回预测的类别。                                #
        ############################################################################

        ############################################################################
        #                             代码结束部分                                  #
        ############################################################################
        return pred_classes