import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet50


class Model(nn.Module):
    def __init__(self, feature_dim=128):
        super(Model, self).__init__()

        self.f = []
        # 遍历resnet50模型的子模块
        for name, module in resnet50().named_children():
            # 替换卷积层：将7x7卷积改为3x3卷积，调整步长和填充
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            # 排除线性层和最大池化层，将其余层添加到特征提取器
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.f.append(module)
        # 编码器：由筛选后的ResNet50层组成
        self.f = nn.Sequential(*self.f)
        # 投影头：将编码器输出映射到低维特征空间
        self.g = nn.Sequential(nn.Linear(2048, 512, bias=False), nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))

    def forward(self, x):
        # 通过编码器提取特征
        x = self.f(x)
        # 展平特征图
        feature = torch.flatten(x, start_dim=1)
        # 通过投影头得到输出
        out = self.g(feature)
        # 返回归一化的特征和输出
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)
