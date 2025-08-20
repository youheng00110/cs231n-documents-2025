import copy
from einops import rearrange  # 用于维度重排的工具函数
from torch import einsum  # 用于爱因斯坦求和的函数

from torch import nn
import torch
import torch.nn.functional as F
import math


def exists(x):
    return x is not None  # 检查变量是否存在（非None）


def default(val, d):
    if exists(val):
        return val  # 如果val存在则返回val
    return d() if callable(d) else d  # 否则返回默认值d（d可为可调用对象）


def Upsample(dim, dim_out=None):
    """将图像特征分辨率上采样2倍"""
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="bilinear"),  # 双线性插值上采样
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1),  # 上采样后通过卷积调整通道
    )


def Downsample(dim, dim_out=None):
    """将图像特征分辨率下采样2倍"""
    return nn.Conv2d(dim, default(dim_out, dim), kernel_size=2, stride=2)  # 用步长为2的卷积实现下采样


class RMSNorm(nn.Module):
    """RMSNorm层，是计算高效的简化版LayerNorm"""

    def __init__(self, dim):
        super().__init__()
        self.scale = dim**0.5  # 缩放因子（维度的平方根）
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))  # 可学习的缩放参数

    def forward(self, x):
        return F.normalize(x, dim=1) * self.g * self.scale  # 沿通道维度归一化后乘以缩放参数


class SinusoidalPosEmb(nn.Module):
    """时间步的正弦位置嵌入"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim  # 嵌入维度

    def forward(self, x):
        device = x.device  # 获取输入设备
        half_dim = self.dim // 2  # 半维度（正弦和余弦各占一半）
        emb = math.log(10000) / (half_dim - 1)  # 计算指数衰减因子
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)  # 生成指数序列
        emb = x[:, None] * emb[None, :]  # 时间步与指数序列相乘
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)  # 拼接正弦和余弦分量
        return emb


class Block(nn.Module):
    """带有特征调制的卷积块"""

    def __init__(self, dim, dim_out):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding=1)  # 卷积投影
        self.norm = RMSNorm(dim_out)  # RMS归一化
        self.act = nn.GELU()  # GELU激活函数

    def forward(self, x, scale_shift=None):
        x = self.proj(x)  # 卷积操作
        x = self.norm(x)  # 归一化

        # 缩放和偏移用于调制输出。这是一种特征融合变体，比简单相加特征图更有效
        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift  # 特征调制：x = x*(scale+1) + shift

        x = self.act(x)  # 激活
        return x


class ResnetBlock(nn.Module):
    """带有上下文相关特征调制的类ResNet块"""

    def __init__(self, dim, dim_out, context_dim):
        super().__init__()
        self.dim = dim  # 输入通道数
        self.dim_out = dim_out  # 输出通道数
        self.context_dim = context_dim  # 上下文特征维度

        # MLP用于将上下文特征转换为调制参数（scale和shift）
        self.mlp = (
            nn.Sequential(nn.GELU(), nn.Linear(context_dim, dim_out * 2))
            if exists(context_dim)
            else None
        )

        self.block1 = Block(dim, dim_out)  # 第一个卷积块
        self.block2 = Block(dim_out, dim_out)  # 第二个卷积块
        # 残差连接的卷积（如果输入输出通道不同则调整通道，否则恒等映射）
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()
        self.dropout = nn.Dropout(0.1)  # dropout层

    def forward(self, x, context=None):

        scale_shift = None
        if exists(self.mlp) and exists(context):
            context = self.mlp(context)  # 上下文特征转换为调制参数
            context = rearrange(context, "b c -> b c 1 1")  # 调整维度以匹配特征图
            scale_shift = context.chunk(2, dim=1)  # 分割为scale和shift

        h = self.block1(x, scale_shift=scale_shift)  # 第一个块（带调制）
        h = self.dropout(h)  # dropout
        h = self.block2(h)  # 第二个块
        return h + self.res_conv(x)  # 残差连接


class Unet(nn.Module):
    def __init__(
        self,
        dim,  # 基础通道数
        condition_dim,  # 条件特征（如文本嵌入）的维度
        dim_mults=(1, 2, 4, 8),  # 各层通道数的倍率
        channels=3,  # 输入图像的通道数（如RGB为3）
        uncond_prob=0.2,  # 训练时丢弃条件的概率（用于无分类器引导）
    ):
        super().__init__()

        self.init_conv = nn.Conv2d(channels, dim, 3, padding=1)  # 初始卷积：将输入通道转为dim
        self.channels = channels  # 输入通道数

        # 各层的通道数，如[dim, dim*1, dim*2, ...]
        dims = [dim] + [dim * m for m in dim_mults]
        # 下采样各层的输入输出通道对，如[(d1, d2), (d2, d3), ...]
        in_out = list(zip(dims[:-1], dims[1:]))
        # 上采样各层的输入输出通道对（与下采样反向），如[(dn, dn-1), ..., (d2, d1)]
        in_out_ups = [(b, a) for a, b in reversed(in_out)]

        # 将时间步编码为上下文特征
        context_dim = dim * 4  # 上下文特征维度
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),  # 时间步的正弦嵌入
            nn.Linear(dim, context_dim),  # 线性映射
            nn.GELU(),  # 激活
            nn.Linear(context_dim, context_dim),  # 线性映射
        )

        # 将条件（如文本嵌入）编码为上下文特征
        self.condition_dim = condition_dim
        self.condition_mlp = nn.Sequential(
            nn.Linear(condition_dim, context_dim),  # 线性映射
            nn.GELU(),  # 激活
            nn.Linear(context_dim, context_dim),  # 线性映射
        )

        # 训练时丢弃条件的概率（用于无分类器引导）
        self.uncond_prob = uncond_prob

        # U-Net的下采样和上采样块列表
        # self.downs是ModuleList的ModuleList（下采样各层）
        self.downs = nn.ModuleList([])
        # self.ups是ModuleList的ModuleList（上采样各层）
        self.ups = nn.ModuleList([])

        ####################################################################
        # 下采样块
        ####################################################################
        for ind, (dim_in, dim_out) in enumerate(in_out):
            down_block = None
            ##################################################################
            # 任务：创建一个U-Net下采样层`down_block`作为ModuleList。
            # 它应该是包含3个块的ModuleList：[ResnetBlock, ResnetBlock, Downsample]。
            # 每个ResnetBlock输入输出通道均为dim_in。
            # 确保将context_dim传入每个ResnetBlock。
            # Downsample块输入通道为dim_in，输出通道为dim_out。
            # 为了能加载预训练权重，请严格遵循此ModuleList结构。
            ##################################################################

            ##################################################################
            self.downs.append(down_block)

        # 中间块（下采样到最深层后的处理）
        mid_dim = dims[-1]  # 最深层的通道数
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, context_dim=context_dim)
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, context_dim=context_dim)

        ####################################################################
        # 上采样块
        ####################################################################
        # 通过与下采样块镜像对称创建上采样块。
        # self.ups同样是ModuleList的ModuleList。
        # 每个块列表包含3个块：[Upsample, ResnetBlock, ResnetBlock]。
        for ind, (dim_in, dim_out) in enumerate(in_out_ups):
            up_block = None
            ##################################################################
            # 任务：创建一个U-Net上采样层作为ModuleList。
            # 它应该是包含3个块的ModuleList：[Upsample, ResnetBlock, ResnetBlock]。
            # 这将与对应的下采样块镜像对称。
            # 不要忘记通过将输入与下采样路径的跳跃连接拼接，使两个ResnetBlock的输入通道为2*dim_out。
            ##################################################################

            self.ups.append(up_block)
            ##################################################################

        # 最终卷积：将特征映射回输入通道数
        self.final_conv = nn.Conv2d(dim, channels, 1)

    def cfg_forward(self, x, time, model_kwargs={}):
        """无分类器引导的前向传播。model_kwargs应包含`cfg_scale`。"""

        cfg_scale = model_kwargs.pop("cfg_scale")  # 引导尺度
        print("无分类器引导尺度:", cfg_scale)
        model_kwargs = copy.deepcopy(model_kwargs)  # 深拷贝以避免修改原参数

        ##################################################################
        # 任务：使用https://arxiv.org/pdf/2207.12598的公式(6)实现无分类器引导，即
        # x = (scale + 1) * eps(x_t, cond) - scale * eps(x_t, empty)
        #
        # 你需要调用self.forward两次。
        # 对于无条件采样，传入`text_emb`=None。
        ##################################################################

        ##################################################################

        return x

    def forward(self, x, time, model_kwargs={}):
        """U-Net的前向传播。
        参数:
            x: 输入张量，形状为(batch_size, channels, height, width)。
            time: 时间步张量，形状为(batch_size,)。
            model_kwargs: 包含额外模型输入的字典，包括
                "text_emb"（文本嵌入），形状为(batch_size, condition_dim)。

        返回:
            x: 输出张量，形状为(batch_size, channels, height, width)。
        """

        if "cfg_scale" in model_kwargs:
            return self.cfg_forward(x, time, model_kwargs)  # 若需要无分类器引导，则调用对应方法

        # 嵌入时间步
        context = self.time_mlp(time)

        # 嵌入条件并与时间上下文相加
        cond_emb = model_kwargs["text_emb"]
        if cond_emb is None:
            cond_emb = torch.zeros(x.shape[0], self.condition_dim, device=x.device)  # 无条件时用零向量
        if self.training:
            # 训练时随机丢弃条件（用于无分类器引导的训练）
            mask = (torch.rand(cond_emb.shape[0]) > self.uncond_prob).float()  # 掩码：1保留，0丢弃
            mask = mask[:, None].to(cond_emb.device)  # 调整维度并移至对应设备
            cond_emb = cond_emb * mask  # 应用掩码
        context = context + self.condition_mlp(cond_emb)  # 合并时间和条件上下文

        # 初始卷积
        x = self.init_conv(x)

        ##################################################################
        # 任务：在上下文条件下处理`x`通过U-Net。
        #
        # 1. 下采样：
        #    - 将`x`通过每个下采样块并传入上下文。
        #    - 每个ResNet块后，保存输出（特征图）到列表或字典中，
        #      作为上采样路径的跳跃连接。
        #    - 确保将上下文传入每个ResNet块。
        #
        # 2. 中间层：
        #    - 将`x`通过中间块并传入上下文。
        #
        # 3. 上采样：
        #    - 将`x`通过每个上采样块并传入上下文。
        #    - 每个ResNet块前，将输入与下采样路径对应的跳跃连接拼接。
        #    - 确保将上下文传入每个ResNet块。
        ##################################################################

        ##################################################################

        # 最终卷积
        x = self.final_conv(x)

        return x
 