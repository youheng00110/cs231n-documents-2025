import torch
import torch.nn as nn
from tqdm.auto import tqdm
import math


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model,
        *,
        image_size,
        timesteps=1000,
        objective="pred_noise",
        beta_schedule="sigmoid",
    ):
        super().__init__()

        self.model = model
        self.channels = 3
        self.image_size = image_size
        self.objective = objective
        assert objective in {
            "pred_noise",
            "pred_x_start",
        }, "目标函数必须是pred_noise（预测噪声）或pred_x_start（预测初始图像）"

        # 一个辅助函数，将一些常量注册为缓冲区，以确保它们与模型参数在同一设备上
        # 参见https://pytorch.org/docs/stable/generated/torch.nn.Module.html
        # 每个缓冲区都可以通过`self.name`访问
        register_buffer = lambda name, val: self.register_buffer(name, val.float())

        #############################################################################
        # 噪声调度的beta和alpha值
        #############################################################################
        betas = get_beta_schedule(beta_schedule, timesteps)
        self.num_timesteps = int(betas.shape[0])
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)  # alpha_bar_t
        register_buffer("betas", betas)  # 可以通过self.betas访问
        register_buffer("alphas", alphas)  # 可以通过self.alphas访问
        register_buffer("alphas_cumprod", alphas_cumprod)  # self.alphas_cumprod

        #############################################################################
        # 在x_t、x_0和噪声之间转换所需的其他系数
        # 注意，根据公式(4)及其在公式(14)中的重参数化，
        # x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        # 其中噪声采样自N(0, 1)
        #############################################################################
        register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )

        #############################################################################
        # 论文公式(6)和(7)中的后验分布q(x_{t-1} | x_t, x_0)的参数
        #############################################################################
        # alpha_bar_{t-1}
        alphas_cumprod_prev = nn.functional.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )
        posterior_var = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        posterior_std = torch.sqrt(posterior_var.clamp(min=1e-20))
        register_buffer("posterior_std", posterior_std)

        #################################################################
        # 损失权重
        #################################################################
        snr = alphas_cumprod / (1 - alphas_cumprod)
        loss_weight = torch.ones_like(snr) if objective == "pred_noise" else snr
        register_buffer("loss_weight", loss_weight)

    def normalize(self, img):
        return img * 2 - 1

    def unnormalize(self, img):
        return (img + 1) * 0.5

    def predict_start_from_noise(self, x_t, t, noise):
        """根据论文公式(14)从x_t和噪声中获取x_start
        参数:
            x_t: (b, *) 张量。带噪声的图像。
            t: (b,) 张量。时间步。
            noise: (b, *) 张量。来自N(0, 1)的噪声。
        返回:
            x_start: (b, *) 张量。初始图像。
        """
        x_start = None
        ####################################################################
        # 任务:
        # 根据公式(4)和公式(14)转换x_t和噪声以得到x_start。
        # 查看`__init__`方法中的系数，并使用`extract`函数。
        ####################################################################

        ####################################################################
        return x_start

    def predict_noise_from_start(self, x_t, t, x_start):
        """根据论文公式(14)从x_t和x_start中获取噪声
        参数:
            x_t: (b, *) 张量。带噪声的图像。
            t: (b,) 张量。时间步。
            x_start: (b, *) 张量。初始图像。
        返回:
            pred_noise: (b, *) 张量。预测的噪声。
        """
        pred_noise = None
        ####################################################################
        # 任务:
        # 根据公式(4)和公式(14)转换x_t和噪声以得到x_start。
        # 查看`__init__`方法中的系数，并使用`extract`函数。
        ####################################################################

        ####################################################################
        return pred_noise

    def q_posterior(self, x_start, x_t, t):
        """根据论文公式(6)和(7)获取后验分布q(x_{t-1} | x_t, x_0)
        参数:
            x_start: (b, *) 张量。预测的初始图像。
            x_t: (b, *) 张量。带噪声的图像。
            t: (b,) 张量。时间步。
        返回:
            posterior_mean: (b, *) 张量。后验分布的均值。
            posterior_std: (b, *) 张量。后验分布的标准差。
        """
        posterior_mean = None
        posterior_std = None
        ####################################################################
        # 我们已经为你实现了这个方法。
        c1 = extract(self.posterior_mean_coef1, t, x_t.shape)
        c2 = extract(self.posterior_mean_coef2, t, x_t.shape)
        posterior_mean = c1 * x_start + c2 * x_t
        posterior_std = extract(self.posterior_std, t, x_t.shape)
        ####################################################################
        return posterior_mean, posterior_std

    @torch.no_grad()
    def p_sample(self, x_t, t: int, model_kwargs={}):
        """根据论文公式(6)从p(x_{t-1} | x_t)中采样。仅在推理时使用。
        参数:
            x_t: (b, *) 张量。带噪声的图像。
            t: int。采样时间步。
            model_kwargs: 模型的额外参数。
        返回:
            x_tm1: (b, *) 张量。采样得到的图像。
        """
        t = torch.full((x_t.shape[0],), t, device=x_t.device, dtype=torch.long)  # (b,)
        x_tm1 = None  # 从p(x_{t-1} | x_t)中采样x_{t-1}

        ##################################################################
        # 任务: 根据公式(6)实现采样步骤p(x_{t-1} | x_t):
        #
        # 步骤:
        #   1. 通过调用self.model并传入适当的参数来获取模型预测。
        #   2. 根据self.objective，模型输出可以是噪声或x_start。
        #      可以根据需要调用self.predict_start_from_noise或
        #      self.predict_noise_from_start来恢复另一个。
        #   3. 将预测的x_start裁剪到有效范围[-1, 1]。这确保了
        #      去噪迭代过程中的生成稳定性。
        #   4. 使用self.q_posterior获取q(x_{t-1} | x_t, x_0)的均值和标准差，
        #      并采样x_{t-1}。
        ##################################################################
        
        ##################################################################

        return x_tm1

    @torch.no_grad()
    def sample(self, batch_size=16, return_all_timesteps=False, model_kwargs={}):

        shape = (batch_size, self.channels, self.image_size, self.image_size)
        img = torch.randn(shape, device=self.betas.device)
        imgs = [img]

        for t in tqdm(
            reversed(range(0, self.num_timesteps)),
            desc="采样循环时间步",
            total=self.num_timesteps,
        ):
            img = self.p_sample(img, t, model_kwargs=model_kwargs)
            imgs.append(img)

        res = img if not return_all_timesteps else torch.stack(imgs, dim=1)
        res = self.unnormalize(res)
        return res

    def q_sample(self, x_start, t, noise):
        """根据论文公式(4)从q(x_t | x_0)中采样。

        参数:
            x_start: (b, *) 张量。初始图像。
            t: (b,) 张量。时间步。
            noise: (b, *) 张量。来自N(0, 1)的噪声。
        返回:
            x_t: (b, *) 张量。带噪声的图像。
        """

        x_t = None
        ####################################################################
        # 任务:
        # 根据论文公式(4)实现从q(x_t | x_0)的采样。
        # 提示: (1) 查看`__init__`方法中的预计算系数。
        # (2) 使用上面定义的`extract`函数提取给定时间步`t`的系数。
        # (3) 回想一下，从N(mu, sigma^2)中采样可以表示为：x_t = mu + sigma * noise，其中噪声采样自N(0, 1)。
        # 大约3行代码。
        ####################################################################

        ####################################################################
        return x_t

    def p_losses(self, x_start, model_kwargs={}):
        b, nts = x_start.shape[0], self.num_timesteps
        t = torch.randint(0, nts, (b,), device=x_start.device).long()  # (b,)
        x_start = self.normalize(x_start)  # (b, *)
        noise = torch.randn_like(x_start)  # (b, *)
        target = noise if self.objective == "pred_noise" else x_start  # (b, *)
        loss_weight = extract(self.loss_weight, t, target.shape)  # (b, *)
        loss = None

        ####################################################################
        # 任务:
        # 根据论文公式(14)实现损失函数。
        # 首先，使用`q_sample`函数从q(x_t | x_0)中采样x_t。
        # 然后，通过调用self.model并传入适当的参数来获取模型预测。
        # 最后，计算加权MSE损失。
        # 大约3-4行代码。
        ####################################################################

        ####################################################################

        return loss


def extract(a, t, x_shape):
    """
    根据给定的时间步提取相应的系数值。

    该函数根据给定的时间步`t`从系数张量`a`中收集值，并将它们重塑为所需的形状，以便
    它能与给定形状`x_shape`的张量进行广播。

    参数:
        a (torch.Tensor): 形状为(T,)的张量，包含所有时间步的系数值。
        t (torch.Tensor): 形状为(b,)的张量，表示批次中每个样本的时间步。
        x_shape (tuple): 输入图像张量的形状，通常为(b, c, h, w)。

    返回:
        torch.Tensor: 形状为(b, 1, 1, 1)的张量，包含从a中提取的对应每个批次元素时间步的系数值，并已适当重塑。
    """
    b, *_ = t.shape  # 从时间步张量中提取批次大小
    out = a.gather(-1, t)  # 根据`t`从`a`中收集系数值
    out = out.reshape(
        b, *((1,) * (len(x_shape) - 1))
    )  # 重塑为(b, 1, 1, 1)以进行广播
    return out


def linear_beta_schedule(timesteps):
    """
    线性调度，在原始ddpm论文中提出
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    余弦调度
    提出于https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def sigmoid_beta_schedule(timesteps, start=-3, end=3, tau=1, clamp_min=1e-5):
    """
    sigmoid调度
    提出于https://arxiv.org/abs/2212.11972 - 图8
    在训练大于64x64的图像时效果更好
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (
        v_end - v_start
    )
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def get_beta_schedule(beta_schedule, timesteps):
    if beta_schedule == "linear":
        beta_schedule_fn = linear_beta_schedule
    elif beta_schedule == "cosine":
        beta_schedule_fn = cosine_beta_schedule
    elif beta_schedule == "sigmoid":
        beta_schedule_fn = sigmoid_beta_schedule
    else:
        raise ValueError(f"未知的beta调度 {beta_schedule}")

    betas = beta_schedule_fn(timesteps)
    return betas
