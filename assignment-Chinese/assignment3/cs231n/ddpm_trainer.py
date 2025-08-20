import math
import os
import torch
from torch.utils.data import DataLoader  # 用于数据加载
from torch.optim import Adam  # 导入Adam优化器
from torchvision.utils import save_image  # 用于保存图像
from tqdm.auto import tqdm  # 用于显示进度条


def cycle(dl):
    """循环迭代数据加载器，不断生成批次数据"""
    while True:
        for data in dl:
            yield data


class Trainer(object):
    """扩散模型的训练器类，负责模型的训练、保存、加载和采样等功能"""
    
    def __init__(
        self,
        diffusion_model,  # 扩散模型实例
        dataset,  # 训练数据集
        device,  # 训练设备（CPU或GPU）
        *,
        train_batch_size=256,  # 训练批次大小
        train_lr=1e-3,  # 训练学习率
        weight_decay=0.0,  # 权重衰减系数
        train_num_steps=100000,  # 总训练步数
        adam_betas=(0.9, 0.99),  # Adam优化器的beta参数
        sample_every=1000,  # 每多少步生成一次样本
        save_every=10000,  # 每多少步保存一次模型
        results_folder=None,  # 结果保存文件夹
    ):
        super().__init__()

        assert results_folder is not None, "必须指定结果文件夹"
        self.diffusion_model = diffusion_model  # 扩散模型

        self.device = device  # 设备
        self.num_samples = 25  # 每次采样的图像数量
        self.save_every = save_every  # 模型保存间隔
        self.sample_every = sample_every  # 采样间隔
        self.batch_size = train_batch_size  # 批次大小
        self.train_num_steps = train_num_steps  # 总训练步数

        # 数据集和数据加载器
        self.ds = dataset
        dl = DataLoader(
            self.ds,
            batch_size=train_batch_size,
            shuffle=True,  # 打乱数据
            pin_memory=False,  # 不使用 pinned memory
            num_workers=0,  # 不使用多进程加载
        )
        self.dl = cycle(dl)  # 循环数据加载器

        # 优化器
        self.opt = Adam(
            diffusion_model.parameters(),  # 模型参数
            lr=train_lr,  # 学习率
            betas=adam_betas,  # beta参数
            weight_decay=weight_decay,  # 权重衰减
        )

        self.results_folder = results_folder  # 结果文件夹
        os.makedirs(self.results_folder, exist_ok=True)  # 创建文件夹（若不存在）

        # 步数计数器状态
        self.step = 0  # 当前训练步数

    def save(self, milestone):
        """保存模型 checkpoint

        参数:
            milestone: 保存的里程碑（通常是当前步数）
        """
        ckpt_path = os.path.join(self.results_folder, f"model-{milestone}.pt")
        print(f"将模型保存到 {ckpt_path}。")
        data = {
            "step": self.step,  # 当前步数
            "model": self.diffusion_model.state_dict(),  # 模型参数
            "opt": self.opt.state_dict(),  # 优化器状态
        }

        torch.save(data, ckpt_path)  # 保存数据

    def load(self, milestone):
        """加载模型 checkpoint

        参数:
            milestone: 要加载的里程碑（通常是保存时的步数）
        """
        ckpt_path = os.path.join(self.results_folder, f"model-{milestone}.pt")
        print(f"从 {ckpt_path} 加载模型。")
        data = torch.load(ckpt_path, map_location=self.device, weights_only=True)  # 加载数据

        self.diffusion_model.load_state_dict(data["model"])  # 加载模型参数
        self.step = data["step"]  # 恢复步数
        self.opt.load_state_dict(data["opt"])  # 加载优化器状态

        # 将模型和优化器移动到同一设备
        device = self.device
        self.diffusion_model.to(device)
        for state in self.opt.state.values():
            if isinstance(state, torch.Tensor):
                state.data = state.data.to(device)
            elif isinstance(state, dict):
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)

    def train(self):
        """训练模型的主函数"""
        device = self.device
        self.diffusion_model.to(device)  # 将模型移动到设备

        all_losses = []  # 记录所有损失值

        # 创建进度条，初始值为当前步数，总步数为训练总步数
        with tqdm(initial=self.step, total=self.train_num_steps) as pbar:

            while self.step < self.train_num_steps:  # 当未达到总训练步数时

                data, model_kwargs = next(self.dl)  # 获取下一个批次数据
                data = data.to(device)  # 将数据移动到设备
                model_kwargs["text_emb"] = model_kwargs["text_emb"].to(device)  # 将文本嵌入移动到设备

                self.opt.zero_grad()  # 清空梯度
                # 计算扩散模型的损失
                loss = self.diffusion_model.p_losses(data, model_kwargs=model_kwargs)
                loss.backward()  # 反向传播计算梯度
                # 梯度裁剪，防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(self.diffusion_model.parameters(), 1.0)
                self.opt.step()  # 更新参数

                pbar.set_description(f"损失: {loss.item():.4f}")  # 更新进度条描述
                all_losses.append(loss.item())  # 记录损失

                self.step += 1  # 步数加1

                # 每save_every步保存一次模型
                if self.step % self.save_every == 0:
                    self.save(self.step)

                # 每sample_every步生成一次样本
                if self.step % self.sample_every == 0:
                    self.diffusion_model.eval()  # 切换到评估模式

                    with torch.no_grad():  # 禁用梯度计算
                        # 获取随机的模型参数（如文本嵌入）
                        model_kwargs = self.ds.random_model_kwargs(self.num_samples)
                        model_kwargs["text_emb"] = model_kwargs["text_emb"].to(device)

                        # 生成样本
                        all_images = self.diffusion_model.sample(
                            batch_size=self.num_samples, model_kwargs=model_kwargs
                        )

                    # 保存生成的样本图像
                    save_image(
                        all_images,
                        os.path.join(self.results_folder, f"sample-{self.step}.png"),
                        nrow=int(math.sqrt(self.num_samples)),  # 每行显示的图像数量
                    )

                pbar.update(1)  # 进度条更新

        return all_losses  # 返回所有损失值列表
