import numpy as np

from . import optim
from .coco_utils import sample_coco_minibatch, decode_captions  # 导入COCO数据集相关工具函数

import torch  # 导入PyTorch库


class CaptioningSolverPytorch(object):
    """
    CaptioningSolverPytorch类封装了所有训练基于PyTorch的图像描述模型所需的逻辑。

    要训练模型，首先需要创建一个CaptioningSolver实例，
    向构造函数传递模型、数据集和各种选项（学习率、批量大小等）。
    然后调用train()方法来执行优化过程并训练模型。

    train()方法返回后，实例变量solver.loss_history将包含训练过程中遇到的所有损失值。

    使用示例可能如下：

    data = load_coco_data()
    model = MyAwesomeModel(hidden_dim=100)
    solver = CaptioningSolver(model, data,
                    optim_config={
                      'learning_rate': 1e-3,
                    },
                    num_epochs=10, batch_size=100,
                    print_every=100)
    solver.train()


    CaptioningSolverPytorch适用于符合以下API的模型对象：

      输入：
      - features：图像特征的小批量数组，形状为(N, D)
      - captions：这些图像的描述数组，形状为(N, T)，其中每个元素的范围是(0, V]

      返回：
      - loss：标量损失值
      - grads：字典，键与self.params相同，将参数名映射到损失相对于这些参数的梯度
    """

    def __init__(self, model, data,** kwargs):
        """
        创建一个新的CaptioningSolver实例。

        必需的参数：
        - model：符合上述API的模型对象
        - data：来自load_coco_data的训练和验证数据字典

        可选参数：

        - learning_rate：优化器的学习率。
        - batch_size：训练期间用于计算损失和梯度的小批量大小。
        - num_epochs：训练运行的轮数。
        - print_every：整数；每print_every次迭代将打印训练损失。
        - verbose：布尔值；如果设为false，则训练期间不会打印任何输出。
        """
        self.model = model  # 待训练的模型
        self.data = data    # 数据集

        # 解析关键字参数
        self.learning_rate = kwargs.pop("learning_rate", 0.001)  # 学习率，默认0.001
        self.batch_size = kwargs.pop("batch_size", 100)          # 批量大小，默认100
        self.num_epochs = kwargs.pop("num_epochs", 10)           # 训练轮数，默认10

        self.print_every = kwargs.pop("print_every", 10)  # 打印间隔，默认每10次迭代
        self.verbose = kwargs.pop("verbose", True)        # 是否打印信息，默认True
        # 初始化优化器（Adam）
        self.optim = torch.optim.Adam(list(model.params.values()), self.learning_rate)

        # 如果有额外的关键字参数，抛出错误
        if len(kwargs) > 0:
            extra = ", ".join('"%s"' % k for k in list(kwargs.keys()))
            raise ValueError("未识别的参数 %s" % extra)

        self._reset()  # 重置内部状态

    def _reset(self):
        """
        设置一些用于优化的记录变量。不要手动调用此方法。
        """
        # 设置一些记录变量
        self.epoch = 0  # 当前轮数
        self.loss_history = []  # 损失历史记录


    def _step(self):
        """
        执行单次梯度更新。由train()调用，不应手动调用。
        """
        # 获取训练数据的小批量
        minibatch = sample_coco_minibatch(
            self.data, batch_size=self.batch_size, split="train"
        )
        captions, features, urls = minibatch  # 描述、图像特征、图像URL

        # 将数据转换为PyTorch张量（captions需要是长整数类型）
        captions = torch.from_numpy(captions).long()
        features = torch.from_numpy(features)
        # 计算损失
        loss = self.model.loss(features, captions)
        # 清零梯度
        self.optim.zero_grad()
        # 反向传播计算梯度
        loss.backward()
        # 更新参数
        self.optim.step()
        # 记录损失值
        self.loss_history.append(loss.detach().numpy())

    def train(self):
        """
        执行优化以训练模型。
        """
        # 为模型所有参数开启梯度跟踪
        for k, v in self.model.params.items():
          v.requires_grad_()

        # 计算训练样本数量和每轮的迭代次数
        num_train = self.data["train_captions"].shape[0]
        iterations_per_epoch = max(num_train // self.batch_size, 1)
        num_iterations = self.num_epochs * iterations_per_epoch  # 总迭代次数

        # 执行训练迭代
        for t in range(num_iterations):
            self._step()  # 单次迭代（前向传播+反向传播+参数更新）

            # 可能打印训练损失
            if self.verbose and t % self.print_every == 0:
                print(
                    "(迭代 %d / %d) 损失: %f"
                    % (t + 1, num_iterations, self.loss_history[-1])
                )

            # 在每轮结束时，递增轮计数器
            epoch_end = (t + 1) % iterations_per_epoch == 0
            if epoch_end:
                self.epoch += 1

        # 训练结束后，关闭所有参数的梯度跟踪
        for k, v in self.model.params.items():
          v.requires_grad_(False)
