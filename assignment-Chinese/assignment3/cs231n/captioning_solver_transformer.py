import numpy as np

from . import optim
from .coco_utils import sample_coco_minibatch, decode_captions  # 导入COCO数据集相关工具函数

import torch


class CaptioningSolverTransformer(object):
    """
    CaptioningSolverTransformer封装了所有训练基于Transformer的图像描述生成模型所需的逻辑。

    要训练模型，首先需要构建一个CaptioningSolver实例，
    向构造函数传入模型、数据集和各种选项（学习率、批大小等）。
    然后调用train()方法来执行优化过程并训练模型。

    train()方法返回后，实例变量solver.loss_history将包含训练过程中遇到的所有损失值列表。

    使用示例可能如下：

    data = load_coco_data()  # 加载COCO数据集
    model = MyAwesomeTransformerModel(hidden_dim=100)  # 初始化自定义Transformer模型
    solver = CaptioningSolver(model, data,
                    optim_config={
                      'learning_rate': 1e-3,  # 学习率
                    },
                    num_epochs=10, batch_size=100,  # 训练10轮，批大小100
                    print_every=100)  # 每100次迭代打印一次信息
    solver.train()  # 开始训练


    CaptioningSolverTransformer需要作用于一个符合以下API的模型对象：

      输入：
      - features：图像特征的小批量数组，形状为(N, D)，其中N是批大小，D是特征维度
      - captions：这些图像对应的描述数组，形状为(N, T)，其中每个元素的范围是(0, V]，V是词汇表大小

      返回：
      - loss：标量损失值
      - grads：字典，键与self.params相同，值为损失相对于对应参数的梯度
    """

    def __init__(self, model, data, idx_to_word, **kwargs):
        """
        构造一个新的CaptioningSolver实例。

        必需参数：
        - model：符合上述API的模型对象
        - data：从load_coco_data获取的训练和验证数据字典

        可选参数：

        - learning_rate：优化器的学习率
        - batch_size：训练过程中用于计算损失和梯度的小批量大小
        - num_epochs：训练的轮数
        - print_every：整数；每print_every次迭代打印一次训练损失
        - verbose：布尔值；若设为false，则训练过程中不打印输出
        """
        self.model = model  # 待训练的模型
        self.data = data  # 数据集

        # 解析关键字参数
        self.learning_rate = kwargs.pop("learning_rate", 0.001)  # 学习率，默认0.001
        self.batch_size = kwargs.pop("batch_size", 100)  # 批大小，默认100
        self.num_epochs = kwargs.pop("num_epochs", 10)  # 训练轮数，默认10

        self.print_every = kwargs.pop("print_every", 10)  # 打印间隔，默认10次迭代
        self.verbose = kwargs.pop("verbose", True)  # 是否打印信息，默认True
        # 初始化Adam优化器
        self.optim = torch.optim.Adam(self.model.parameters(), self.learning_rate)

        # 若存在未识别的关键字参数，抛出错误
        if len(kwargs) > 0:
            extra = ", ".join('"%s"' % k for k in list(kwargs.keys()))
            raise ValueError("未识别的参数 %s" % extra)

        self._reset()  # 初始化记录变量

        self.idx_to_word = idx_to_word  # 索引到单词的映射（用于后续可能的解码）

    def _reset(self):
        """
        设置一些优化过程中的记录变量。请勿手动调用此方法。
        """
        # 初始化记录变量
        self.epoch = 0  # 当前轮次
        self.loss_history = []  # 损失历史记录


    def _step(self):
        """
        执行单次梯度更新。由train()调用，请勿手动调用。
        """
        # 获取训练数据的小批量
        minibatch = sample_coco_minibatch(
            self.data, batch_size=self.batch_size, split="train"  # 从训练集采样
        )
        captions, features, urls = minibatch  # 解包：描述、图像特征、图像URL

        # 处理输入输出序列：输入是描述的前T-1个词，输出是后T-1个词（语言模型任务）
        captions_in = captions[:, :-1]
        captions_out = captions[:, 1:]

        # 生成掩码：忽略输出中为NULL token的位置（NULL通常是填充符）
        mask = captions_out != self.model._null

        # 将数据转换为PyTorch张量
        t_features = torch.Tensor(features)
        t_captions_in = torch.LongTensor(captions_in)
        t_captions_out = torch.LongTensor(captions_out)
        t_mask = torch.LongTensor(mask)
        # 模型前向传播，得到预测logits
        logits = self.model(t_features, t_captions_in)

        # 计算损失，记录损失，并执行反向传播和参数更新
        loss = self.transformer_temporal_softmax_loss(logits, t_captions_out, t_mask)
        self.loss_history.append(loss.detach().numpy())  # 记录损失（ detach()分离计算图 ）
        self.optim.zero_grad()  # 清空梯度
        loss.backward()  # 反向传播计算梯度
        self.optim.step()  # 更新参数

    def train(self):
        """
        执行优化过程以训练模型。
        """
        num_train = self.data["train_captions"].shape[0]  # 训练样本数量
        # 每轮的迭代次数（向上取整）
        iterations_per_epoch = max(num_train // self.batch_size, 1)
        total_iterations = self.num_epochs * iterations_per_epoch  # 总迭代次数

        for t in range(total_iterations):
            self._step()  # 执行单次迭代

            # 若需要打印，且当前迭代是print_every的倍数
            if self.verbose and t % self.print_every == 0:
                print(
                    "(迭代 %d / %d) 损失: %f"
                    % (t + 1, total_iterations, self.loss_history[-1])
                )

            # 每轮结束时，更新轮次计数器
            epoch_end = (t + 1) % iterations_per_epoch == 0
            if epoch_end:
                self.epoch += 1

    def transformer_temporal_softmax_loss(self, x, y, mask):
        """
        用于RNN的时间序列版本softmax损失。假设对于长度为T的时间序列、
        批大小为N的小批量数据，我们在每个时间步对大小为V的词汇表进行预测。
        输入x给出所有时间步的所有词汇的得分，y给出每个时间步的真实标签索引。
        我们在每个时间步使用交叉熵损失，对所有时间步的损失求和，并在小批量上取平均。

        额外说明：由于不同长度的序列可能被组合成小批量并填充了NULL token，
        我们可能需要忽略某些时间步的模型输出。可选参数mask告诉我们哪些元素应该对损失有贡献。

        输入：
        - x：输入得分，形状为(N, T, V)
        - y：真实标签索引，形状为(N, T)，其中每个元素满足0 <= y[i, t] < V
        - mask：布尔数组，形状为(N, T)，mask[i, t]表示x[i, t]的得分是否应计入损失

        返回：
        - loss：标量损失值
        """

        N, T, V = x.shape  # N=批大小，T=时间步长，V=词汇表大小

        # 将三维输入展平为二维：(N*T, V)，方便调用交叉熵函数
        x_flat = x.reshape(N * T, V)
        # 将真实标签展平为一维：(N*T,)
        y_flat = y.reshape(N * T)
        # 将掩码展平为一维：(N*T,)
        mask_flat = mask.reshape(N * T)

        # 计算每个位置的交叉熵损失（不进行归约）
        loss = torch.nn.functional.cross_entropy(x_flat, y_flat, reduction='none')
        # 只保留掩码为True的位置的损失
        loss = torch.mul(loss, mask_flat)
        # 计算平均损失（对所有有效位置取平均）
        loss = torch.mean(loss)

        return loss