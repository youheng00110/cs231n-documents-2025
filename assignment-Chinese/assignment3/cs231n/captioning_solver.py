import numpy as np

from . import optim
from .coco_utils import sample_coco_minibatch  # 导入COCO数据集的小批量采样函数


class CaptioningSolver(object):
    """
    CaptioningSolver封装了训练图像描述生成模型所需的所有逻辑。
    该求解器使用optim.py中定义的不同更新规则执行随机梯度下降。

    求解器接收训练和验证数据及标签，以便定期检查模型在训练集和验证集上的分类准确率，
    从而监测过拟合情况。

    要训练模型，需先构建一个CaptioningSolver实例，向构造函数传入模型、数据集和各种选项
    （学习率、批大小等）。然后调用train()方法执行优化过程并训练模型。

    train()方法返回后，model.params将包含训练过程中在验证集上表现最佳的参数。
    此外，实例变量solver.loss_history将包含训练期间遇到的所有损失值列表，
    实例变量solver.train_acc_history和solver.val_acc_history将是包含模型在每个轮次
    对训练集和验证集准确率的列表。

    使用示例如下：

    data = load_coco_data()  # 加载COCO数据集
    model = MyAwesomeModel(hidden_dim=100)  # 初始化自定义模型
    solver = CaptioningSolver(model, data,
                    update_rule='sgd',  # 使用随机梯度下降更新规则
                    optim_config={
                      'learning_rate': 1e-3,  # 学习率
                    },
                    lr_decay=0.95,  # 学习率衰减系数
                    num_epochs=10, batch_size=100,  # 训练10轮，批大小100
                    print_every=100)  # 每100次迭代打印一次信息
    solver.train()  # 开始训练


    CaptioningSolver需要作用于一个符合以下API的模型对象：

    - model.params必须是一个字典，将字符串参数名映射到包含参数值的numpy数组。

    - model.loss(features, captions)必须是一个计算训练时损失和梯度的函数，
      具有以下输入和输出：

      输入：
      - features：图像特征的小批量数组，形状为(N, D)，其中N是批大小，D是特征维度
      - captions：这些图像对应的描述数组，形状为(N, T)，其中每个元素的范围是(0, V]，V是词汇表大小

      返回：
      - loss：标量损失值
      - grads：字典，键与self.params相同，值为损失相对于对应参数的梯度
    """

    def __init__(self, model, data, **kwargs):
        """
        构造一个新的CaptioningSolver实例。

        必需参数：
        - model：符合上述API的模型对象
        - data：从load_coco_data获取的训练和验证数据字典

        可选参数：
        - update_rule：字符串，指定optim.py中的更新规则名称。默认为'sgd'。
        - optim_config：字典，包含将传递给所选更新规则的超参数。
          每个更新规则需要不同的超参数（见optim.py），但所有更新规则都需要
          'learning_rate'参数，因此该参数必须存在。
        - lr_decay：学习率衰减系数；每轮结束后学习率将乘以该值。
        - batch_size：训练期间用于计算损失和梯度的小批量大小。
        - num_epochs：训练的轮数。
        - print_every：整数；每print_every次迭代打印一次训练损失。
        - verbose：布尔值；若设为false，则训练过程中不打印输出。
        """
        self.model = model  # 待训练的模型
        self.data = data  # 数据集

        # 解析关键字参数
        self.update_rule = kwargs.pop("update_rule", "sgd")  # 更新规则，默认sgd
        self.optim_config = kwargs.pop("optim_config", {})  # 优化器配置，默认空字典
        self.lr_decay = kwargs.pop("lr_decay", 1.0)  # 学习率衰减系数，默认不衰减
        self.batch_size = kwargs.pop("batch_size", 100)  # 批大小，默认100
        self.num_epochs = kwargs.pop("num_epochs", 10)  # 训练轮数，默认10

        self.print_every = kwargs.pop("print_every", 10)  # 打印间隔，默认10次迭代
        self.verbose = kwargs.pop("verbose", True)  # 是否打印信息，默认True

        # 若存在未识别的关键字参数，抛出错误
        if len(kwargs) > 0:
            extra = ", ".join('"%s"' % k for k in list(kwargs.keys()))
            raise ValueError("未识别的参数 %s" % extra)

        # 确保更新规则存在，然后将字符串名称替换为实际函数
        if not hasattr(optim, self.update_rule):
            raise ValueError('无效的更新规则 "%s"' % self.update_rule)
        self.update_rule = getattr(optim, self.update_rule)

        self._reset()  # 初始化记录变量

    def _reset(self):
        """
        设置一些优化过程中的记录变量。请勿手动调用此方法。
        """
        # 初始化记录变量
        self.epoch = 0  # 当前轮次
        self.best_val_acc = 0  # 最佳验证集准确率
        self.best_params = {}  # 最佳参数
        self.loss_history = []  # 损失历史记录
        self.train_acc_history = []  # 训练集准确率历史
        self.val_acc_history = []  # 验证集准确率历史

        # 为每个参数深拷贝一份优化配置
        self.optim_configs = {}
        for p in self.model.params:
            d = {k: v for k, v in self.optim_config.items()}
            self.optim_configs[p] = d

    def _step(self):
        """
        执行单次梯度更新。由train()调用，请勿手动调用。
        """
        # 获取训练数据的小批量
        minibatch = sample_coco_minibatch(
            self.data, batch_size=self.batch_size, split="train"  # 从训练集采样
        )
        captions, features, urls = minibatch  # 解包：描述、图像特征、图像URL

        # 计算损失和梯度
        loss, grads = self.model.loss(features, captions)
        self.loss_history.append(loss)  # 记录损失

        # 执行参数更新
        for p, w in self.model.params.items():
            dw = grads[p]  # 梯度
            config = self.optim_configs[p]  # 该参数的优化配置
            # 应用更新规则得到新参数和更新后的配置
            next_w, next_config = self.update_rule(w, dw, config)
            self.model.params[p] = next_w  # 更新参数
            self.optim_configs[p] = next_config  # 更新配置

    def check_accuracy(self, X, y, num_samples=None, batch_size=100):
        """
        检查模型在提供的数据上的准确率。

        输入：
        - X：数据数组，形状为(N, d_1, ..., d_k)
        - y：标签数组，形状为(N,)
        - num_samples：若不为None，则对数据进行子采样，仅在num_samples个数据点上测试模型
        - batch_size：将X和y分成该大小的批次，以避免使用过多内存

        返回：
        - acc：标量，模型正确分类的样本比例
        """
        return 0.0  # 此处返回0.0为占位，实际实现需替换

        # 可能对子数据进行采样
        N = X.shape[0]
        if num_samples is not None and N > num_samples:
            mask = np.random.choice(N, num_samples)
            N = num_samples
            X = X[mask]
            y = y[mask]

        # 分批计算预测
        num_batches = N / batch_size
        if N % batch_size != 0:
            num_batches += 1
        y_pred = []
        for i in range(num_batches):
            start = i * batch_size
            end = (i + 1) * batch_size
            scores = self.model.loss(X[start:end])  # 获取预测分数（注意：此处可能应为model.predict等方法）
            y_pred.append(np.argmax(scores, axis=1))  # 取分数最大的类别作为预测
        y_pred = np.hstack(y_pred)  # 拼接所有批次的预测结果
        acc = np.mean(y_pred == y)  # 计算准确率

        return acc

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

            # 每轮结束时，更新轮次计数器并衰减学习率
            epoch_end = (t + 1) % iterations_per_epoch == 0
            if epoch_end:
                self.epoch += 1
                # 对所有参数的学习率应用衰减
                for k in self.optim_configs:
                    self.optim_configs[k]["learning_rate"] *= self.lr_decay

