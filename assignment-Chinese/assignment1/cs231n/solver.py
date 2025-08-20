"""
训练器类，封装了训练分类模型所需的所有逻辑。
"""
from __future__ import print_function, division
from future import standard_library

standard_library.install_aliases()
from builtins import range
from builtins import object
import os
import pickle as pickle

import numpy as np

from cs231n import optim


class Solver(object):
    """
    Solver 封装了训练分类模型所需的所有逻辑，包括随机梯度下降（SGD）和多种优化器。
    Solver 接收训练和验证数据及标签，可定期检查训练和验证集上的分类准确率，
    以监控过拟合情况。

    训练模型时，你将首先创建一个 Solver 实例，传入模型、数据集及各种选项（学习率、批量大小等），
    然后调用 train() 方法运行优化过程并训练模型。

    train() 方法返回后，model.params 将包含在训练过程中验证集上表现最佳的参数。
    此外，实例变量 solver.loss_history 将包含训练过程中遇到的所有损失，
    实例变量 solver.train_acc_history 和 solver.val_acc_history 将是模型在每个 epoch
    上训练集和验证集准确率的列表。

    示例用法如下：

    data = {
      'X_train': # 训练数据
      'y_train': # 训练标签
      'X_val': # 验证数据
      'y_val': # 验证标签
    }
    model = MyAwesomeModel(hidden_size=100, reg=10)
    solver = Solver(model, data,
                    update_rule='sgd',
                    optim_config={
                      'learning_rate': 1e-4,
                    },
                    lr_decay=0.95,
                    num_epochs=5, batch_size=200,
                    print_every=100)
    solver.train()


    Solver 作用于符合以下 API 的模型对象：

    - model.params 必须是字典，将字符串参数名称映射到包含参数值的 NumPy 数组。

    - model.loss(X, y) 必须是函数，在训练时计算损失和梯度，在测试时计算分类分数，
      输入和输出如下：

      输入：
      - X: 形状为 (N, d_1, ..., d_k) 的小批量输入数据数组
      - y: 形状为 (N,) 的标签数组，y[i] 是 X[i] 的标签

      返回：
      如果 y 为 None，则运行测试时前向传播并返回：
      - scores: 形状为 (N, C) 的数组，scores[i, c] 是 X[i] 属于类别 c 的分数。

      如果 y 不为 None，则运行训练时前向和反向传播并返回元组：
      - loss: 标量损失值
      - grads: 字典，键与 self.params 相同，将参数名称映射到损失相对于这些参数的梯度。
    """

    def __init__(self, model, data, **kwargs):
        """
        构造一个新的 Solver 实例。

        必要参数：
        - model: 符合上述 API 的模型对象
        - data: 包含训练和验证数据的字典，包含：
          'X_train': 形状为 (N_train, d_1, ..., d_k) 的训练图像数组
          'X_val': 形状为 (N_val, d_1, ..., d_k) 的验证图像数组
          'y_train': 形状为 (N_train,) 的训练图像标签数组
          'y_val': 形状为 (N_val,) 的验证图像标签数组

        可选参数：
        - update_rule: optim.py 中的更新规则名称，默认为 'sgd'。
        - optim_config: 包含超参数的字典，将传递给所选更新规则。
          每种更新规则需要不同的超参数（见 optim.py），但所有更新规则都需要
          'learning_rate' 参数，因此必须始终存在。
        - lr_decay: 学习率衰减标量；每个 epoch 后，学习率乘以该值。
        - batch_size: 计算损失和梯度时使用的小批量大小。
        - num_epochs: 训练时运行的 epoch 数量。
        - print_every: 整数；每 print_every 次迭代打印训练损失。
        - verbose: 布尔值；如果设置为 False，则训练期间不打印任何内容。
        - num_train_samples: 用于检查训练准确率的训练样本数量，默认为 1000；
          设置为 None 则使用整个训练集。
        - num_val_samples: 用于检查验证准确率的验证样本数量，默认为 None，
          使用整个验证集。
        - checkpoint_name: 如果不为 None，则每 epoch 在此保存模型检查点。
        """
        self.model = model
        self.X_train = data["X_train"]
        self.y_train = data["y_train"]
        self.X_val = data["X_val"]
        self.y_val = data["y_val"]

        # 解包关键字参数
        self.update_rule = kwargs.pop("update_rule", "sgd")
        self.optim_config = kwargs.pop("optim_config", {})
        self.lr_decay = kwargs.pop("lr_decay", 1.0)
        self.batch_size = kwargs.pop("batch_size", 100)
        self.num_epochs = kwargs.pop("num_epochs", 10)
        self.num_train_samples = kwargs.pop("num_train_samples", 1000)
        self.num_val_samples = kwargs.pop("num_val_samples", None)

        self.checkpoint_name = kwargs.pop("checkpoint_name", None)
        self.print_every = kwargs.pop("print_every", 10)
        self.verbose = kwargs.pop("verbose", True)

        # 如果还有额外的关键字参数，则抛出错误
        if len(kwargs) > 0:
            extra = ", ".join('"%s"' % k for k in list(kwargs.keys()))
            raise ValueError("未识别的参数 %s" % extra)

        # 确保更新规则存在，然后将字符串名称替换为实际函数
        if not hasattr(optim, self.update_rule):
            raise ValueError('无效的更新规则 "%s"' % self.update_rule)
        self.update_rule = getattr(optim, self.update_rule)

        self._reset()

    def _reset(self):
        """
        设置一些用于优化的记录变量。不要手动调用。
        """
        # 设置一些记录变量
        self.epoch = 0
        self.best_val_acc = 0
        self.best_params = {}
        self.loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []

        # 为每个参数复制 optim_config
        self.optim_configs = {}
        for p in self.model.params:
            d = {k: v for k, v in self.optim_config.items()}
            self.optim_configs[p] = d

    def _step(self):
        """
        执行一次梯度更新。此方法由 train() 调用，不应手动调用。
        """
        # 从训练数据中获取一个小批量
        num_train = self.X_train.shape[0]
        batch_mask = np.random.choice(num_train, self.batch_size)
        X_batch = self.X_train[batch_mask]
        y_batch = self.y_train[batch_mask]

        # 计算损失和梯度
        loss, grads = self.model.loss(X_batch, y_batch)
        self.loss_history.append(loss)

        # 执行参数更新
        for p, w in self.model.params.items():
            dw = grads[p]
            config = self.optim_configs[p]
            next_w, next_config = self.update_rule(w, dw, config)
            self.model.params[p] = next_w
            self.optim_configs[p] = next_config

    def _save_checkpoint(self):
        if self.checkpoint_name is None:
            return
        checkpoint = {
            "model": self.model,
            "update_rule": self.update_rule,
            "lr_decay": self.lr_decay,
            "optim_config": self.optim_config,
            "batch_size": self.batch_size,
            "num_train_samples": self.num_train_samples,
            "num_val_samples": self.num_val_samples,
            "epoch": self.epoch,
            "loss_history": self.loss_history,
            "train_acc_history": self.train_acc_history,
            "val_acc_history": self.val_acc_history,
        }
        filename = "%s_epoch_%d.pkl" % (self.checkpoint_name, self.epoch)
        if self.verbose:
            print('正在保存检查点到 "%s"' % filename)
        with open(filename, "wb") as f:
            pickle.dump(checkpoint, f)

    def check_accuracy(self, X, y, num_samples=None, batch_size=100):
        """
        检查模型在提供的数据上的准确率。

        参数：
        - X: 数据数组，形状为 (N, d_1, ..., d_k)
        - y: 标签数组，形状为 (N,)
        - num_samples: 如果不为 None，则从数据中随机抽取 num_samples 个样本进行测试。
        - batch_size: 将 X 和 y 分成大小为 batch_size 的批次，以避免使用过多内存。

        返回：
        - acc: 标量，表示模型正确分类的实例比例。
        """

        # 可能会从数据中随机抽取样本
        N = X.shape[0]
        if num_samples is not None and N > num_samples:
            mask = np.random.choice(N, num_samples)
            N = num_samples
            X = X[mask]
            y = y[mask]

        # 分批计算预测结果
        num_batches = N // batch_size
        if N % batch_size != 0:
            num_batches += 1
        y_pred = []
        for i in range(num_batches):
            start = i * batch_size
            end = (i + 1) * batch_size
            scores = self.model.loss(X[start:end])
            y_pred.append(np.argmax(scores, axis=1))
        y_pred = np.hstack(y_pred)
        acc = np.mean(y_pred == y)

        return acc

    def train(self):
        """
        运行优化过程以训练模型。
        """
        num_train = self.X_train.shape[0]
        iterations_per_epoch = max(num_train // self.batch_size, 1)
        num_iterations = self.num_epochs * iterations_per_epoch

        for t in range(num_iterations):
            self._step()

            # 可能会打印训练损失
            if self.verbose and t % self.print_every == 0:
                print(
                    "(Iteration %d / %d) loss: %f"
                    % (t + 1, num_iterations, self.loss_history[-1])
                )

            # 每个 epoch 结束时，增加 epoch 计数器并衰减学习率
            epoch_end = (t + 1) % iterations_per_epoch == 0
            if epoch_end:
                self.epoch += 1
                for k in self.optim_configs:
                    self.optim_configs[k]["learning_rate"] *= self.lr_decay

            # 在第一次迭代、最后一次迭代以及每个 epoch 结束时，
            # 检查训练集和验证集上的准确率
            first_it = t == 0
            last_it = t == num_iterations - 1
            if first_it or last_it or epoch_end:
                train_acc = self.check_accuracy(
                    self.X_train, self.y_train, num_samples=self.num_train_samples
                )
                val_acc = self.check_accuracy(
                    self.X_val, self.y_val, num_samples=self.num_val_samples
                )
                self.train_acc_history.append(train_acc)
                self.val_acc_history.append(val_acc)
                self._save_checkpoint()

                if self.verbose:
                    print(
                        "(Epoch %d / %d) train acc: %f; val_acc: %f"
                        % (self.epoch, self.num_epochs, train_acc, val_acc)
                    )

                # 记录最佳模型
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.best_params = {}
                    for k, v in self.model.params.items():
                        self.best_params[k] = v.copy()

        # 训练结束后，将最佳参数替换到模型中
        self.model.params = self.best_params
