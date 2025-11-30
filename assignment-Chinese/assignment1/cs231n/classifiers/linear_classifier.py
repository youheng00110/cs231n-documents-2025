from __future__ import print_function

import os
from builtins import range
from builtins import object
import numpy as np
from ..classifiers.softmax import *
from past.builtins import xrange


class LinearClassifier(object):
    """
    线性分类器基类。
    """
    def __init__(self):
        """
        初始化分类器，权重 W 默认为 None。
        """
        self.W = None

    def train(
        self,
        X,
        y,
        learning_rate=1e-3,
        reg=1e-5,
        num_iters=100,
        batch_size=200,
        verbose=False,
    ):
        """
        使用随机梯度下降（SGD）训练线性分类器。

        输入：
        - X: 一个形状为 (N, D) 的 NumPy 数组，包含 N 个训练样本，每个样本的维度为 D。
        - y: 一个形状为 (N,) 的 NumPy 数组，包含训练标签；y[i] = c 表示 X[i] 的标签为 0 <= c < C，其中 C 为类别数。
        - learning_rate: (float) 学习率。
        - reg: (float) 正则化强度。
        - num_iters: (整数) 优化时的迭代步数。
        - batch_size: (整数) 每次迭代使用的训练样本数量。
        - verbose: (布尔值) 如果为 True，则在优化过程中打印进度。

        输出：
        一个列表，包含每次训练迭代时的损失函数值。
        """
        num_train, dim = X.shape
        num_classes = (
            np.max(y) + 1
        )  # 假设 y 的取值范围为 0...K-1，其中 K 为类别数
        if self.W is None:
            # 懒初始化 W
            self.W = 0.001 * np.random.randn(dim, num_classes)

        # 使用随机梯度下降优化 W
        loss_history = []
        for it in range(num_iters):
            X_batch = None
            y_batch = None

            #########################################################################
            # 代办:                                                                 #
            # 从训练数据中随机采样 batch_size 个元素及其对应的标签，用于这一轮梯度下降。#
            # 将数据存储在 X_batch 中，对应的标签存储在 y_batch 中；采样完成后，X_batch#
            # 的形状应为 (batch_size, dim)，y_batch 的形状应为 (batch_size,)          #
            #                                                                       #
            # 提示：使用 np.random.choice 生成索引。有放回采样比无放回采样更快。        #
            #########################################################################
            index=np.random.choice(num_train,batch_size,replace=True)
            X_batch = X[index]
            y_batch = y[index]
            #########################################################################
            #                             你的代码结束                               #
            #########################################################################

            # 计算损失和梯度
            loss, grad = self.loss(X_batch, y_batch, reg)
            loss_history.append(loss)

            # 更新权重
            #########################################################################
            # 代办:                                                                 #
            # 使用梯度和学习率更新权重。                                              #
            #########################################################################
            self.W-=learning_rate*grad
            #########################################################################
            #                             你的代码结束                               #
            #########################################################################

            if verbose and it % 100 == 0:
                print("iteration %d / %d: loss %f" % (it, num_iters, loss))

        return loss_history

    def predict(self, X):
        """
        使用训练好的权重预测数据点的标签。

        输入：
        - X: 一个形状为 (N, D) 的 NumPy 数组，包含 N 个训练样本，每个样本的维度为 D。

        返回：
        - y_pred: 预测的标签，是一个长度为 N 的一维数组，每个元素是一个整数，表示预测的类别。
        """
        y_pred = np.zeros(X.shape[0])
        ###########################################################################
        # 代办:                                                                   #
        # 实现此方法。将预测的标签存储在 y_pred 中。                                #
        ###########################################################################
        y_pred=np.argmax(np.dot(X,self.W),axis=1)
        ###########################################################################
        #                             你的代码结束                                 #
        ###########################################################################
        return y_pred

    def loss(self, X_batch, y_batch, reg):
        """
        计算损失函数及其导数。
        子类将重写此方法。

        输入：
        - X_batch: 一个形状为 (N, D) 的 NumPy 数组，包含 N 个数据点的小批量，每个点的维度为 D。
        - y_batch: 一个形状为 (N,) 的 NumPy 数组，包含小批量的标签。
        - reg: (float) 正则化强度。

        返回：一个元组，包含：
        - 损失值，一个浮点数
        - 相对于 self.W 的梯度，与 W 形状相同的数组
        """
        pass

    def save(self, fname):
        """
        保存模型参数。
        """
        fpath = os.path.join(os.path.dirname(__file__), "../saved/", fname)
        params = {"W": self.W}
        np.save(fpath, params)
        print(fname, "已保存。")
    
    def load(self, fname):
        """
        加载模型参数。
        """
        fpath = os.path.join(os.path.dirname(__file__), "../saved/", fname)
        if not os.path.exists(fpath):
            print(fname, "不可用。")
            return False
        else:
            params = np.load(fpath, allow_pickle=True).item()
            self.W = params["W"]
            print(fname, "已加载。")
            return True


class LinearSVM(LinearClassifier):
    """
    使用多类 SVM 损失函数的子类。
    """
    def loss(self, X_batch, y_batch, reg):
        """
        计算 SVM 损失函数及其梯度。
        """
        return svm_loss_vectorized(self.W, X_batch, y_batch, reg)


class Softmax(LinearClassifier):
    """
    使用 Softmax + 交叉熵损失函数的子类。
    """
    def loss(self, X_batch, y_batch, reg):
        """
        计算 Softmax 损失函数及其梯度。
        """
        return softmax_loss_vectorized(self.W, X_batch, y_batch, reg)
