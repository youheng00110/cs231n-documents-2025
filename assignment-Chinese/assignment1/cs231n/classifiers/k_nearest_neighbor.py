from builtins import range
from builtins import object
import numpy as np
from past.builtins import xrange


class KNearestNeighbor(object):
    """
    一个基于 L2 距离的 k 近邻分类器。
    """

    def __init__(self):
        """
        初始化分类器，无需执行任何操作。
        """
        pass

    def train(self, X, y):
        """
        训练分类器。对于 k 近邻分类器来说，这仅仅是记住训练数据。

        输入：
        - X: 一个形状为 (num_train, D) 的 NumPy 数组，包含 num_train 个训练样本，
             每个样本的维度为 D。
        - y: 一个形状为 (N,) 的 NumPy 数组，包含训练标签，其中 y[i] 是 X[i] 的标签。
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1, num_loops=0):
        """
        使用该分类器预测测试数据的标签。

        输入：
        - X: 一个形状为 (num_test, D) 的 NumPy 数组，包含 num_test 个测试样本，
             每个样本的维度为 D。
        - k: 用于投票预测标签的最近邻的数量。
        - num_loops: 决定使用哪种实现来计算训练点和测试点之间的距离。

        返回：
        - y: 一个形状为 (num_test,) 的 NumPy 数组，包含测试数据的预测标签，
             其中 y[i] 是测试点 X[i] 的预测标签。
        """
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        elif num_loops == 2:
            dists = self.compute_distances_two_loops(X)
        else:
            raise ValueError("Invalid value %d for num_loops" % num_loops)

        return self.predict_labels(dists, k=k)

    def compute_distances_two_loops(self, X):
        """
        使用两层嵌套循环计算每个测试点与每个训练点之间的距离。

        输入：
        - X: 一个形状为 (num_test, D) 的 NumPy 数组，包含测试数据。

        返回：
        - dists: 一个形状为 (num_test, num_train) 的 NumPy 数组，其中 dists[i, j]
          是第 i 个测试点与第 j 个训练点之间的欧几里得距离。
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            for j in range(num_train):
                ##############################################################################
                # 代办:                                                                      #
                # 计算第 i 个测试点与第 j 个训练点之间的 L2 距离，并将结果存储在 dists[i, j] 中。#
                # 不要使用维度循环，也不要使用 np.linalg.norm()。                              #
                ##############################################################################
                pass
        return dists

    def compute_distances_one_loop(self, X):
        """
        使用一层循环计算每个测试点与每个训练点之间的距离。

        输入 / 输出：与 compute_distances_two_loops 相同
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            ###########################################################################
            # 代办:                                                                   #
            # 计算第 i 个测试点与所有训练点之间的 L2 距离，并将结果存储在 dists[i, :] 中。#
            # 不要使用 np.linalg.norm()。                                              #
            ###########################################################################
            pass
        return dists

    def compute_distances_no_loops(self, X):
        """
        不使用显式循环计算每个测试点与每个训练点之间的距离。

        输入 / 输出：与 compute_distances_two_loops 相同
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        #################################################################################
        # 代办:                                                                         # 
        # 使用基本数组运算计算所有测试点与所有训练点之间的 L2 距离，并将结果存储在 dists 中。#
        # 不要使用 scipy 中的函数，也不要使用 np.linalg.norm()。                          #
        # 提示：尝试使用矩阵乘法和两个广播求和来表示 L2 距离。                             #
        #################################################################################
       
        return dists

    def predict_labels(self, dists, k=1):
        """
        给定一个测试点与训练点之间的距离矩阵，预测每个测试点的标签。

        输入：
        - dists: 一个形状为 (num_test, num_train) 的 NumPy 数组，其中 dists[i, j]
          是第 i 个测试点与第 j 个训练点之间的距离。

        返回：
        - y: 一个形状为 (num_test,) 的 NumPy 数组，包含测试数据的预测标签，
             其中 y[i] 是测试点 X[i] 的预测标签。
        """
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            # 一个长度为 k 的列表，存储第 i 个测试点的 k 个最近邻的标签。
            closest_y = []
            #########################################################################
            # 代办:                                                                 #
            # 使用距离矩阵找到第 i 个测试点的 k 个最近邻，并使用 self.y_train 找到这些 #
            # 邻点的标签。将这些标签存储在 closest_y 中。                             #
            # 提示：查看 numpy.argsort 函数。                                        #
            #########################################################################
            
            #########################################################################
            # 代办:                                                                 #
            # 现在你已经找到了 k 个最近邻的标签，你需要在 closest_y 列表中找到最常见的  #
            # 标签。将此标签存储在 y_pred[i] 中。如果出现平局，则选择较小的标签。       #
            #########################################################################
            
            
        return y_pred
