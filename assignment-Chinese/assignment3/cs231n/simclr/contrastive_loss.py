import torch
import numpy as np


def sim(z_i, z_j):
    """两个向量之间的归一化点积。

    输入：
    - z_i: 1xD 张量。
    - z_j: 1xD 张量。
    
    返回：
    - 一个标量值，表示 z_i 和 z_j 之间的归一化点积。
    """
    norm_dot_product = None
    ##############################################################################
    # 你的代码开始处                                                              #
    #                                                                            #
    # 提示：torch.linalg.norm 可能会有帮助。                                      #
    ##############################################################################
    # 归一化点积 = (z_i · z_j) / (||z_i|| * ||z_j||)
    norm_dot_product = (z_i @ z_j.T).squeeze() / (torch.linalg.norm(z_i) * torch.linalg.norm(z_j))
    
    ##############################################################################
    # 你的代码结束处                                                             #
    ##############################################################################
    
    return norm_dot_product


def simclr_loss_naive(out_left, out_right, tau):
    """计算一个批次上的对比损失 L（朴素循环版本）。
    
    输入：
    - out_left: NxD 张量；投影头 g() 的输出，SimCLR 模型的左分支。
    - out_right: NxD 张量；投影头 g() 的输出，SimCLR 模型的右分支。
    每行是批次中一个增强样本的 z 向量。out_left 和 out_right 中相同行构成一个正样本对。
    换句话说，对于所有 k=0...N-1，(out_left[k], out_right[k]) 构成一个正样本对。
    - tau: 标量值，温度参数，决定指数增长的速度。
    
    返回：
    - 一个标量值；批次中所有正样本对的总损失。定义见笔记本。
    """
    N = out_left.shape[0]  # 训练样本的总数
    
    # 将 out_left 和 out_right 拼接成一个 2*N x D 张量。
    out = torch.cat([out_left, out_right], dim=0)  # [2*N, D]
    
    total_loss = 0
    for k in range(N):  # 遍历每个正样本对 (k, k+N)
        z_k, z_k_N = out[k], out[k+N]
        
        ##############################################################################
        # 你的代码开始处                                                              #
        #                                                                            #
        # 提示：计算 l(k, k+N) 和 l(k+N, k)。                                         #
        ##############################################################################
        # 计算 l(k, k+N)
        numerator_1 =torch.exp(sim(z_k.unsqueeze(0), z_k_N.unsqueeze(0)) / tau)
        num_1 = 0
        for i in range(0,2*N):
            if i != k:
                num_1+=torch.exp(sim(z_k.unsqueeze(0),out[i].unsqueeze(0)) / tau)
        numerator_2=torch.exp(sim(z_k_N.unsqueeze(0), z_k.unsqueeze(0)) / tau)
        num_2 = 0
        for i in range(0,2*N):
            if i != k+N:
                num_2+=torch.exp(sim(z_k_N.unsqueeze(0),out[i].unsqueeze(0)) / tau)
        loss_1 = -torch.log(numerator_1 / num_1)
        loss_2 = -torch.log(numerator_2 / num_2)
        total_loss += loss_1 + loss_2

        ##############################################################################
        # 你的代码结束处                                                             #
        ##############################################################################
    
    # 最后，我们需要将总损失除以 2N，即批次中的样本数。
    total_loss = total_loss / (2*N)
    return total_loss


def sim_positive_pairs(out_left, out_right):
    """正样本对之间的归一化点积。

    输入：
    - out_left: NxD 张量；投影头 g() 的输出，SimCLR 模型的左分支。
    - out_right: NxD 张量；投影头 g() 的输出，SimCLR 模型的右分支。
    每行是批次中一个增强样本的 z 向量。
    out_left 和 out_right 中相同行构成一个正样本对。
    
    返回：
    - 一个 Nx1 张量；每行 k 是 out_left[k] 和 out_right[k] 之间的归一化点积。
    """
    pos_pairs = None
    
    ##############################################################################
    # 你的代码开始处                                                              #
    #                                                                            #
    # 提示：torch.linalg.norm 可能会有帮助。                                      #
    ##############################################################################
    # 方法1: 逐元素相乘后按行求和，再除以范数
    # 计算点积 (按行)
    dot_products = (out_left * out_right).sum(dim=1)  # [N]
    
    # 计算范数
    norm_left = torch.linalg.norm(out_left, dim=1)   # [N]
    norm_right = torch.linalg.norm(out_right, dim=1) # [N]
    
    # 归一化点积
    pos_pairs = dot_products / (norm_left * norm_right)  # [N]
    pos_pairs = pos_pairs.unsqueeze(1)  # [N, 1]
    
    ##############################################################################
    # 你的代码结束处                                                             #
    ##############################################################################
    return pos_pairs


def compute_sim_matrix(out):
    """计算批次中所有增强样本对之间归一化点积的 2N x 2N 矩阵。

    输入：
    - out: 2N x D 张量；每行是单个增强样本的 z 向量（投影头的输出）。
    批次中总共有 2N 个增强样本。
    
    返回：
    - sim_matrix: 2N x 2N 张量；矩阵中的每个元素 i, j 是 out[i] 和 out[j] 之间的归一化点积。
    """
    sim_matrix = None
    
    ##############################################################################
    # 你的代码开始处                                                             #
    ##############################################################################
    #矩阵乘法
    #normalize
    norms = torch.linalg.norm(out, dim=1, keepdim=True)  # 计算每个向量的范数，形状为 (2N, 1)
    normalized_out = out / norms  # 归一化每个向量，形状为 (2N, D)
    sim_matrix = normalized_out @ normalized_out.T  # 计算归一化点积矩阵，形状为 (2N, 2N)

    
    ##############################################################################
    # 你的代码结束处                                                             #
    ##############################################################################
    return sim_matrix


def simclr_loss_vectorized(out_left, out_right, tau, device='cuda'):
    """计算批次上的对比损失 L（向量化版本）。不允许使用循环。
    
    输入和输出与 simclr_loss_naive 相同。
    """
    N = out_left.shape[0]
    
    # 将 out_left 和 out_right 拼接成一个 2*N x D 张量。
    out = torch.cat([out_left, out_right], dim=0)  # [2*N, D]
    
    # 计算批次中所有增强样本对之间的相似度矩阵。
    sim_matrix = compute_sim_matrix(out)  # [2*N, 2*N]
    
    ##############################################################################
    # 你的代码开始处。按照提示进行。                                             #
    ##############################################################################
    
    # 步骤 1：使用 sim_matrix 计算所有增强样本的分母值。
    # 提示：计算 e^(sim / tau) 并存储到 exponential 中，其形状应为 2N x 2N。
    exponential = torch.exp(sim_matrix / tau)  # [2*N, 2*N]
    
    # 这个二进制掩码将 k=i 的项置零。
    mask = (torch.ones_like(exponential, device=device) - torch.eye(2 * N, device=device)).to(device).bool()
    
    # 应用二进制掩码。
    exponential = exponential.masked_select(mask).view(2 * N, -1)  # [2*N, 2*N-1]
    
    # 提示：计算所有增强样本的分母值。这应该是一个 2N x 1 向量。
    denom = exponential.sum(dim=1, keepdim=True)  # [2*N, 1]

    # 步骤 2：计算正样本对之间的相似度。
    # 你可以通过两种方式实现：
    # 选项 1：从 sim_matrix 中提取相应的索引。
    # 选项 2：使用 sim_positive_pairs()。
    
    
    # 步骤 3：计算所有增强样本的分子值。
    numerator = torch.exp(sim_positive_pairs(out_left, out_right) / tau)  # [N, 1]
    numerator = torch.cat([numerator, numerator], dim=0)  # [2*N, 1]
    
    
    # 步骤 4：现在你已经有了所有增强样本的分子和分母，计算总损失。
    loss = -torch.log(numerator / denom)  # [2*N, 1]
    loss = loss.mean()  # 计算平均损失，返回标量
    
    ##############################################################################
    # 你的代码结束处                                                             #
    ##############################################################################
    
    return loss


def rel_error(x,y):
    """计算相对误差，用于比较两个值的接近程度。"""
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))