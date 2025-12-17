from builtins import range
import numpy as np


def affine_forward(x, w, b):
    """计算仿射（全连接）层的前向传播。

    输入x的形状为(N, d_1, ..., d_k)，包含N个样本的小批量，
    其中每个样本x[i]的形状为(d_1, ..., d_k)。我们会将每个输入重塑为维度为D = d_1 * ... * d_k的向量，
    然后将其转换为维度为M的输出向量。

    输入：
    - x: 包含输入数据的numpy数组，形状为(N, d_1, ..., d_k)
    - w: 权重的numpy数组，形状为(D, M)
    - b: 偏置的numpy数组，形状为(M,)

    返回：
    - out: 输出，形状为(N, M)
    - cache: (x, w, b)，用于反向传播的缓存数据
    """
    # 将输入展平成二维矩阵 (N, D)
    N = x.shape[0]
    x_row = x.reshape(N, -1)
    out = x_row.dot(w) + b

    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """计算仿射（全连接）层的反向传播。

    输入：
    - dout: 上游导数，形状为(N, M)
    - cache: 元组，包含：
      - x: 输入数据，形状为(N, d_1, ... d_k)
      - w: 权重，形状为(D, M)
      - b: 偏置，形状为(M,)

    返回：
    - dx: 相对于x的梯度，形状为(N, d1, ..., d_k)
    - dw: 相对于w的梯度，形状为(D, M)
    - db: 相对于b的梯度，形状为(M,)
    """
    x, w, b = cache
    N = x.shape[0]
    x_row = x.reshape(N, -1)

    # dout 形状为 (N, M)
    dw = x_row.T.dot(dout)
    db = np.sum(dout, axis=0)
    dx_row = dout.dot(w.T)
    dx = dx_row.reshape(x.shape)

    return dx, dw, db


def relu_forward(x):
    """计算整流线性单元（ReLU）层的前向传播。

    输入：
    - x: 任意形状的输入

    返回：
    - out: 输出，与x形状相同
    - cache: x，用于反向传播的缓存数据
    """
    out = np.maximum(0, x)
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """计算整流线性单元（ReLU）层的反向传播。

    输入：
    - dout: 上游导数，任意形状
    - cache: 输入x，与dout形状相同

    返回：
    - dx: 相对于x的梯度
    """
    x = cache
    dx = dout.copy()
    dx[x <= 0] = 0
    return dx


def softmax_loss(x, y):
    """计算softmax分类的损失和梯度。

    输入：
    - x: 输入数据，形状为(N, C)，其中x[i, j]是第i个输入在第j类上的得分
    - y: 标签向量，形状为(N,)，其中y[i]是x[i]的标签，且0 <= y[i] < C

    返回：
    - loss: 标量损失值
    - dx: 相对于x的损失梯度
    """
    # 数值稳定处理
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)

    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N

    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N

    return loss, dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """批归一化的前向传播。

    在训练期间，样本均值和（未校正的）样本方差从迷你批量统计中计算，并用于归一化输入数据。
    在训练期间，我们还保持每个特征的均值和方差的指数衰减运行平均值，这些平均值用于测试时的归一化。

    在每个时间步，我们使用基于动量参数的指数衰减来更新均值和方差的运行平均值：

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    注意，批归一化论文建议不同的测试时行为：他们使用大量训练图像计算每个特征的样本均值和方差，
    而不是使用运行平均值。在本实现中，我们选择使用运行平均值，因为它们不需要额外的估计步骤；
    torch7的批归一化实现也使用运行平均值。

    输入：
    - x: 形状为(N, D)的数据
    - gamma: 形状为(D,)的缩放参数
    - beta: 形状为(D,)的偏移参数
    - bn_param: 包含以下键的字典：
      - mode: 'train'或'test'；必需
      - eps: 数值稳定性的常数
      - momentum: 运行均值/方差的常数
      - running_mean: 形状为(D,)的特征运行均值数组
      - running_var: 形状为(D,)的特征运行方差数组

    返回：
    - out: 形状为(N, D)的输出
    - cache: 反向传播所需的中间值元组
    """
    mode = bn_param["mode"]
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)

    N, D = x.shape
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == "train":
        ########################################################################
        # 实现批归一化的训练时前向传播。                                         #
        # 使用迷你批量统计计算均值和方差，使用这些统计量归一化输入数据，           #
        # 并使用gamma和beta对归一化数据进行缩放和偏移。                          #
        #                                                                      #
        # 应将输出存储在变量out中。反向传播所需的任何中间值应存储在cache变量中。   #
        #                                                                      #
        # 还应使用计算出的样本均值和方差以及动量变量来更新运行均值和运行方差，     #
        # 将结果存储在running_mean和running_var变量中。                         #
        #                                                                      #
        # 注意，尽管需要跟踪运行方差，但应基于标准差（方差的平方根）归一化数据！   #
        # 参考原始论文（https://arxiv.org/abs/1502.03167）可能会有帮助。         #
        ########################################################################
        x_mean=np.mean(x,axis=0)
        x_var=np.var(x,axis=0)
        x_norm=(x-x_mean)/np.sqrt(x_var+eps)
        out=gamma*x_norm+beta
        # 存储中间变量以供反向传播使用
        running_mean=momentum*running_mean+(1-momentum)*x_mean
        running_var=momentum*running_var+(1-momentum)*x_var
        cache=(x,x_norm,x_mean,x_var,gamma,beta,eps)
        ########################################################################
        #                           你的代码结束                                #
        ########################################################################
    elif mode == "test":
        ################################################################################
        # 实现批归一化的测试时前向传播。                                                 #
        # 使用运行均值和方差归一化输入数据，然后使用gamma和beta对归一化数据进行缩放和偏移。#
        # 将结果存储在out变量中。                                                      #
        ##############################################################################
        x_norm=(x-running_mean)/np.sqrt(running_var+eps)
        out=gamma*x_norm+beta
        #######################################################################
        #                          你的代码结束                                 #
        #######################################################################
    else:
        raise ValueError('无效的批归一化前向模式 "%s"' % mode)

    # 将更新后的运行均值存储回bn_param
    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """批归一化的反向传播。

    对于本实现，应在纸上画出批归一化的计算图，并通过中间节点反向传播梯度。

    输入：
    - dout: 上游导数，形状为(N, D)
    - cache: 来自batchnorm_forward的中间值变量。

    返回：
    - dx: 相对于输入x的梯度，形状为(N, D)
    - dgamma: 相对于缩放参数gamma的梯度，形状为(D,)
    - dbeta: 相对于偏移参数beta的梯度，形状为(D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # 实现批归一化的反向传播。将结果存储在dx、dgamma和dbeta变量中。              #
    # 这里给出的是“分步展开”的推导过程，变量多一些、结构更清晰。              #
    ###########################################################################
    # 从缓存中取出前向传播的中间结果
    x, x_norm, x_mean, x_var, gamma, beta, eps = cache

    N, D = x.shape

    # 1) 对应 out = gamma * x_norm + beta
    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(dout * x_norm, axis=0)
    dx_norm = dout * gamma

    # 2) 对应 x_norm = (x - mean) / sqrt(var + eps)
    x_mu = x - x_mean                        # (N, D)
    sqrt_var = np.sqrt(x_var + eps)          # (D,)
    inv_sqrt_var = 1.0 / sqrt_var            # (D,)

    # dx_norm = x_mu * inv_sqrt_var
    # 分两条链：对 x_mu 和 对 inv_sqrt_var
    dx_mu1 = dx_norm * inv_sqrt_var          # 来自直接除以 sqrt_var 的那条路

    # inv_sqrt_var = (x_var + eps)^(-1/2)
    # d(inv_sqrt_var)/dvar = -1/2 * (x_var + eps)^(-3/2)
    dvar = np.sum(dx_norm * x_mu * (-0.5) * (x_var + eps) ** (-1.5), axis=0)

    # x_var = 1/N * sum(x_mu^2)
    # dvar/dx_mu = 2/N * x_mu
    dx_mu2 = (2.0 / N) * x_mu * dvar         # 通过方差这条路回传到 x_mu

    # 汇总来自两条路径对 x_mu 的梯度
    dx_mu = dx_mu1 + dx_mu2                  # (N, D)

    # x_mu = x - mean
    # 对 x: 1， 对 mean: -1
    dmean = -np.sum(dx_mu, axis=0)           # 来自 x_mu 对 mean 的贡献

    # mean = 1/N * sum(x)
    # dmean/dx = 1/N
    dx1 = dx_mu                              # 来自 x_mu 对 x 的部分
    dx2 = dmean / N                          # 来自 mean 对 x 的部分

    dx = dx1 + dx2                           # (N, D)

    ###########################################################################
    #                             你的代码结束                                 #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """批归一化的替代反向传播。

    对于本实现，应在纸上计算批归一化反向传播的导数并尽可能简化。应能推导出反向传播的简单表达式。
    更多提示参见jupyter笔记本。

    注意：本实现应期望接收与batchnorm_backward相同的cache变量，但可能不会使用cache中的所有值。

    输入/输出：与batchnorm_backward相同
    """
    dx, dgamma, dbeta = None, None, None
    ############################################################################
    # 实现批归一化的反向传播。将结果存储在dx、dgamma和dbeta变量中。               #
    # 这里给出的是等价但更加紧凑的向量化公式。                                   #
    ###########################################################################
    x, x_norm, x_mean, x_var, gamma, beta, eps = cache

    N, D = x.shape

    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(dout * x_norm, axis=0)
    dx_norm = dout * gamma

    x_mu = x - x_mean
    std_inv = 1.0 / np.sqrt(x_var + eps)

    dx = (
        1.0 / N
        * std_inv
        * (
            N * dx_norm
            - np.sum(dx_norm, axis=0)
            - x_mu * (std_inv ** 2) * np.sum(dx_norm * x_mu, axis=0)
        )
    )

    ###########################################################################
    #                             你的代码结束                                 #
    ###########################################################################

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """层归一化的前向传播。

    在训练和测试时，输入数据都按每个数据点进行归一化，然后使用与批归一化相同的gamma和beta参数进行缩放和偏移。

    注意，与批归一化不同，层归一化在训练和测试时的行为是相同的，不需要跟踪任何运行平均值。

    输入：
    - x: 形状为(N, D)的数据
    - gamma: 形状为(D,)的缩放参数
    - beta: 形状为(D,)的偏移参数
    - ln_param: 包含以下键的字典：
        - eps: 数值稳定性的常数

    返回：
    - out: 形状为(N, D)的输出
    - cache: 反向传播所需的中间值元组
    """
    out, cache = None, None
    eps = ln_param.get("eps", 1e-5)
    ##############################################################################
    # 实现层归一化的训练时前向传播。                                               #
    # 归一化输入数据，并使用gamma和beta对归一化数据进行缩放和偏移。                 #
    # 提示：这可以通过稍微修改批归一化的训练时实现，并插入一两行精心设计的代码来完成。#
    # 特别是，能否想到任何矩阵变换，可以使你复制批归一化代码并几乎不做修改？         #
    #############################################################################
    x_mean=np.mean(x,axis=1,keepdims=True)
    x_var=np.var(x,axis=1,keepdims=True)
    x_norm=(x-x_mean)/np.sqrt(x_var+eps)
    out=gamma*x_norm+beta
    cache=(x,x_norm,x_mean,x_var,gamma,beta,eps)
    ###########################################################################
    #                             你的代码结束                                 #
    ###########################################################################
    return out, cache


def layernorm_backward(dout, cache):
    """层归一化的反向传播。

    对于本实现，可以在很大程度上依赖已经为批归一化所做的工作。

    输入：
    - dout: 上游导数，形状为(N, D)
    - cache: 来自layernorm_forward的中间值变量。

    返回：
    - dx: 相对于输入x的梯度，形状为(N, D)
    - dgamma: 相对于缩放参数gamma的梯度，形状为(D,)
    - dbeta: 相对于偏移参数beta的梯度，形状为(D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # 实现层归一化的反向传播。                                                  #
    #                                                                         #
    # 提示：这可以通过稍微修改批归一化的训练时实现来完成。前向传播的提示仍然适用！ #
    ###########################################################################
    x, x_norm, x_mean, x_var, gamma, beta, eps = cache
    N, D = x.shape
    dbeta=np.sum(dout,axis=0)
    dgamma=np.sum(dout*x_norm,axis=0)
    dx_norm=dout*gamma
    x_mu=x - x_mean
    std_inv=1.0/np.sqrt(x_var+eps)
    dx = (1.0 / D) * std_inv * (D * dx_norm - np.sum(dx_norm, axis=1, keepdims=True) - x_mu * (std_inv ** 2) * np.sum(dx_norm * x_mu, axis=1, keepdims=True))

    ###########################################################################
    #                             你的代码结束                                 #
    ###########################################################################
    return dx, dgamma, dbeta

def dropout_forward(x, dropout_param):
    """倒置丢弃法（inverted dropout）的前向传播。

    注意这与标准丢弃法不同。这里，p是保留神经元输出的概率，而非丢弃神经元输出的概率。
    更多细节参见http://cs231n.github.io/neural-networks-2/#reg。

    输入：
    - x: 任意形状的输入数据
    - dropout_param: 包含以下键的字典：
      - p: 丢弃参数。我们以概率p保留每个神经元的输出。
      - mode: 'test'或'train'。若为训练模式，则执行丢弃；若为测试模式，则直接返回输入。
      - seed: 随机数生成器的种子。传入种子可使函数具有确定性，这在梯度检查中需要，但在实际网络中不需要。

    输出：
    - out: 与x形状相同的数组。
    - cache: 元组(dropout_param, mask)。训练模式下，mask是用于与输入相乘的丢弃掩码；测试模式下，mask为None。
    """
    p, mode = dropout_param["p"], dropout_param["mode"]
    if "seed" in dropout_param:
        np.random.seed(dropout_param["seed"])  # 设置随机种子以保证确定性

    mask = None
    out = None

    if mode == "train":
        #######################################################################
        # 实现训练阶段的倒置丢弃法前向传播。将丢弃掩码存储在mask变量中。         #
        #######################################################################
        mask = (np.random.rand(*x.shape) < p) / p
        out = x * mask
        #######################################################################
        #                           你的代码结束                               #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # 实现测试阶段的倒置丢弃法前向传播。                                    #
        #######################################################################
        out = x
        #######################################################################
        #                            你的代码结束                              #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)  # 确保输出数据类型与输入一致

    return out, cache


def dropout_backward(dout, cache):
    """倒置丢弃法的反向传播。

    输入：
    - dout: 上游导数，任意形状
    - cache: 来自dropout_forward的(dropout_param, mask)
    """
    dropout_param, mask = cache
    mode = dropout_param["mode"]

    dx = None
    if mode == "train":
        #######################################################################
        # 实现训练阶段的倒置丢弃法反向传播。                                    #
        #######################################################################
        dx = dout * mask
        #######################################################################
        #                          你的代码结束                                 #
        #######################################################################
    elif mode == "test":
        dx = dout  # 测试模式下，梯度直接传递
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """卷积层前向传播的朴素实现。

    输入包含N个数据点，每个数据点有C个通道、高度H和宽度W。我们使用F个不同的滤波器对每个输入进行卷积，
    每个滤波器覆盖所有C个通道，高度为HH，宽度为WW。

    输入：
    - x: 输入数据，形状为(N, C, H, W)
    - w: 滤波器权重，形状为(F, C, HH, WW)
    - b: 偏置，形状为(F,)
    - conv_param: 包含以下键的字典：
      - 'stride': 水平和垂直方向上相邻感受野之间的像素数（步长）。
      - 'pad': 用于对输入进行零填充的像素数。

    填充时，应在输入的高度和宽度轴上对称地放置'pad'个零（即两侧各放pad个）。注意不要直接修改原始输入x。

    返回：
    - out: 输出数据，形状为(N, F, H', W')，其中H'和W'由下式计算：
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # 实现卷积前向传播。提示：可以使用np.pad函数进行填充。                       #
    ###########################################################################
    H_out=1+(x.shape[2]+2*conv_param['pad']-w.shape[2])//conv_param['stride']
    W_out=1+(x.shape[3]+2*conv_param['pad']-w.shape[3])//conv_param['stride']
    out=np.zeros((x.shape[0],w.shape[0],H_out,W_out))
    x_pad=np.pad(x,((0,0),(0,0),(conv_param['pad'],conv_param['pad']),(conv_param['pad'],conv_param['pad'])),mode="constant")
    for n in range(x.shape[0]):
        for i in range(H_out):
            for j in range(W_out):
                for f in range(w.shape[0]):
                    out[n,f,i,j]=x_pad[n,:,i*conv_param['stride']:i*conv_param['stride']+w.shape[2],j*conv_param['stride']:j*conv_param['stride']+w.shape[3]].flatten().dot(w[f].flatten())+b[f]
    ###########################################################################
    #                             你的代码结束                                 #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """卷积层反向传播的朴素实现。

    输入：
    - dout: 上游导数。
    - cache: 来自conv_forward_naive的(x, w, b, conv_param)元组

    返回：
    - dx: 相对于x的梯度
    - dw: 相对于w的梯度
    - db: 相对于b的梯度
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # 实现卷积反向传播。                                                       #
    ###########################################################################
    x, w, b, conv_param = cache
    db=np.sum(dout,axis=(0,2,3))
    dw=np.zeros_like(w)
    dx_pad=np.zeros((x.shape[0],x.shape[1],x.shape[2]+2*conv_param['pad'],x.shape[3]+2*conv_param['pad']))
    x_pad=np.pad(x,((0,0),(0,0),(conv_param['pad'],conv_param['pad']),(conv_param['pad'],conv_param['pad'])),mode="constant")
    for n in range(x.shape[0]):
        for i in range(dout.shape[2]):
            for j in range(dout.shape[3]):
                for f in range(w.shape[0]):
                    dw[f]+=dout[n,f,i,j]*x_pad[n,:,i*conv_param['stride']:i*conv_param['stride']+w.shape[2],j*conv_param['stride']:j*conv_param['stride']+w.shape[3]]
                    dx_pad[n,:,i*conv_param['stride']:i*conv_param['stride']+w.shape[2],j*conv_param['stride']:j*conv_param['stride']+w.shape[3]]+=dout[n,f,i,j]*w[f]
    dx=dx_pad[:,:,conv_param['pad']:dx_pad.shape[2]-conv_param['pad'],conv_param['pad']:dx_pad.shape[3]-conv_param['pad']]
    ###########################################################################
    #                             你的代码结束                                 #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """最大池化层前向传播的朴素实现。

    输入：
    - x: 输入数据，形状为(N, C, H, W)
    - pool_param: 包含以下键的字典：
      - 'pool_height': 每个池化区域的高度
      - 'pool_width': 每个池化区域的宽度
      - 'stride': 相邻池化区域之间的距离

    这里不需要填充，例如可假设：
      - (H - pool_height) % stride == 0
      - (W - pool_width) % stride == 0

    返回：
    - out: 输出数据，形状为(N, C, H', W')，其中H'和W'由下式计算：
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # 实现最大池化前向传播。                                                   #
    ###########################################################################
    N,C,H,W=x.shape
    out=np.zeros((x.shape[0],x.shape[1],  1 + (H - pool_param['pool_height']) // pool_param['stride'],1 + (W - pool_param['pool_width']) // pool_param['stride']))
    for n in range(out.shape[0]):
        for c in range(out.shape[1]):
            for i in range(out.shape[2]):
                for j in range(out.shape[3]):
                    out[n,c,i,j]=np.max(x[n,c,
                                          i*pool_param['stride']:i*pool_param['stride']+pool_param['pool_height'],
                                          j*pool_param['stride']:j*pool_param['stride']+pool_param['pool_width']])
    ###########################################################################
    #                             你的代码结束                                 #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """最大池化层反向传播的朴素实现。

    输入：
    - dout: 上游导数
    - cache: 来自前向传播的(x, pool_param)元组

    返回：
    - dx: 相对于x的梯度
    """
    dx = None
    ###########################################################################
    # 实现最大池化反向传播。                                                   #
    ###########################################################################
    x, pool_param = cache
    dx=np.zeros_like(x)
    out=np.zeros_like(dout)
    for n in range(dout.shape[0]):
        for c in range(dout.shape[1]):
            for i in range(dout.shape[2]):
                for j in range(dout.shape[3]):
                    window=x[n,c,i*pool_param['stride']:i*pool_param['stride']+pool_param['pool_height'],j*pool_param['stride']:j*pool_param['stride']+pool_param['pool_width']]
                    mask=(window==np.max(window))
                    dx[n,c,
                       i*pool_param['stride']:i*pool_param['stride']+pool_param['pool_height'],
                       j*pool_param['stride']:j*pool_param['stride']+pool_param['pool_width']]+=dout[n,c,i,j]*mask
    ###########################################################################
    #                             你的代码结束                                 #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """空间批归一化的前向传播。

    输入：
    - x: 输入数据，形状为(N, C, H, W)
    - gamma: 缩放参数，形状为(C,)
    - beta: 偏移参数，形状为(C,)
    - bn_param: 包含以下键的字典：
      - mode: 'train'或'test'；必需
      - eps: 数值稳定性常数
      - momentum: 运行均值/方差的常数。momentum=0表示每次完全丢弃旧信息，
        而momentum=1表示从不纳入新信息。默认momentum=0.9在大多数情况下适用。
      - running_mean: 形状为(D,)的特征运行均值数组
      - running_var: 形状为(D,)的特征运行方差数组

    返回：
    - out: 输出数据，形状为(N, C, H, W)
    - cache: 反向传播所需的值
    """
    out, cache = None, None

    ###########################################################################
    # 实现空间批归一化的前向传播。                                              #
    #                                                                         #
    # 提示：可通过调用上面实现的标准批归一化函数来实现空间批归一化。              #
    # 实现应该非常简短；我们的实现不到5行。                                     #
    ###########################################################################
    # 
    ###########################################################################
    #                             你的代码结束                                 #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """空间批归一化的反向传播。

    输入：
    - dout: 上游导数，形状为(N, C, H, W)
    - cache: 前向传播中的值

    返回：
    - dx: 相对于输入的梯度，形状为(N, C, H, W)
    - dgamma: 相对于缩放参数的梯度，形状为(C,)
    - dbeta: 相对于偏移参数的梯度，形状为(C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # 实现空间批归一化的反向传播。                                              #
    #                                                                         #
    # 提示：可通过调用上面实现的标准批归一化函数来实现空间批归一化。              #
    # 实现应该非常简短；我们的实现不到5行。                                     #
    ###########################################################################
    # 
    ###########################################################################
    #                             你的代码结束                                 #
    ###########################################################################

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """空间组归一化的前向传播。
    
    与层归一化不同，组归一化将数据中的每个样本分成G个连续的部分，然后独立地对每个部分进行归一化。
    然后对数据应用逐特征的偏移和缩放，方式与批归一化和层归一化相同。

    输入：
    - x: 输入数据，形状为(N, C, H, W)
    - gamma: 缩放参数，形状为(1, C, 1, 1)
    - beta: 偏移参数，形状为(1, C, 1, 1)
    - G: 要划分的组数，必须是C的约数
    - gn_param: 包含以下键的字典：
      - eps: 数值稳定性常数

    返回：
    - out: 输出数据，形状为(N, C, H, W)
    - cache: 反向传播所需的值
    """
    out, cache = None, None
    eps = gn_param.get("eps", 1e-5)
    ###########################################################################
    # 实现空间组归一化的前向传播。                                              #
    # 这与层归一化的实现极其相似。                                              #
    # 具体来说，思考如何转换矩阵，使得大部分代码可复用训练时的批归一化和层归一化！ #
    ###########################################################################
    # 
    ###########################################################################
    #                             你的代码结束                                 #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """空间组归一化的反向传播。

    输入：
    - dout: 上游导数，形状为(N, C, H, W)
    - cache: 前向传播中的值

    返回：
    - dx: 相对于输入的梯度，形状为(N, C, H, W)
    - dgamma: 相对于缩放参数的梯度，形状为(1, C, 1, 1)
    - dbeta: 相对于偏移参数的梯度，形状为(1, C, 1, 1)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # 实现空间组归一化的反向传播。                                              #
    # 这与层归一化的实现极其相似。                                              #
    ###########################################################################
    # 
    ###########################################################################
    #                             你的代码结束                                 #
    ###########################################################################
    return dx, dgamma, dbeta
