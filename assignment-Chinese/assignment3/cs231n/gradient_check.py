import numpy as np
from random import randrange


def eval_numerical_gradient(f, x, verbose=True, h=0.00001):
    """
    计算函数f在点x处的数值梯度的简单实现
    - f是一个接受单个参数的函数
    - x是要计算梯度的点（numpy数组）
    """

    fx = f(x)  # 计算原始点的函数值
    grad = np.zeros_like(x)  # 初始化梯度数组，形状与x相同
    # 迭代x中的所有索引
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:

        # 计算x+h处的函数值
        ix = it.multi_index  # 多维度索引
        oldval = x[ix]  # 保存原始值
        x[ix] = oldval + h  # 增加h
        fxph = f(x)  # 计算f(x + h)
        x[ix] = oldval - h  # 减少h
        fxmh = f(x)  # 计算f(x - h)
        x[ix] = oldval  # 恢复原始值

        # 使用中心差分公式计算偏导数
        grad[ix] = (fxph - fxmh) / (2 * h)  # 斜率
        if verbose:
            print(ix, grad[ix])
        it.iternext()  # 移动到下一个维度

    return grad


def eval_numerical_gradient_array(f, x, df, h=1e-5):
    """
    为接受numpy数组并返回numpy数组的函数计算数值梯度
    """
    grad = np.zeros_like(x)  # 初始化梯度数组
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        ix = it.multi_index  # 多维度索引

        oldval = x[ix]
        x[ix] = oldval + h
        pos = f(x).copy()  # 计算x+h处的函数值
        x[ix] = oldval - h
        neg = f(x).copy()  # 计算x-h处的函数值
        x[ix] = oldval  # 恢复原始值

        # 计算梯度：(f(x+h) - f(x-h))与df的点积除以2h
        grad[ix] = np.sum((pos - neg) * df) / (2 * h)
        it.iternext()
    return grad


def eval_numerical_gradient_blobs(f, inputs, output, h=1e-5):
    """
    为操作输入和输出blob的函数计算数值梯度

    假设f接受多个输入blob作为参数，最后是一个用于写入输出的blob。
    例如，f可能被这样调用：
    f(x, w, out)
    其中x和w是输入Blob，f的结果将被写入out。

    输入：
    - f: 函数
    - inputs: 输入blob的元组
    - output: 输出blob
    - h: 步长
    """
    numeric_diffs = []
    for input_blob in inputs:
        diff = np.zeros_like(input_blob.diffs)  # 初始化梯度差异数组
        # 迭代输入blob的所有元素
        it = np.nditer(input_blob.vals, flags=["multi_index"], op_flags=["readwrite"])
        while not it.finished:
            idx = it.multi_index  # 多维度索引
            orig = input_blob.vals[idx]  # 保存原始值

            input_blob.vals[idx] = orig + h
            f(*(inputs + (output,)))  # 计算f在x+h处的值
            pos = np.copy(output.vals)
            input_blob.vals[idx] = orig - h
            f(*(inputs + (output,)))  # 计算f在x-h处的值
            neg = np.copy(output.vals)
            input_blob.vals[idx] = orig  # 恢复原始值

            # 计算梯度差异
            diff[idx] = np.sum((pos - neg) * output.diffs) / (2.0 * h)

            it.iternext()
        numeric_diffs.append(diff)
    return numeric_diffs


def eval_numerical_gradient_net(net, inputs, output, h=1e-5):
    """计算神经网络的数值梯度"""
    return eval_numerical_gradient_blobs(
        lambda *args: net.forward(), inputs, output, h=h
    )


def grad_check_sparse(f, x, analytic_grad, num_checks=10, h=1e-5):
    """
    随机采样一些元素，只返回这些维度上的数值梯度
    用于梯度检查，验证解析梯度的正确性
    """

    for i in range(num_checks):
        # 随机生成一个索引
        ix = tuple([randrange(m) for m in x.shape])

        oldval = x[ix]
        x[ix] = oldval + h  # 增加h
        fxph = f(x)  # 计算f(x + h)
        x[ix] = oldval - h  # 减少h
        fxmh = f(x)  # 计算f(x - h)
        x[ix] = oldval  # 恢复原始值

        # 计算数值梯度和解析梯度
        grad_numerical = (fxph - fxmh) / (2 * h)
        grad_analytic = analytic_grad[ix]
        # 计算相对误差
        rel_error = abs(grad_numerical - grad_analytic) / (
            abs(grad_numerical) + abs(grad_analytic)
        )
        print(
            "数值梯度: %f 解析梯度: %f, 相对误差: %e"
            % (grad_numerical, grad_analytic, rel_error)
        )
