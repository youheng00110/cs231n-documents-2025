"""
数值梯度计算工具模块
"""
from __future__ import print_function
from builtins import range
from past.builtins import xrange

import numpy as np
from random import randrange


def eval_numerical_gradient(f, x, verbose=True, h=0.00001):
    """
    使用中心差分法计算函数 f 在点 x 处的数值梯度。

    参数
    ----
    f : callable
        单变量函数。
    x : ndarray
        梯度计算点。
    verbose : bool
        是否打印每步计算的维度索引和梯度值。
    h : float
        中心差分步长。

    返回
    ----
    grad : ndarray, shape 与 x 相同
        f 在 x 处的数值梯度。
    """
    fx = f(x)
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        ix = it.multi_index
        oldval = x[ix]
        x[ix] = oldval + h
        fxph = f(x)
        x[ix] = oldval - h
        fxmh = f(x)
        x[ix] = oldval
        grad[ix] = (fxph - fxmh) / (2 * h)
        if verbose:
            print(ix, grad[ix])
        it.iternext()
    return grad


def eval_numerical_gradient_array(f, x, df, h=1e-5):
    """
    计算接收和返回 NumPy 数组的函数 f 在 x 处的数值梯度。

    参数
    ----
    f : callable
        返回数组的函数。
    x : ndarray
        梯度计算点。
    df : ndarray
        输出的梯度，与 f(x) 形状一致。
    h : float
        中心差分步长。

    返回
    ----
    grad : ndarray, shape 与 x 相同
        f 在 x 处的数值梯度。
    """
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        ix = it.multi_index
        oldval = x[ix]
        x[ix] = oldval + h
        pos = f(x).copy()
        x[ix] = oldval - h
        neg = f(x).copy()
        x[ix] = oldval
        grad[ix] = np.sum((pos - neg) * df) / (2 * h)
        it.iternext()
    return grad


def eval_numerical_gradient_blobs(f, inputs, output, h=1e-5):
    """
    计算作用于输入和输出“blob”（通常用于神经网络计算）的函数的数值梯度。

    参数
    ----
    f : callable
        函数，接收输入 blob 和输出 blob。
    inputs : tuple of input blobs
        输入数据。
    output : output blob
        输出数据。
    h : float
        中心差分步长。

    返回
    ----
    numeric_diffs : list of ndarray
        每个输入 blob 对应的数值梯度。
    """
    numeric_diffs = []
    for input_blob in inputs:
        diff = np.zeros_like(input_blob.diffs)
        it = np.nditer(
            input_blob.vals, flags=["multi_index"], op_flags=["readwrite"]
        )
        while not it.finished:
            idx = it.multi_index
            orig = input_blob.vals[idx]
            input_blob.vals[idx] = orig + h
            f(*(inputs + (output,)))
            pos = np.copy(output.vals)
            input_blob.vals[idx] = orig - h
            f(*(inputs + (output,)))
            neg = np.copy(output.vals)
            input_blob.vals[idx] = orig
            diff[idx] = np.sum((pos - neg) * output.diffs) / (2.0 * h)
            it.iternext()
        numeric_diffs.append(diff)
    return numeric_diffs


def eval_numerical_gradient_net(net, inputs, output, h=1e-5):
    """
    为神经网络计算数值梯度。
    """
    return eval_numerical_gradient_blobs(
        lambda *args: net.forward(), inputs, output, h=h
    )


def grad_check_sparse(f, x, analytic_grad, num_checks=10, h=1e-5):
    """
    使用中心差分法随机检查函数 f 在 x 处的数值梯度，与解析梯度对比。

    参数
    ----
    f : callable
        函数，接收单个输入。
    x : ndarray
        梯度计算点。
    analytic_grad : ndarray
        解析梯度。
    num_checks : int
        检查的随机维度数量。
    h : float
        中心差分步长。
    """
    for i in range(num_checks):
        ix = tuple([randrange(m) for m in x.shape])
        oldval = x[ix]
        x[ix] = oldval + h
        fxph = f(x)
        x[ix] = oldval - h
        fxmh = f(x)
        x[ix] = oldval
        grad_numerical = (fxph - fxmh) / (2 * h)
        grad_analytic = analytic_grad[ix]
        rel_error = abs(grad_numerical - grad_analytic) / (
            abs(grad_numerical) + abs(grad_analytic)
        )
        print(
            "数值梯度: %f, 解析梯度: %f, 相对误差: %e"
            % (grad_numerical, grad_analytic, rel_error)
        )
