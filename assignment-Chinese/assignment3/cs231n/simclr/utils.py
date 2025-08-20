import pandas as pd
import torch
import torch.optim as optim
from thop import profile, clever_format  # 用于计算模型 FLOPs 和参数
from torch.utils.data import DataLoader
from tqdm import tqdm  # 用于显示进度条
from .contrastive_loss import *  # 导入对比损失函数

def train(model, data_loader, train_optimizer, epoch, epochs, batch_size=32, temperature=0.5, device='cuda'):
    """用一个 epoch 训练在 ./model.py 中定义的模型。
    
    输入：
    - model: 模型类对象，如 ./model.py 中定义的。
    - data_loader: torch.utils.data.DataLoader 对象；加载训练数据。可以假设加载的数据已经过增强。
    - train_optimizer: torch.optim.Optimizer 对象；为训练应用优化器。
    - epoch: 整数；当前 epoch 编号。
    - epochs: 整数；总 epoch 数量。
    - batch_size: 每个批次的训练样本数。
    - temperature: 浮点数；simclr_loss_vectorized 中使用的温度（tau）参数。
    - device: 定义 torch 张量的设备名称。

    返回：
    - 平均损失。
    """
    model.train()  # 设置模型为训练模式
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)  # 初始化总损失、总样本数和进度条
    for data_pair in train_bar:
        x_i, x_j, target = data_pair  # 获取一对增强图像和目标标签
        x_i, x_j = x_i.to(device), x_j.to(device)  # 将数据移至指定设备
        
        out_left, out_right, loss = None, None, None
        ##############################################################################
        # 你的代码开始处                                                              #
        #                                                                            #
        # 查看 model.py 文件以了解模型的输入和输出。                                   #
        # 将 x_i 和 x_j 通过模型以获得 out_left, out_right。                          #
        # 然后使用 simclr_loss_vectorized 计算损失。                                  #
        ##############################################################################
        
        
        ##############################################################################
        # 你的代码结束处                                                             #
        ##############################################################################
        
        train_optimizer.zero_grad()  # 清零梯度
        loss.backward()  # 反向传播计算梯度
        train_optimizer.step()  # 更新参数

        total_num += batch_size  # 累加总样本数
        total_loss += loss.item() * batch_size  # 累加总损失
        # 更新进度条描述
        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))

    return total_loss / total_num  # 返回平均损失


def train_val(model, data_loader, train_optimizer, epoch, epochs, device='cuda'):
    """训练或验证模型（根据是否提供优化器）"""
    is_train = train_optimizer is not None  # 判断是训练还是验证模式
    model.train() if is_train else model.eval()  # 设置模型模式
    loss_criterion = torch.nn.CrossEntropyLoss()  # 交叉熵损失函数

    # 初始化统计变量：总损失、top1正确率、top5正确率、总样本数和进度条
    total_loss, total_correct_1, total_correct_5, total_num, data_bar = 0.0, 0.0, 0.0, 0, tqdm(data_loader)
    with (torch.enable_grad() if is_train else torch.no_grad()):  # 训练时启用梯度，验证时禁用
        for data, target in data_bar:
            data, target = data.to(device), target.to(device)  # 将数据移至指定设备
            out = model(data)  # 模型输出
            loss = loss_criterion(out, target)  # 计算损失

            if is_train:  # 训练模式下更新参数
                train_optimizer.zero_grad()
                loss.backward()
                train_optimizer.step()

            total_num += data.size(0)  # 累加总样本数
            total_loss += loss.item() * data.size(0)  # 累加总损失
            # 按预测概率排序
            prediction = torch.argsort(out, dim=-1, descending=True)
            # 计算top1正确率
            total_correct_1 += torch.sum((prediction[:, 0:1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            # 计算top5正确率
            total_correct_5 += torch.sum((prediction[:, 0:5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()

            # 更新进度条描述
            data_bar.set_description('{} Epoch: [{}/{}] Loss: {:.4f} ACC@1: {:.2f}% ACC@5: {:.2f}%'
                                     .format('Train' if is_train else 'Test', epoch, epochs, total_loss / total_num,
                                             total_correct_1 / total_num * 100, total_correct_5 / total_num * 100))

    # 返回平均损失、top1正确率和top5正确率
    return total_loss / total_num, total_correct_1 / total_num * 100, total_correct_5 / total_num * 100


def test(model, memory_data_loader, test_data_loader, epoch, epochs, c, temperature=0.5, k=200, device='cuda'):
    """使用加权 KNN 测试模型性能"""
    model.eval()  # 设置模型为评估模式
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []  # 初始化统计变量和特征库
    with torch.no_grad():  # 禁用梯度计算
        # 生成特征库
        for data, _, target in tqdm(memory_data_loader, desc='Feature extracting'):
            feature, out = model(data.to(device))  # 提取特征
            feature_bank.append(feature)  # 将特征添加到特征库
        # [D, N] 转置特征库以便后续计算相似度
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N] 获取特征库中所有样本的标签
        feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=feature_bank.device)
        # 遍历测试数据，通过加权 KNN 搜索预测标签
        test_bar = tqdm(test_data_loader)
        for data, _, target in test_bar:
            data, target = data.to(device), target.to(device)  # 将数据移至指定设备
            feature, out = model(data)  # 提取测试样本特征

            total_num += data.size(0)  # 累加总测试样本数
            # 计算每个特征向量与特征库之间的余弦相似度 ---> [B, N]
            sim_matrix = torch.mm(feature, feature_bank)
            
            # [B, K] 获取每个测试样本的 top K 相似特征
            sim_weight, sim_indices = sim_matrix.topk(k=k, dim=-1)
            # [B, K] 获取这些相似特征对应的标签
            sim_labels = torch.gather(feature_labels.expand(data.size(0), -1), dim=-1, index=sim_indices)
            # 计算相似度权重的指数（除以温度参数）
            sim_weight = (sim_weight / temperature).exp()

            # 为每个类别计数
            one_hot_label = torch.zeros(data.size(0) * k, c, device=device)
            # [B*K, C] 将相似标签转换为 one-hot 编码
            one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
            # 加权得分 ---> [B, C]
            pred_scores = torch.sum(one_hot_label.view(data.size(0), -1, c) * sim_weight.unsqueeze(dim=-1), dim=1)

            # 按预测得分排序
            pred_labels = pred_scores.argsort(dim=-1, descending=True)
            # 计算 top1 正确率
            total_top1 += torch.sum((pred_labels[:, :1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            # 计算 top5 正确率
            total_top5 += torch.sum((pred_labels[:, :5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            # 更新测试进度条描述
            test_bar.set_description('Test Epoch: [{}/{}] Acc@1:{:.2f}% Acc@5:{:.2f}%'
                                     .format(epoch, epochs, total_top1 / total_num * 100, total_top5 / total_num * 100))

    # 返回 top1 和 top5 正确率
    return total_top1 / total_num * 100, total_top5 / total_num * 100
