import torch
from tqdm.auto import tqdm  # 导入tqdm库，用于显示进度条


def train_val(model, data_loader, train_optimizer, epoch, epochs, device='cpu'):
    """
    训练或验证模型的函数
    
    参数:
    - model: 要训练或验证的模型
    - data_loader: 数据加载器
    - train_optimizer: 训练优化器，如果为None则表示验证模式
    - epoch: 当前轮次
    - epochs: 总轮次
    - device: 运行设备（cpu或gpu）
    """
    is_train = train_optimizer is not None  # 判断是否为训练模式
    model.train() if is_train else model.eval()  # 设置模型为训练或评估模式
    loss_criterion = torch.nn.CrossEntropyLoss()  # 定义交叉熵损失函数

    # 初始化统计变量
    total_loss, total_correct_1, total_correct_5, total_num = 0.0, 0.0, 0.0, 0
    data_bar = tqdm(data_loader)  # 创建进度条

    # 根据模式决定是否启用梯度计算
    with (torch.enable_grad() if is_train else torch.no_grad()):
        for data, target in data_bar:
            # 将数据和标签移动到指定设备
            data, target = data.to(device), target.to(device)
            out = model(data)  # 模型前向传播，得到预测输出
            loss = loss_criterion(out, target)  # 计算损失

            if is_train:  # 如果是训练模式，执行反向传播和参数更新
                train_optimizer.zero_grad()  # 清空梯度
                loss.backward()  # 反向传播计算梯度
                train_optimizer.step()  # 更新参数

            # 累计统计信息
            total_num += data.size(0)  # 累计样本数量
            total_loss += loss.item() * data.size(0)  # 累计损失
            
            # 对预测结果按概率降序排序
            prediction = torch.argsort(out, dim=-1, descending=True)
            # 计算Top-1准确率（预测的第一个结果是否正确）
            total_correct_1 += torch.sum((prediction[:, 0:1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            # 计算Top-5准确率（注释掉，如需使用可取消注释）
            # total_correct_5 += torch.sum((prediction[:, 0:5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()

            # 更新进度条显示信息
            data_bar.set_description('{} Epoch: [{}/{}] Loss: {:.4f} ACC@1: {:.3f}%'
                                     .format('Train' if is_train else 'Test', epoch, epochs, 
                                             total_loss / total_num, total_correct_1 / total_num * 100))

    # 返回平均损失、Top-1准确率和Top-5准确率
    return total_loss / total_num, total_correct_1 / total_num, total_correct_5 / total_num


class ClassificationSolverViT:
    """视觉Transformer的分类求解器类，用于模型的训练和评估"""
    
    def __init__(self, train_data, test_data, model, **kwargs):
        self.model = model  # 要训练的模型
        self.train_data = train_data  # 训练数据集
        self.test_data = test_data  # 测试数据集

        # 解析关键字参数
        self.learning_rate = kwargs.pop("learning_rate", 1.e-4)  # 学习率，默认1e-4
        self.weight_decay = kwargs.pop("weight_decay", 0.0)  # 权重衰减系数，默认0
        self.batch_size = kwargs.pop("batch_size", 64)  # 批大小，默认64
        self.num_epochs = kwargs.pop("num_epochs", 2)  # 训练轮数，默认2

        # 初始化Adam优化器
        self.optimizer = torch.optim.Adam(self.model.parameters(), 
                                         self.learning_rate, 
                                         weight_decay=self.weight_decay)
        self.loss_criterion = torch.nn.CrossEntropyLoss()  # 交叉熵损失函数

        self._reset()  # 初始化记录变量

    def _reset(self):
        """初始化训练过程中的记录变量"""
        self.epoch = 0  # 当前轮次
        # 存储训练和测试的损失及准确率
        self.results = {'train_loss': [], 'train_acc@1': [], 
                       'test_loss': [], 'test_acc@1': []}

    def train(self, device='cpu'):
        """训练模型的主函数"""
        # 创建训练和测试数据加载器
        train_loader = torch.utils.data.DataLoader(self.train_data, 
                                                  batch_size=self.batch_size, 
                                                  shuffle=True)
        test_loader = torch.utils.data.DataLoader(self.test_data, 
                                                 batch_size=self.batch_size, 
                                                 shuffle=False)

        # 将模型和损失函数移动到指定设备
        self.model.to(device)
        self.loss_criterion.to(device)

        best_acc = 0.0  # 记录最佳测试准确率
        for epoch in range(self.num_epochs):
            # 训练模型
            train_loss, train_acc_1, _ = train_val(self.model, train_loader, 
                                                  self.optimizer, epoch, self.num_epochs, device)
            # 记录训练结果
            self.results['train_loss'].append(train_loss)
            self.results['train_acc@1'].append(train_acc_1)

            # 验证模型
            test_loss, test_acc_1, _ = train_val(self.model, test_loader, 
                                                None, epoch, self.num_epochs, device)
            # 记录测试结果
            self.results['test_loss'].append(test_loss)
            self.results['test_acc@1'].append(test_acc_1)
            
            # 更新最佳准确率
            if test_acc_1 > best_acc:
                best_acc = test_acc_1
        
        # 保存最佳测试准确率
        self.results["best_test_acc"] = best_acc

