from PIL import Image
from torchvision import transforms
from torchvision.datasets import CIFAR10
import random
import torch

def compute_train_transform(seed=123456):
    """
    该函数返回对单张训练图像的数据增强组合。
    完成以下代码行。提示：查看torchvision.transforms中的可用函数
    """
    random.seed(seed)
    torch.random.manual_seed(seed)
    
    # 应用颜色抖动的变换，亮度=0.4，对比度=0.4，饱和度=0.4，色调=0.1
    color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  
    
    train_transform = transforms.Compose([
        ##############################################################################
        # 你的代码开始处                                                             #
        #                                                                            #
        # 提示：查看torchvision.transforms中定义的变换函数                            #
        # 第一个操作已作为示例为你完成。
        ##############################################################################
        # 步骤1：随机调整大小并裁剪为32x32
        transforms.RandomResizedCrop(32),
        # 步骤2：以0.5的概率水平翻转图像

        # 步骤3：以0.8的概率应用颜色抖动（可以使用上面定义的"color_jitter"）

        # 步骤4：以0.2的概率将图像转换为灰度图

        ##############################################################################
        # 你的代码结束处                                                             #
        ##############################################################################
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])
    return train_transform
    
def compute_test_transform():
    """计算测试集的变换"""
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])
    return test_transform


class CIFAR10Pair(CIFAR10):
    """CIFAR10数据集的扩展，用于生成图像对。
    """
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        x_i = None
        x_j = None

        if self.transform is not None:
            ##############################################################################
            # 你的代码开始处                                                             #
            #                                                                            #
            # 对图像应用self.transform以生成论文中的x_i和x_j                              #
            ##############################################################################
            pass
            ##############################################################################
            # 你的代码结束处                                                             #
            ##############################################################################

        if self.target_transform is not None:
            target = self.target_transform(target)

        return x_i, x_j, target