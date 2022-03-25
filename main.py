
import torch

import torch.nn as nn
import creatdataset

from accurary import learning_curve
from resnext import Resnext
from test import test
from train import train


def load_dataset(batch_size):


    root = r"C:\Users\Jia\PycharmProjects\pythonProject\resnet_dataset"
    train_set = creatdataset.MyDataset(root, mode="train")
    train_iter = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    val_set = creatdataset.MyDataset(root, mode="val")
    val_iter = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)

    test_set = creatdataset.MyDataset(root, mode="test")
    test_iter = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_iter, val_iter, test_iter


import torch.optim as optim
BATCH_SIZE = 128  # 批大小
NUM_EPOCHS = 12  # Epoch大小
NUM_CLASSES = 3  # 分类的个数
LEARNING_RATE = 0.01  # 梯度下降学习率
MOMENTUM = 0.9  # 冲量大小
WEIGHT_DECAY = 0.0005  # 权重衰减系数
NUM_PRINT = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # GPU or CPU运行


def main():
    net = Resnext(NUM_CLASSES)
    net = net.to(DEVICE)

    train_iter, val_iter, test_iter = load_dataset(BATCH_SIZE)  # 导入训练集和测试集

    criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数损失计算器
    # 优化器
    optimizer = optim.SGD(
        net.parameters(),
        # 构建好神经网络后，网络的参数都保存在parameters()函数当中
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
        nesterov=True
        # Nesterov动量梯度下降

    )
    # 调整学习率 (step_size:每训练step_size个epoch，更新一次参数; gamma:更新lr的乘法因子)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    record_train, record_val = train(net, train_iter, criterion, optimizer, \
          NUM_EPOCHS, DEVICE, NUM_PRINT, lr_scheduler, val_iter)

    learning_curve(record_train, record_val) # 画出准确率曲线


    if test_iter is not None:  # 判断验证集是否为空 (注意这里将调用test函数)
            test(net, test_iter, criterion, DEVICE)



main()

