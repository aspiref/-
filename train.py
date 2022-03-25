# 训练模型
import time

import torch

def train(net, train_iter, criterion, optimizer, num_epochs, device, num_print, lr_scheduler=None, val_iter=None):
    net.train()
    record_train = list()
    # 记录每一Epoch下训练集的准确率
    # List() 方法用于将元组转换为列表。
    record_val = list()

    for epoch in range(num_epochs):
        print("========== epoch: [{}/{}] ==========".format(epoch + 1, num_epochs))
        total, correct, train_loss = 0, 0, 0
        start = time.time()

        for i, (X, y) in enumerate(train_iter):
            # enumerate就是枚举的意思，把元素一个个列举出来，第一个是什么，第二个是什么，所以他返回的是元素以及对应的索引。
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            X, y = X.to(device), y.to(device)
            output = net(X)
            loss = criterion(output, y) # 计算LOSS

            optimizer.zero_grad()  # 梯度归零
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数

            train_loss += loss.item()
            total += y.size(0)
            correct += (output.argmax(dim=1) == y).sum().item() # 累积预测正确的样本数
            # output.argmax(dim=1) 返回指定维度最大值的序号，dim=1，把dim=1这个维度的，变成这个维度的最大值的index
            # 即 dim=0是取每一列最大值的下标，dim=1是取每一行最大值的下标
            train_acc = 100.0 * correct / total

            if (i + 1) % num_print == 0:
                print("step: [{}/{}], train_loss: {:.3f} | train_acc: {:6.3f}% | lr: {:.6f}" \
                    .format(i + 1, len(train_iter), train_loss / (i + 1), \
                            train_acc, get_cur_lr(optimizer)))


        if lr_scheduler is not None:
            # 调整梯度下降算法的学习率
            lr_scheduler.step()
        # 输出训练的时间
        print("--- cost time: {:.4f}s ---".format(time.time() - start))

        if val_iter is not None:  # 判断验证集是否为空 (注意这里将调用val函数)
            record_val.append(val(net, val_iter, criterion, device)) # 每训练一个Epoch模型，使用验证集进行验证模型的准确度
        record_train.append(train_acc)
        # append() 方法用于在列表末尾追加新的对象。
    # 返回每一个Epoch下测试集和训练集的准确率
    torch.save(net.state_dict(),"net1.pth")
    return record_train, record_val

# 验证模型
def val(net, val_iter, criterion, device):
    total, correct = 0, 0

    net.eval()# 验证模式

    with torch.no_grad():
        print("*************** val ***************")
        for X, y in val_iter:
            X, y = X.to(device), y.to(device)# CPU or GPU运行

            output = net(X)  # 计算输出
            val_loss = criterion(output, y)  # 计算损失
            total += y.size(0)  # 计算测试集总样本数
            correct += (output.argmax(dim=1) == y).sum().item()


            val_acc = 100.0 * correct / total  # 测试集准确率

           # 输出验证集的损失
    print("val_loss: {:.3f} | val_acc: {:6.3f}%" \
              .format(val_loss.item(),val_acc))
    print("************************************\n")

    # 训练模式 （因为这里是因为每经过一个Epoch就使用测试集一次，使用验证集后，进入下一个Epoch前将模型重新置于训练模式）
    net.train()

    return val_acc

# 返回学习率lr的函数
def get_cur_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']