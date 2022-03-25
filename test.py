import torch


def test(net, test_iter, criterion, device):
    total, correct = 0, 0

    net.eval()# 测试模式

    with torch.no_grad():
        print("*************** test ***************")
        for X, y in test_iter:
            X, y = X.to(device), y.to(device)# CPU or GPU运行

            output = net(X)  # 计算输出
            loss = criterion(output, y)  # 计算损失

            total += y.size(0)  # 计算测试集总样本数
            correct += (output.argmax(dim=1) == y).sum().item()  # 计算验证集预测准确的样本数

        test_acc = 100.0 * correct / total  # 测试集准确率

        # 输出测试集的损失
    print("test_loss: {:.3f} | test_acc: {:6.3f}%" \
          .format(loss.item(), test_acc))
    print("************************************\n")


    return loss.item(),test_acc