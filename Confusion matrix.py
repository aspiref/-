import os
import json

import torch
from torchvision import transforms, datasets
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from prettytable import PrettyTable

import creatdataset
import resnext

class ConfusionMatrix(object):
    """
    注意，如果显示的图像不全，是matplotlib版本问题
    本例程使用matplotlib-3.2.1(windows and ubuntu)绘制正常
    需要额外安装prettytable库,能够将输出打印成列表的形式
    """
    def __init__(self, num_classes: int, labels: list):
        self.matrix = np.zeros((num_classes, num_classes))
        # 创造一个shape为num_classes*num_classes的正方形混淆矩阵，且初始化为0
        self.num_classes = num_classes
        self.labels = labels

    def update(self, preds, labels):
        # 将预测的值和输入标签输入进来
        for p, t in zip(preds, labels):
           # p代表预测值，t代表真实标签
            self.matrix[p, t] += 1

    def summary(self):
        # 精度/准确率
        sum_TP = 0
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]
            # 统计所有对角线元素的和
        acc = sum_TP / np.sum(self.matrix)
        print("the model accuracy is ", acc)

        # 精确率, 召回率（真阳性率）, 特异度 是对某个类别的，所以即使是多分类，在针对某个类别是也就可以认为成是他和不是他两种类别。而准确率是对整体的
        table = PrettyTable()
        # 使用PrettyTable库初始化一张表
        table.field_names = ["", "Precision", "Recall", "Specificity"]
        for i in range(self.num_classes):
            TP = self.matrix[i, i]
            # 对角线上的数值大小，即true positive 即真阳性，预测正确的个数
            FP = np.sum(self.matrix[i, :]) - TP
            # false positive 即假阳性，没有预测正确的个数，行代表的一个类别的预测值，所以每个类别的fp就是用这一行减去对角线上的数字即可
            FN = np.sum(self.matrix[:, i]) - TP
            # false negative 即假阴性，在真实值里没有预测正确的个数，就是不是他这个类别预测错误了。每一列代表的是每个分类的真实值，然后fn就是用一列减去对角线上的数字
            TN = np.sum(self.matrix) - TP - FP - FN
            # true negative 即其他类别预测正确的negative，即真阴性，就是不是他的类别预测正确了
            Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.
            # 后面这个3是保证了精度为3，即小数点后三位
            Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.
            # 召回率（真阳性率、灵敏度）：实际为阳性的样本中，判断有反应的比例
            Specificity = round(TN / (TN + FP), 3) if TN + FP != 0 else 0.
            # 特异度；实际上没有病而被正常诊断的概率，即实际上为阴性的样本中，判断无反应的比例
            # 具体可以看https://www.bilibili.com/video/BV1tf4y1R7x8?spm_id_from=333.788.top_right_bar_window_history.content.click
            table.add_row([self.labels[i], Precision, Recall, Specificity])
        print(table)

    def plot(self):
        matrix = self.matrix
        print(matrix)
        plt.imshow(matrix, cmap=plt.cm.Blues)
        # 颜色变换从白色到蓝色，注意大小写

        # 设置x轴坐标label，且旋转45度
        plt.xticks(range(self.num_classes), self.labels, rotation=45)
        # 设置y轴坐标label
        plt.yticks(range(self.num_classes), self.labels)
        # 显示colorbar,即色谱，知道数值的密集程度
        plt.colorbar()
        plt.xlabel('True Labels')
        plt.ylabel('Predicted Labels')
        plt.title('Confusion matrix')

        # 在图中标注数量/概率信息
        thresh = matrix.max() / 2
        #设置了一个阈值，是为了让超过阈值的数字颜色变白色，要不然看不见了
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                # 注意这里的matrix[y, x]不是matrix[x, y]，是因为坐标原点在左上角
                info = int(matrix[y, x])
                # 第y行第x列的元素
                plt.text(x, y, info,
                         verticalalignment='center',
                         # 绘制在水平中心
                         horizontalalignment='center',
                         # 绘制在竖直中心
                         color="white" if info > thresh else "black")
        plt.tight_layout()
        # 让图形显示的更加的紧凑
        plt.show()
        # 将混淆矩阵进行展示


if __name__ == '__main__':
    # 这个函数的作用是只有在这个函数中运行时下面这个代码才能运行，在其他脚本文件使用时，即使import这个函数也不行
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)



    root = r"C:\Users\Jia\PycharmProjects\pythonProject\resnet_dataset"
    test_set = creatdataset.MyDataset(root, mode="test")
    test_iter = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False, num_workers=0)

# 下面是进行迁移学习了，将自己的网络模型载入过来
    net = resnext.Resnext(num_classes=3)
    # load pretrain weights
    model_weight_path = r"C:\Users\Jia\PycharmProjects\pythonProject\net1.pth"
    assert os.path.exists(model_weight_path), "cannot find {} file".format(model_weight_path)
    net.load_state_dict(torch.load(model_weight_path, map_location=device))
    net.to(device)

    # 载入json文件
    try:
        json_file = open(r"C:\Users\Jia\PycharmProjects\pythonProject\class_indices.json",'r')
        class_indict = json.load(json_file)
    except Exception as e:
        print(e)
        exit(-1)

    labels = [label for _, label in class_indict.items()]
    # 因为json文件里是字典形式，咱这不需要其key值，只要其标签，所以for后面是_
    confusion = ConfusionMatrix(num_classes=3, labels=labels)
    net.eval()
    with torch.no_grad():
        for test_data in tqdm(test_iter):
            #  tqdm()是一个显示进度条的操作
            test_images, test_labels = test_data
            outputs = net(test_images.to(device))
            outputs = torch.softmax(outputs, dim=1)
            # 将结果以概率形式输出
            outputs = torch.argmax(outputs, dim=1)
            confusion.update(outputs.to("cpu").numpy(), test_labels.to("cpu").numpy())
            # outputs.to("cpu").numpy() 是预测值
    confusion.plot()
    confusion.summary()


