# 画出每一个Epoch下验证集和训练集的准确率
from matplotlib import pyplot as plt


def learning_curve(record_train, record_val=None):
    plt.style.use("ggplot")
    # 画图工具：https://blog.csdn.net/viviliving/article/details/107690844
    plt.plot(range(1, len(record_train) + 1), record_train, label="train acc")
    # plt.plot的用法 https://zhuanlan.zhihu.com/p/258106097
    if record_val is not None:
        plt.plot(range(1, len(record_val) + 1), record_val, label="val acc")

    plt.legend(loc=4)
    # loc:图例位置
    plt.title("learning curve")
    plt.xticks(range(0, len(record_train) + 1, 5))
    plt.yticks(range(0, 101, 5))
    plt.xlabel("epoch")
    plt.ylabel("accuracy")

    plt.show()
