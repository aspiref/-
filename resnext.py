import torch
import torch.nn as nn
class Block(nn.Module):
    def __init__(self,in_channels, out_channels, stride=1, is_shortcut=False):
        super(Block,self).__init__()
        # 继承block的父类进行初始化。而且是用父类的初始化方法来初始化继承的属性
        self.relu = nn.ReLU(inplace=True)
        # inplace-选择是否进行覆盖运算，即是否将得到的值计算得到的值覆盖之前的值
        self.is_shortcut = is_shortcut
        # 判断是否需要进行虚线部分
        self.conv1 = nn.Sequential(
            # Sequential一个有序的容器，神经网络模块将按照在传入构造器的顺序依次被添加到计算图中执行，
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=1,stride=stride,bias=False),
            nn.BatchNorm2d(out_channels // 4),
            # 添加BatchNorm2d进行数据的归一化处理，这使得数据在进行Relu之前不会因为数据过大而导致网络性能的不稳定
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels // 4, out_channels // 4, kernel_size=3, stride=1, padding=1, groups=32,
                                   bias=False),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(out_channels // 4, out_channels, kernel_size=1,stride=1,bias=False),
            nn.BatchNorm2d(out_channels),
        )
        if is_shortcut:
            self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=stride,bias=1),
            nn.BatchNorm2d(out_channels)
        )
    def forward(self, x):
        x_shortcut = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        if self.is_shortcut:
            x_shortcut = self.shortcut(x_shortcut)
        x = x + x_shortcut
        x = self.relu(x)
        return x

class Resnext(nn.Module):
    def __init__(self,num_classes,layer=[4,4,6,3]):
        super(Resnext,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.conv2 = self._make_layer(64,256,1,num=layer[0])
        self.conv3 = self._make_layer(256,512,2,num=layer[1])
        self.conv4 = self._make_layer(512,1024,2,num=layer[2])
        self.conv5 = self._make_layer(1024,2048,2,num=layer[3])
        self.global_average_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048,num_classes)
        # in_features指的是输入的二维张量的大小，即输入的[batch_size, size]中的size。
        # out_features指的是输出的二维张量的大小，即输出的二维张量的形状为[batch_size，output_size]，当然，它也代表了该全连接层的神经元个数。
        # 从输入输出的张量的shape角度来理解，相当于一个输入为[batch_size, in_features]的张量变换成了[batch_size, out_features]的输出张量。
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.global_average_pool(x)
        x = torch.flatten(x,1)
        # 1）flatten(x,1)是按照x的第1个维度拼接（按照列来拼接，横向拼接）；
        # 2）flatten(x,0)是按照x的第0个维度拼接（按照行来拼接，纵向拼接）；
        # 3）有时候会遇到flatten里面有两个维度参数，flatten(x, start_dim, end_dimension)，此时flatten函数执行的功能是将从start_dim到end_dim之间的所有维度值乘起来，
        # 其他的维度保持不变。例如x是一个size为[4,5,6]的tensor, flatten(x, 0, 1)的结果是一个size为[20,6]的tensor。
        x = self.fc(x)
        return x

    # 形成单个Stage的网络结构
    def _make_layer(self,in_channels,out_channels,stride,num):
        layers = []
        block_1=Block(in_channels, out_channels,stride=stride,is_shortcut=True)
        layers.append(block_1)
        # 该部分是将每个blocks的第一个残差结构保存在layers列表中。
        for i in range(1, num): # Identity Block结构叠加; 基于[3, 4, 6, 3]
            layers.append(Block(out_channels,out_channels,stride=1,is_shortcut=False))
            # 该部分是将每个blocks的剩下残差结构保存在layers列表中，这样就完成了一个blocks的构造。
        return nn.Sequential(*layers)
    # 返回Conv Block和Identity Block的集合，形成一个Stage的网络结构