import torch
import os, glob
import random, csv
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class MyDataset(Dataset):

    def __init__(self, root, mode):

        super(MyDataset, self).__init__()
        self.root = root
        # self.resize = resize

        self.name2label = {}
        # 返回指定目录下的文件列表，并对文件列表进行排序，
        # os.listdir每次返回目录下的文件列表顺序会不一致，
        # 排序是为了每次返回文件列表顺序一致，即排序后，文件夹顺序就固定了
        for name in sorted(os.listdir(os.path.join(root))):
            # 过滤掉非目录文件
            if not os.path.isdir(os.path.join(root, name)):
                continue
            # 构建字典，名字：0~4数字,字典是具有键值对的。就是有健key、值value，且一对一对存储的
            self.name2label[name] = len(self.name2label.keys())
            # name2label:{文件夹名，类别编号}

        # eg: {'squirtle': 4, 'bulbasaur': 0, 'pikachu': 3, 'mewtwo': 2, 'charmander': 1}
        print(self.name2label)

        # image, label  load——csv完成label读取
        self.images, self.labels = self.load_csv("images.csv")

        # 对数据集进行划分为train和test
        if mode == "train":
            self.images = self.images[:int(0.6 * len(self.images))]
            self.labels = self.labels[:int(0.6 * len(self.labels))]
        elif mode == "val":
            self.images = self.images[int(0.6 * len(self.images)):int(0.8 * len(self.images))]
            self.labels = self.labels[int(0.6 * len(self.labels)):int(0.8 * len(self.labels))]
        elif mode == "test":
            self.images = self.images[int(0.8 * len(self.images)):int(len(self.images))]
            self.labels = self.labels[int(0.8 * len(self.labels)):int(len(self.labels))]

        # else: # 20% = 80%~100%
        #     self.images = self.images[int(0.8 * len(self.images)):]
        #     self.labels = self.labels[int(0.8 * len(self.labels)):]

    # 将目录下的图片路径与其对应的标签写入csv文件，
    # 并将csv文件写入的内容读出，返回图片名与其标签
    def load_csv(self, filename):
        """
        :param filename:
        :return:
        """
        # 是否已经存在了cvs文件
        if not os.path.exists(os.path.join(self.root, filename)):
            images = [] # 将所有的信息组成一个列表，类别信息通过中间的一个路径判断
            for name in self.name2label.keys():
                # 获取指定目录下所有的满足后缀的图像名
                # mydataset/mewtwo/00001.png
                images += glob.glob(os.path.join(self.root, name, "*.png"))
            # 1165 'mydataset/pikachu/00000058.png'
            print(len(images), images)

            # 将元素打乱
            random.shuffle(images)
            with open(os.path.join(self.root, filename), mode="w", newline="") as f:
                writer = csv.writer(f)
                for img in images:  # 'mydataset/pikachu/00000058.png'
                    name = img.split(os.sep)[-2]
                    label = self.name2label[name]
                    # 将图片路径以及对应的标签写入到csv文件中
                    # 'mydataset/pikachu/00000058.png', 0
                    writer.writerow([img, label])
                print("writen into csv file: ", filename)

        # 如果已经存在了csv文件，则读取csv文件
        images, labels = [], []
        with open(os.path.join(self.root, filename)) as f:
            reader = csv.reader(f)
            for row in reader:
                #row ： 'mydataset/pikachu/00000058.png', 0
                img, label = row
                label = int(label)

                images.append(img)
                labels.append(label)
        assert len(images) == len(labels)

        return images, labels

    def __len__(self):
        return len(self.images)


    def __getitem__(self, idx):
        # idx~[0~len(images)]
        # self.images, self.labels
        # img: 'mydataset/bulbasaur/00000000.png'
        # label: 0

        img, label = self.images[idx], self.labels[idx]

        tf = transforms.Compose([
            transforms.RandomCrop(112, padding=4),  # 先四周填充0，在吧图像随机裁剪成112*112
            transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
         ])

        img = tf(Image.open(img))
        label = torch.tensor(label)

        return img, label




