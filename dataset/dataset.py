import csv
import glob
import os
from random import random

import torch
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
from albumentations import CenterCrop,Compose,Resize,RandomCrop
from albumentations.pytorch.transforms import ToTensor

import cv2
from torchvision.transforms import transforms


class DeepfakeDataset(Dataset):

    def __init__(self,normal_root,malicious_root,mode,resize,csv_root):
        self.normal_root = normal_root          #有普通图片的总文件夹 F:\dataset\faceforensics++\manipulated_sequences\Deepfakes\c23\our_imgs
        self.malicious_root = malicious_root    #有被修改图片的总文件夹
        self.csv_root = csv_root  #存放一个csv文件，保存图片和标签
        self.resize = resize #修改输入图片大小尺寸
        # self.datalabel= 1 if '_' in str(self.normal_root).split('\\')[-1] else print("error the img has some problem") #加载正常图片
        # self.datalabel= 0 if '_' in str(self.malicious_root).split('\\')[-1] else print("error the img has some problem") #加载经过了deepfake的图片

        self.img2label = {}  #这是一个保存分类的字典
        self.img2label["malicious"] = 1
        self.img2label["normal"] = 0
        # {"malicious":1,"normal":0}

        # image, label
        self.images, self.labels = self.load_csv("images.csv")

        # 对数据集进行划分
        if mode == "train":  # 60%
            self.train_images = self.images[int(0.2 * len(self.images)):int(0.8 * len(self.images))]
            self.train_labels = self.labels[int(0.2 * len(self.labels)):int(0.8 * len(self.labels))]
            self.images = self.train_images
            self.labels = self.train_labels
        elif mode == "valid":  # 20% = 60%~80%
            self.val_images = self.images[int(0.1 * len(self.images)):int(0.2 * len(self.images))]
            self.val_images.extend(self.images[int(0.8 * len(self.images)):int(0.9 * len(self.images))])
            self.val_labels = self.labels[int(0.1 * len(self.labels)):int(0.2 * len(self.labels))]
            self.val_labels.extend(self.labels[int(0.8 * len(self.labels)):int(0.9 * len(self.labels))])

            self.images = self.val_images
            self.labels = self.val_labels
        else:  # 20% = 80%~100%
            self.test_images = self.images[:int(0.1 * len(self.images))]
            self.test_images.extend(self.images[int(0.9 * len(self.images)):])
            self.test_labels = self.labels[:int(0.1 * len(self.labels))]
            self.test_labels.extend(self.labels[int(0.9 * len(self.labels)):])

            self.images = self.test_images
            self.labels = self.test_labels



    def load_csv(self, filename):
        """
        主要是保存和加载图片和标签
        :param filename: 形如‘imag.csv'这样的文件
        :return:
        """
        # 是否已经存在了cvs文件
        if not os.path.exists(os.path.join(self.csv_root, filename)):
            # 这个是存储图片地址的列表
            normal_images = []
            malicious_images = []

            # 正常图片的路径载入
            for root, dirs, files in os.walk(os.path.join(self.normal_root)):
                for dir in dirs:
                    # 获取指定目录下所有的满足后缀的图像名
                    normal_images += glob.glob(os.path.join(os.path.join(root, dir), "*.png"))
            # 虚假图片的路径载入
            for root, dirs, files in os.walk(os.path.join(self.malicious_root)):
                for dir in dirs:
                    # 获取指定目录下所有的满足后缀的图像名
                    malicious_images += glob.glob(os.path.join(os.path.join(root, dir), "*.png"))

            with open(os.path.join(self.csv_root, filename), mode="w", newline="") as f:
                writer = csv.writer(f)
                #两种图片路径的载入
                for img in normal_images:
                    name = 'normal'
                    label = self.img2label[name]
                    # 将图片路径以及对应的标签写入到csv文件中
                    # 'F:\\dataset\\faceforensics++\\original_sequences\\youtube\\c23\\our_imgs\\000\\000-3.png', 0
                    writer.writerow([img, label])

                for img in malicious_images:
                    name = 'malicious'
                    label = self.img2label[name]
                    # 将图片路径以及对应的标签写入到csv文件中
                    writer.writerow([img, label])

                print("writen into csv file: ", filename)

        # 如果已经存在了csv文件，则读取csv文件
        images, labels = [], []
        with open(os.path.join(self.csv_root, filename)) as f:
            reader = csv.reader(f)
            for row in reader:
                # 'F:\\dataset\\faceforensics++\\original_sequences\\youtube\\c23\\our_imgs\\000\\000-3.png', 0
                img, label = row
                label = int(label)

                images.append(img)
                labels.append(label)


        assert len(images) == len(labels)

        return images, labels

    def __getitem__(self, idx):
        # idx~[0~len(images)]
        # self.images, self.labels
        # img: 'F:\\dataset\\faceforensics++\\original_sequences\\youtube\\c23\\our_imgs\\000\\000-3.png'
        # label: 0
        img, label = self.images[idx], self.labels[idx]

        tf = transforms.Compose([
            lambda x: Image.open(x).convert("RGB"),  # string path => image data
            # transforms.Resize((int(self.resize), int(self.resize))),
            transforms.Resize((int(self.resize * 1.25), int(self.resize * 1.25))),
            transforms.RandomRotation(30),
            transforms.CenterCrop(self.resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        img = tf(img)
        label = torch.tensor(label)

        return img, label

    def __len__(self):
        return len(self.images)

    def denormalize(self, x_hat):
        '''
        这个函数现在没什么用,是一个逆归一化
        '''
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        # x_hat = (x-mean)/std
        # x = x_hat*std = mean
        # x: [c, h, w]
        # mean: [3] => [3, 1, 1]
        mean = torch.tensor(mean).unsqueeze(1).unsqueeze(1)
        std = torch.tensor(std).unsqueeze(1).unsqueeze(1)
        # print(mean.shape, std.shape)
        x = x_hat * std + mean

        return x



