import torch
import os
import numpy as np
import random
from torch.utils import data
from torchvision import transforms as trans
from sklearn.metrics import recall_score, precision_score, accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc as cal_auc
from PIL import Image
import sys
import logging

from torch import nn
from sklearn.metrics import confusion_matrix
from dataset.dataset import DeepfakeDataset
from trainer import Trainer


class FFDataset(data.Dataset):

    def __init__(self, dataset_root, frame_num=300, size=299, augment=True):
        self.data_root = dataset_root
        self.frame_num = frame_num
        self.train_list = self.collect_image(self.data_root)   #将文件夹中的每一个图像（绝对）路径加到train_list中
        if augment:
            self.transform = trans.Compose([
                                            trans.RandomHorizontalFlip(p=0.5),
                                            trans.ToTensor()])
            print("Augment True!")
        else:
            self.transform = trans.ToTensor()
        self.max_val = 1.
        self.min_val = -1.
        self.size = size

    def collect_image(self, root):
        image_path_list = []
        for split in os.listdir(root):
            split_root = os.path.join(root, split)
            img_list = os.listdir(split_root)
            random.shuffle(img_list)
            img_list = img_list if len(img_list) < self.frame_num else img_list[:self.frame_num]
            for img in img_list:
                img_path = os.path.join(split_root, img)
                image_path_list.append(img_path)
        return image_path_list

    def read_image(self, path):
        img = Image.open(path)
        return img

    def resize_image(self, image, size):
        img = image.resize((size, size))
        return img

    def __getitem__(self, index):
        image_path = self.train_list[index]
        img = self.read_image(image_path)
        img = self.resize_image(img,size=self.size)
        img = self.transform(img)
        img = img * (self.max_val - self.min_val) + self.min_val
        return img

    def __len__(self):
        return len(self.train_list)



def get_dataset(name = 'train', size=299, root='E:\\Dataset_pre\\ff++_dataset\\', frame_num=300, augment=True):
    root = os.path.join(root, name)
    fake_root = os.path.join(root,'fake')

    fake_list = ['Deepfakes', 'FaceSwap', 'Face2Face', 'NeuralTextures']

    total_len = len(fake_list)
    dset_lst = []
    for i in range(total_len):
        fake = os.path.join(fake_root , fake_list[i])
        dset = FFDataset(fake, frame_num, size, augment)
        dset.size = size
        dset_lst.append(dset)
    return torch.utils.data.ConcatDataset(dset_lst), total_len


def evaluate(model, normal_root,malicious_root,csv_root, mode='test',):

    my_dataset = DeepfakeDataset(normal_root=normal_root, malicious_root=malicious_root, mode=mode, resize=380,
                                 csv_root=csv_root)
    malicious_name = malicious_root.split('/')[-1]
    print("This is the {} {} dataset!".format(malicious_name,mode))
    print("dataset size:{}".format(len(my_dataset)))

    bz = 8
    # torch.cache.empty_cache()
    with torch.no_grad():
        y_true, y_pred = [], []

        dataloader = torch.utils.data.DataLoader(
            dataset=my_dataset,
            batch_size=bz,
            shuffle=True,
            num_workers=0
        )

        device = torch.device("cuda")
        correct = 0
        total = len(my_dataset)

        for i , (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            # print(x.shape)
            output = model(x)
            y_pred.extend(output.sigmoid().flatten().tolist())
            y_true.extend(y.flatten().tolist())

        print(" ")

        y_true, y_pred = np.array(y_true), np.array(y_pred)
        fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1)

        AUC = cal_auc(fpr, tpr)

        for i in range(len(y_pred)):
            if y_pred[i] < 0.5:
                y_pred[i] = 0
            else:
                y_pred[i] = 1

        r_acc = accuracy_score(y_true, y_pred)
        con_mat = confusion_matrix(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)

    return r_acc, AUC, con_mat, recall, precision







