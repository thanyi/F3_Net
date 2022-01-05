'''
这个文件主要是针对在colab上调试运行的时候使用的，所以可能是导致不能进行在本地运行
'''


import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from models import *
from trainer import *
from xception import *
import numpy as np
from torchvision import transforms
from PIL import Image
import os
from dataset import DeepfakeDataset
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc as cal_auc



max_epoch = 5
loss_freq = 40
mode = 'FAD'  # ['Original', 'FAD', 'LFS', 'Both', 'Mix']
pretrained_path = 'xception-b5690688.pth'
device = torch.device("cuda")

normal_root = r"/content/data/normal_dlib"
malicious_root = r"/content/data/FaceSwap_dlib"
csv_root = r"/content/data/csv"

train_data = DeepfakeDataset(normal_root=normal_root, malicious_root=malicious_root, mode='train', resize=299,
                             csv_root=csv_root)
val_data = DeepfakeDataset(normal_root=normal_root, malicious_root=malicious_root, mode='val', resize=299,
                           csv_root=csv_root)

train_data_size = len(train_data)
val_data_size = len(val_data)

print('train_data_size:', train_data_size)
print('val_data_size:', val_data_size)

train_loader = DataLoader(train_data, 16, shuffle=True)
val_loader = DataLoader(val_data, 16, shuffle=True)

def evaluate(model, mode='valid'):
    my_dataset = DeepfakeDataset(normal_root=normal_root, malicious_root=malicious_root, mode=mode, resize=299,
                               csv_root=csv_root)
    bz = 64
    # torch.cache.empty_cache()
    with torch.no_grad():
        y_true, y_pred = [], []

        dataloader = torch.utils.data.DataLoader(
            dataset = my_dataset,
            batch_size = bz,
            shuffle = True,
            num_workers = 0
        )

        device = torch.device("cuda")
        correct = 0
        total = len(my_dataset)

        for x, y in dataloader:
            x, y = x.to(device), y.to(device)

            output = model.forward(x)
            y_pred.extend(output.sigmoid().flatten().tolist())
            y_true.extend(y.flatten().tolist())

        y_true, y_pred = np.array(y_true), np.array(y_pred)
        fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1)

        AUC = cal_auc(fpr, tpr)

        idx_real = np.where(y_true == 0)[0]
        idx_fake = np.where(y_true == 1)[0]

        r_acc = accuracy_score(y_true[idx_real], y_pred[idx_real] > 0.5)

    return r_acc, AUC

if __name__ == '__main__':
    datasetname = malicious_root.split('/')[-1]

    model = Trainer([0], mode, pretrained_path)
    model.model.load_state_dict(torch.load("/content/drive/MyDrive/models/F3/test_6(git_version)/model2.pth"))

    model.model.to(device)

    model.model.eval()

    r_acc, auc = evaluate(model)
    print("模型在{}数据集上的acc为：".format(datasetname) + str(r_acc))
    print("模型在{}数据集上的auc为：".format(datasetname) + str(auc))
    # model.model.train()