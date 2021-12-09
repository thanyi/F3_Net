import os

import numpy as np
import torch
from sklearn.metrics import roc_curve
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import auc as cal_auc
from torchvision import transforms
from PIL import Image
import os
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score
from dataset import DeepfakeDataset
from my_train import evaluate


os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 在确保所有gpu可用的前提下，可设置多个gpu，否则torch.cuda.is_availabel()显示为false
osenvs = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
gpu_ids = [*range(osenvs)]
loss_freq = 40
mode = 'FAD'  # ['Original', 'FAD', 'LFS', 'Both', 'Mix']
pretrained_path = 'xception-b5690688.pth'
device = torch.device("cuda")

normal_root = r"/content/data/normal_dlib"
malicious_root = r"/content/data/Face2Fcae_dlib"
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



if __name__ == '__main__':
    model = torch.load("/content/drive/MyDrive/models/F3/test_3/model_2.pth")
    model.model.to(device)

    model.model.eval()

    r_acc, auc = evaluate(model)
    print("模型在face2face数据集上的acc为：" + str(r_acc))
    print("模型在face2face数据集上的auc为：" + str(auc))
