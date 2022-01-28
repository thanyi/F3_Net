import glob

import torch
import os
import numpy as np
import random
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision import transforms as trans, transforms
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc as cal_auc
from PIL import Image
import sys
import logging
from kaggle_dfdc_model.wsdan import WSDAN
from dataset.dataset import DeepfakeDataset
from trainer import Trainer
import utils.kaggle_conf as config


model = WSDAN(num_classes=2, M=8, net='xception', pretrained=config.xception_pretained_path)
model.load_state_dict(torch.load(config.wsdan_pretained_path))

model.to(config.device)
model.eval()

test_data = DeepfakeDataset(normal_root=config.normal_root, malicious_root= config.malicious_root, mode='test', resize=config.resize,
                             csv_root=config.csv_root)
test_loader = DataLoader(test_data, 32, shuffle=True,num_workers=0)

y_pred=[]
y_true=[]

for x, y in test_loader:
    x, y = x.to(config.device), y.to(config.device)

    output, f, a = model(x)
    output1 = output.argmax(1).tolist()
    output2 = output.argmax(1).flatten().tolist()
    y_pred.extend(output)
    y_true.extend(y.flatten().tolist())

y_true, y_pred = np.array(y_true), np.array(y_pred)
fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1)

AUC = cal_auc(fpr, tpr)

for i in range(len(y_pred)):
    if y_pred[i] < 0.5:
        y_pred[i] = 0
    else:
        y_pred[i] = 1

r_acc = accuracy_score(y_true, y_pred)
print(r_acc)