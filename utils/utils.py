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
import torch.nn.functional as F
from torch import nn
from sklearn.metrics import confusion_matrix
from dataset.dataset import DeepfakeDataset


class AMSoftmaxLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, margin=0.35, scale=30.0):
        super(AMSoftmaxLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.margin = margin
        self.scale = scale
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, feat_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, feat, labels):
        assert feat.size(1) == self.feat_dim, "输入特征维度与预期不符。"
        assert feat.size(0) == labels.size(0), "特征和标签的 batch size 不匹配。"

        # L2 范数归一化
        feat_norm = F.normalize(feat, p=2, dim=1)
        weight_norm = F.normalize(self.weight, p=2, dim=1)

        # 计算余弦相似度
        cos_theta = torch.mm(feat_norm, weight_norm.t())

        # 添加余弦边缘
        target_logit = cos_theta[torch.arange(0, feat.size(0)), labels].view(-1, 1)
        sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
        cos_theta_m = cos_theta - self.margin
        cos_theta_m = target_logit * cos_theta_m / (cos_theta - target_logit * (1 - self.margin))

        # 更新角度
        final_theta = cos_theta * 1.0
        final_theta.scatter_(1, labels.view(-1, 1).long(), cos_theta_m)
        final_theta *= self.scale

        # 计算 AM-Softmax Loss
        loss = F.cross_entropy(final_theta, labels)
        return loss


def evaluate(model, normal_root,malicious_root,csv_root, mode='test',loss_mode='logits'):

    my_dataset = DeepfakeDataset(normal_root=normal_root, malicious_root=malicious_root, mode=mode, resize=380,
                                 csv_root=csv_root)
    malicious_name = malicious_root.split('/')[-1]
    print("This is the {} {} dataset!".format(malicious_name,mode))
    print("dataset size:{}".format(len(my_dataset)))

    bz = 16
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

            if loss_mode!='logits':
                output = torch.nn.Softmax(dim=1)(output)
                output = output[:, 1].unsqueeze(1)  # [:, 1] 表示舍掉第0维只要第1维

            y_pred.extend(output.sigmoid().flatten().tolist())
            y_true.extend(y.flatten().tolist())

        print(" ")

        y_true, y_pred = np.array(y_true), np.array(y_pred)
        y_pred = np.where(y_pred >= 0.5, 1, 0)
        fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1)

        AUC = cal_auc(fpr, tpr)



        r_acc = accuracy_score(y_true, y_pred)
        con_mat = confusion_matrix(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)

    return r_acc, AUC, con_mat, recall, precision







