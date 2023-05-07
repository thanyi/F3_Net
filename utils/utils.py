import torch
import os
import numpy as np
import random
from torch.utils import data
from torchvision.transforms import transforms
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
from torch.autograd import Variable

# class AMSoftmaxLoss(nn.Module):
#     def __init__(self, num_classes, feat_dim, margin=0.35, scale=30.0):
#         super(AMSoftmaxLoss, self).__init__()
#         self.num_classes = num_classes
#         self.feat_dim = feat_dim
#         self.margin = margin
#         self.scale = scale
#         self.weight = nn.Parameter(torch.FloatTensor(num_classes, feat_dim))
#         nn.init.xavier_uniform_(self.weight)

#     def forward(self, feat, labels):
#         assert feat.size(1) == self.feat_dim, "输入特征维度与预期不符。"
#         assert feat.size(0) == labels.size(0), "特征和标签的 batch size 不匹配。"
#         labels = labels.long()

#         # L2 范数归一化
#         feat_norm = F.normalize(feat, p=2, dim=1)
#         weight_norm = F.normalize(self.weight, p=2, dim=1)

#         # 计算余弦相似度
#         cos_theta = torch.mm(feat_norm, weight_norm.t())

#         # 添加余弦边缘
#         target_logit = cos_theta[torch.arange(0, feat.size(0), dtype=torch.long), labels].view(-1, 1)
#         sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
#         cos_theta_m = cos_theta - self.margin
#         cos_theta_m = target_logit * cos_theta_m / (cos_theta - target_logit * (1 - self.margin))

#         # 更新角度
#         final_theta = cos_theta * 1.0
#         final_theta.scatter_(1, labels.view(-1, 1).long(), cos_theta_m)
#         final_theta *= self.scale

#         # 计算 AM-Softmax Loss
#         loss = F.cross_entropy(final_theta, labels)
#         return loss

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1).long()

        logpt = F.log_softmax(input,dim=1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.s

# class FocalLoss(nn.Module):
#     def __init__(self, alpha=0.25, gamma=2.0):
#         super(FocalLoss, self).__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#
#     def forward(self, input, target):
#         # Sigmoid激活函数应用在输入上
#         prob = torch.sigmoid(input)
#
#         # 计算二元交叉熵损失
#         bce_loss = F.binary_cross_entropy(prob, target, reduction='none')
#
#         # 计算focal权重
#         focal_weight = torch.pow(torch.abs(target - prob), self.gamma)
#
#         # 应用权重和alpha系数
#         focal_loss = self.alpha * focal_weight * bce_loss
#
#         # 损失求和并平均
#         return focal_loss.mean()

class AMSoftmax(nn.Module):
    def __init__(self,
                 in_feats,
                 n_classes=10,
                 m=0.35,
                 s=30):
        super(AMSoftmax, self).__init__()
        self.m = m
        self.s = s
        self.in_feats = in_feats
        self.W = torch.nn.Parameter(torch.randn(in_feats, n_classes), requires_grad=True)
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.W, gain=1)

    def forward(self, x, lb):
        assert x.size()[0] == lb.size()[0]
        assert x.size()[1] == self.in_feats
        lb = lb.long()

        x_norm = torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        x_norm = torch.div(x, x_norm)
        w_norm = torch.norm(self.W, p=2, dim=0, keepdim=True).clamp(min=1e-12)
        w_norm = torch.div(self.W, w_norm)
        costh = torch.mm(x_norm, w_norm)
        # print(x_norm.shape, w_norm.shape, costh.shape)
        lb_view = lb.view(-1, 1)
        if lb_view.is_cuda: lb_view = lb_view.cpu()
        delt_costh = torch.zeros(costh.size()).scatter_(1, lb_view, self.m)
        if x.is_cuda: delt_costh = delt_costh.cuda()
        costh_m = costh - delt_costh
        costh_m_s = self.s * costh_m
        loss = self.ce(costh_m_s, lb)
        return loss


def evaluate(model, normal_root, malicious_root, csv_root, mode='test', loss_mode='logits'):
    my_dataset = DeepfakeDataset(normal_root=normal_root, malicious_root=malicious_root, mode=mode, resize=380,
                                 csv_root=csv_root)
    malicious_name = malicious_root.split('/')[-1]
    print("This is the {} {} dataset!".format(malicious_name, mode))
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

        for i, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            # print(x.shape)
            output = model(x)

            if loss_mode != 'logits':
                output = torch.nn.Softmax(dim=1)(output)
                output = output[:, 1].unsqueeze(1)  # 是选择，表示就是选择第1列（第0列是起点）

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





