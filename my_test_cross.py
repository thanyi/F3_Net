'''
这个文件主要是针对在colab上调试运行的时候使用的，所以可能是导致不能进行在本地运行
'''
import os
import sys
from glob import glob

from torch.utils.data import DataLoader
from models.models_effi_srm_se_2 import *

import numpy as np
from dataset.dataset import DeepfakeDataset
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc as cal_auc
from utils.utils import evaluate
import utils.f3net_conf as config

import timm.models.efficientnet as effnet
from models.models_effi_srm_se_2 import get_eff_state_dict


import torch
import torch.nn as nn




class EffNet(nn.Module):
    def __init__(self, arch='b7'):
        super(EffNet, self).__init__()
        fc_size = {'b1': 1280, 'b2': 1408, 'b3': 1536, 'b4': 1792,
                   'b5': 2048, 'b6': 2304, 'b7': 2560}
        assert arch in fc_size.keys()
        effnet_model = getattr(effnet, 'tf_efficientnet_%s_ns' % arch)
        self.encoder = effnet_model()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(fc_size[arch], 1)

    def forward(self, x):
        x = self.encoder.forward_features(x)
        x = self.avg_pool(x).flatten(1)
        x = self.dropout(x)
        x = self.fc(x)
        return x



def modelTest():
    root = r"D:\DeepFakeProject_in_D\deepfake_project\eliminate_project\screen_shot\face_recognition\deepware.pt"

    model = EffNet("b7")

    state_dict = torch.load(root)

    model_name = "deepware"


    model.load_state_dict(torch.load(root))

    model.to(config.device)
    model.eval()
    print(f"当前模型文件：{root}")
    data_name = input ("要测试的数据集是：")
    if data_name == "dfdc":
        real_root = config.dfdc_real_root
        syn_root = config.dfdc_syn_root
        csv_root = config.dfdc_csv_root
    elif data_name == "celeb-df":
        real_root = config.celeb_real_root
        syn_root = config.celeb_syn_root
        csv_root = config.celeb_csv_root
    elif data_name == "ff++":
        real_root = config.ff_real_root
        syn_root = config.ff_syn_root
        csv_root = config.ff_csv_root
    else :
        print ("data name error!")




    r_acc, auc ,con_mat ,recall, precision  = evaluate(model, real_root , syn_root , csv_root , "test")
    print(model_name+f"模型在{data_name}数据集上的acc为：" + str(r_acc))
    print(model_name+f"模型在{data_name}数据集上的auc为：" + str(auc))
    print(model_name+f"模型在{data_name}数据集上的recall为：" + str(recall))
    print(model_name+f"模型在{data_name}数据集上的precision为：" + str(precision))
    print("---------------------------")



if __name__ == '__main__':


    modelTest()
