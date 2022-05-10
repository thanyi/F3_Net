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



def modelTest():


    # 模型调用
    model = F3Net()
    model_list = []


    root = r"/home/jiahaozhong/model/f3net/f3_eff_srm_se"

    model_list += glob(os.path.join(root,"model?.pth"))

    for model_name in model_list:

        model.load_state_dict(torch.load(model_name))
        model.to(config.device)
        model.eval()
        print(f"当前模型文件：{model_name}")
        r_acc, auc = evaluate(model, config.dfdc_root , config.dfdc_syn_root, config.dfdc_csv_root, "test")
        print(model_name+"模型在DFDC数据集上的acc为：" + str(r_acc))
        print(model_name+"模型在DFDC数据集上的auc为：" + str(auc))
        print("---------------------------")



if __name__ == '__main__':


    modelTest()
