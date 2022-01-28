'''
这个文件主要是针对在colab上调试运行的时候使用的，所以可能是导致不能进行在本地运行
'''

from torch.utils.data import DataLoader
from trainer import *
from models.xception import *
import numpy as np
from dataset.dataset import DeepfakeDataset
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc as cal_auc
from utils.utils import evaluate,kaggle_evaluate
import utils.f3net_conf as config


def f3netTest():
    datasetname = config.malicious_root.split('/')[-1]

    model = Trainer([0], config.mode, config.pretrained_path)
    model.model.load_state_dict(torch.load(config.model_path_name))

    model.model.to(config.device)

    model.model.eval()

    r_acc, auc = evaluate(model, config.normal_root, config.malicious_root, config.csv_root, "test")
    print(config.model_path_name+"模型在{}数据集上的acc为：".format(datasetname) + str(r_acc))
    print(config.model_path_name+"模型在{}数据集上的auc为：".format(datasetname) + str(auc))


def kaggle_Dfdc_Test():
    datasetname = config.malicious_root.split('/')[-1]
    r_acc, auc  = kaggle_evaluate(config.normal_root, config.malicious_root, config.csv_root, "test")
    print(config.model_path_name+"模型在{}数据集上的acc为：".format(datasetname) + str(r_acc))
    print(config.model_path_name+"模型在{}数据集上的auc为：".format(datasetname) + str(auc))

if __name__ == '__main__':
    f3netTest()