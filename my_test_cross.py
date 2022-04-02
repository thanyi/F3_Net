'''
这个文件主要是针对在colab上调试运行的时候使用的，所以可能是导致不能进行在本地运行
'''
import sys
from torch.utils.data import DataLoader
from models.models import *
from models.xception import *
import numpy as np
from dataset.dataset import DeepfakeDataset
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc as cal_auc
from utils.utils import evaluate
import utils.f3net_conf as f3_config



def f3netTest(malicious_root):

    datasetname = malicious_root.split('/')[-1]

    # 模型调用
    f3net = F3Net()

    f3net.load_state_dict(torch.load(f3_config.model_path_name))
    f3net.to(f3_config.device)
    f3net.eval()

    r_acc, auc = evaluate(f3net, f3_config.normal_root, f3_config.malicious_root, f3_config.csv_root, "test")
    print(f3_config.model_path_name+"模型在{}数据集上的acc为：".format(datasetname) + str(r_acc))
    print(f3_config.model_path_name+"模型在{}数据集上的auc为：".format(datasetname) + str(auc))




if __name__ == '__main__':
    malicious_root=config.malicious_root

    f3netTest(malicious_root)
