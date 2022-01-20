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


max_epoch = 5
loss_freq = 40
mode = 'Both'  # ['Original', 'FAD', 'LFS', 'Both', 'Mix']
pretrained_path = 'models/xception-b5690688.pth'
device = torch.device("cuda")

normal_root = r"/content/data/normal_dlib"
malicious_root = r"/content/data/Deepfakes_dlib"
csv_root = r"/content/data/csv"
path_name = "/content/drive/MyDrive/models/F3/test_8/model2.pth"

def f3netTest():
    datasetname = malicious_root.split('/')[-1]

    model = Trainer([0], mode, pretrained_path)
    model.model.load_state_dict(torch.load(path_name))

    model.model.to(device)

    model.model.eval()

    r_acc, auc = evaluate(model, normal_root, malicious_root, csv_root, "test")
    print(path_name+"模型在{}数据集上的acc为：".format(datasetname) + str(r_acc))
    print(path_name+"模型在{}数据集上的auc为：".format(datasetname) + str(auc))


def kaggle_Dfdc_Test():
    datasetname = malicious_root.split('/')[-1]
    r_acc, auc  = kaggle_evaluate(normal_root, malicious_root, csv_root, "test")
    print(path_name+"模型在{}数据集上的acc为：".format(datasetname) + str(r_acc))
    print(path_name+"模型在{}数据集上的auc为：".format(datasetname) + str(auc))

if __name__ == '__main__':
    f3netTest()