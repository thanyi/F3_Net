import os
from datetime import time

import numpy as np
import torch
from sklearn.metrics import roc_curve
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import auc as cal_auc
from dataset.dataset import DeepfakeDataset
from trainer import Trainer
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score
from utils.utils import evaluate

#gpu设定


os.environ['CUDA_VISIBLE_DEVICES'] = '0'   #在确保所有gpu可用的前提下，可设置多个gpu，否则torch.cuda.is_availabel()显示为false
osenvs = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
gpu_ids = [*range(osenvs)]
max_epoch = 5
loss_freq = 40
mode = 'Both' # ['Original', 'FAD', 'LFS', 'Both', 'Mix']
pretrained_path = 'models/xception-b5690688.pth'

normal_root = r"/content/data/normal_dlib"
malicious_root = r"/content/data/Deepfakes_dlib"
csv_root = r"/content/data/csv"




if __name__ == '__main__':
    writer = SummaryWriter("")
    device = torch.device('cuda')

    train_data = DeepfakeDataset(normal_root=normal_root,malicious_root=malicious_root,mode='train',resize=299,csv_root=csv_root)
    val_data = DeepfakeDataset(normal_root=normal_root,malicious_root=malicious_root,mode='valid',resize=299,csv_root=csv_root)

    train_data_size = len(train_data)
    val_data_size = len(val_data)

    print('train_data_size:',train_data_size)
    print('val_data_size:',val_data_size)

    train_loader = DataLoader(train_data, 32 ,shuffle = True)
    val_loader = DataLoader(val_data, 32 ,shuffle = True)

    # 记录训练的次数
    total_train_step = 0
    # 记录测试的次数
    total_test_step = 0

    # train
    model = Trainer(gpu_ids, mode, pretrained_path)
    model.model.to(device)
    model.total_steps = 0

    times = 0
    for epoch in range(1,10):
        print("第{}个epoch".format(epoch))
        train_step = 0

        for i, (X, y) in enumerate(train_loader):
            model.model.train()

            X = X.to(device)
            y = y.to(device)

            model.set_input(X, y)
            loss = model.optimize_weight()

            train_step+=1
            print("第{}个batch训练完成".format(train_step))
            print("Loss: {}".format(model.loss.item()))

        model.total_steps += 1
        print("epoch训练次数：{}, Loss: {}".format(model.total_steps, model.loss.item()))

        if model.total_steps % 1 == 0:
            times+=1
            torch.save(model.model.state_dict(), "/content/drive/MyDrive/models/F3/test_6(git_version)/model{}.pth".format(times))

        if epoch % 1 == 0:
            model.model.eval()

            r_acc, auc = evaluate(model,normal_root,malicious_root,csv_root,"valid")
            print("本次epoch的acc为：" + str(r_acc))
            print("本次epoch的auc为：" + str(auc))
            model.model.train()

        elif epoch % 5 == 0:
            model.model.eval()

            r_acc, auc = evaluate(model,normal_root,malicious_root,csv_root,"test")
            print("本次epoch的acc为（test）：" + str(r_acc))
            print("本次epoch的auc为（test）：" + str(auc))
            model.model.train()





