'''
这个文件是我们项目自己准备使用的模型训练脚本文件
'''
import datetime
import time

import torch
from sklearn.metrics import roc_curve
from torch import nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import auc as cal_auc
from dataset.dataset import DeepfakeDataset
from models.models_effi_srm import *
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score
from utils.utils import evaluate
import utils.f3net_conf as config
from trainer import Trainer





def f3net_training(iftrained = False):
    device = torch.device('cuda')

    train_data = DeepfakeDataset(normal_root=config.normal_root, malicious_root=config.malicious_root, mode='train',
                                 resize=380,
                                 csv_root=config.csv_root)
    train_data_size = len(train_data)
    print('train_data_size:', train_data_size)

    bz = 1
    train_loader = DataLoader(train_data, bz, shuffle=True)

    # train

    model = F3Net(mode = "Both")

    if iftrained == True:
        model.load_state_dict(
            torch.load(r"C:\Users\ethanyi\Desktop\security_competition\models\F3net_effi_srm\model1.pth"))

    model.to(device)


    if model.mode =="FAD":
        for param in model.FAD_eff.parameters():
            param.requires_grad = False
    elif model.mode =="SRM":
        for param in model.SRM_eff.parameters():
            param.requires_grad = False
    else:
        for param in model.FAD_eff.parameters():
            param.requires_grad = False
        for param in model.SRM_eff.parameters():
            param.requires_grad = False


    loss_fn = nn.BCEWithLogitsLoss().to(device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0002, betas=(0.9, 0.999))
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    times = 0
    for epoch in range(1, 10):
        starttime = time.time()
        print("----------------第{}个epoch--------------".format(epoch))

        for i, (X, y) in enumerate(train_loader):
            model.train()

            X = X.to(device)

            y = y.to(device)

            output = model(X)

            output = output.squeeze(-1)

            loss = loss_fn(output, y.float())


            optimizer.zero_grad()

            loss.backward()
            optimizer.step()


            if i % 10 == 0:
                print(f"{i}/{train_data_size//bz} batch训练完成 " + f"Loss: {loss.item()}")

            if i % 1000 ==0:
                endtime = time.time()
                during_time  = endtime - starttime
                minutes, s = divmod(during_time, 60)
                print(f"已用时间 : {minutes}分钟 。。。。。。。")

        print("epoch训练次数：{}, Loss: {}".format(epoch, loss.item()))

        if epoch % 1 == 0:
            times += 2
            torch.save(model.state_dict(),
                       r"C:\Users\ethanyi\Desktop\security_competition\models\F3net_effi_srm\model{}.pth".format(times))
            print("模型保存成功")
            model.eval()

            r_acc, auc = evaluate(model, config.normal_root, config.malicious_root, config.csv_root, "valid")
            print("本次epoch的acc为：" + str(r_acc))
            print("本次epoch的auc为：" + str(auc))
            model.train()

        scheduler.step()



if __name__ == '__main__':
    # model = F3Net()
    # model.load_state_dict(torch.load(r"C:\Users\ethanyi\Desktop\security_competition\models\F3net_effi_srm\model1.pth"))
    # model.to("cuda:0")
    # r_acc, auc = evaluate(model, config.normal_root, config.malicious_root, config.csv_root, "valid")
    # print("本次epoch的acc为：" + str(r_acc))
    # print("本次epoch的auc为：" + str(auc))

    f3net_training(iftrained=True)




