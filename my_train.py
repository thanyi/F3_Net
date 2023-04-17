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
from models.models_effi_srm_se_2 import *
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score
from utils.utils import evaluate
import utils.f3net_conf as config
from trainer import Trainer
from torch.utils.tensorboard import SummaryWriter
import os

def f3net_training(iftrained=False):
    # 将 test accuracy 保存到 "tensorboard/train" 文件夹
    log_dir = os.path.join('/tf_logs', 'test')
    test_writer = SummaryWriter(log_dir=log_dir)

    # 将 valid accuracy 保存到 "tensorboard/valid" 文件夹
    log_dir = os.path.join('/tf_logs', 'valid')
    valid_writer = SummaryWriter(log_dir=log_dir)

    # 将 loss 保存到 "tensorboard/loss" 文件夹
    log_dir = os.path.join('/tf_logs', 'loss')
    loss_writer = SummaryWriter(log_dir=log_dir)

    # 将 recall 保存到 "tensorboard/recall" 文件夹
    log_dir = os.path.join('/tf_logs', 'recall')
    recall_writer = SummaryWriter(log_dir=log_dir)

    # 将 precision 保存到 "tensorboard/precision" 文件夹
    log_dir = os.path.join('/tf_logs', 'precision')
    precision_writer = SummaryWriter(log_dir=log_dir)

    device = config.device

    train_data = DeepfakeDataset(normal_root=config.normal_root, malicious_root=config.malicious_root, mode='train',
                                 resize=380,
                                 csv_root=config.csv_root)
    train_data_size = len(train_data)
    print('train_data_size:', train_data_size)

    bz = 4
    train_loader = DataLoader(train_data, bz, shuffle=True)

    # train

    model = F3Net(mode="Both")
    model_name = input("请输入model_name")


    if iftrained == True:
        model.load_state_dict(
            torch.load(r"/hy-nas/model/model-eff-se-3_1.pth"))

    model.to(device)

    print("mode: Both")
    # for param in model.FAD_eff.parameters():
    # param.requires_grad = False
    # for param in model.SRM_eff.parameters():
    # param.requires_grad = False
    # for param in model.FAD_eff.encoder.conv_stem.parameters():
    # param.requires_grad = True
    # for param in model.SRM_eff.encoder.conv_stem.parameters():
    # param.requires_grad = True

    for param in model.eff.parameters():
        param.requires_grad = False
    for param in model.eff.encoder.conv_stem.parameters():
        param.requires_grad = True

    loss_fn = nn.BCEWithLogitsLoss().to(device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.1)
    scheduler = lr_scheduler.StepLR(optimizer=optimizer,step_size=1,gamma=0.2)

    times = 0
    running_loss = 0.0
    running_loss_rate = 0

    for epoch in range(50):
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

            running_loss += loss.item()
            running_loss_rate +=1
            if i % 500 == 0:
                print(f"{i}/{train_data_size // bz} batch训练完成 " + f"Loss: {loss.item()}")

            if i % 1000 == 0:
                endtime = time.time()
                during_time = endtime - starttime
                minutes, s = divmod(during_time, 60)
                print(f"已用时间 : {minutes}分钟。。。。。。。")
                # 开始绘制loss曲线，取1000次iteration的平均值
                loss_writer.add_scalar('训练时Loss值的变化', running_loss/running_loss_rate  , epoch * len(train_loader) + i)
                running_loss = 0.0
                running_loss_rate = 0

        print("epoch训练次数：{}, Loss: {}".format(epoch, loss.item()))

        if epoch % 1 == 0:
            times += 1
            torch.save(model.state_dict(),
                       f"/hy-nas/model/{model_name}_{times}.pth")
            print("模型保存成功")
            model.eval()


            r_acc, auc ,con_mat, recall ,precision= evaluate(model, config.normal_root, config.malicious_root, config.csv_root, "valid")
            print("本次epoch的acc为：" + str(r_acc))
            print("本次epoch的auc为：" + str(auc))
            print(f"{con_mat}为本次epoch的混淆矩阵" )

            valid_writer.add_scalar('Accuracy (Train)', r_acc , epoch+1)
            recall_writer.add_scalar('precision and Recall (Train)', recall , epoch+1)
            precision_writer.add_scalar('precision and Recall (Train)', precision , epoch+1)

            with open (f"/hy-nas/model/{model_name}_{times}.txt","a+") as f:
                f.write("Accuracy (Train)：" + str(round(r_acc, 2) * 100) + "%")
                f.write("Recall (Train)：" + str(round(recall, 2) * 100) + "%")
                f.write("precision (Train)：" + str(round(precision, 2) * 100) + "%")
                f.write("\n")


            r_acc, auc ,con_mat ,recall, precision = evaluate(model, config.dfdc_root, config.dfdc_syn_root, config.dfdc_csv_root, "test")
            print("-------本次epoch在DFDC上的acc为：" + str(r_acc))
            print("-------本次epoch在DFDC上的auc为：" + str(auc))
            print(f"{con_mat}为本次epoch的混淆矩阵")

            precision_writer.add_scalar('precision and Recall (Test)', precision, epoch + 1)
            recall_writer.add_scalar('precision and Recall (Test)', recall, epoch + 1)
            test_writer.add_scalar('Accuracy (Test)', r_acc, epoch + 1)

            with open (f"/hy-nas/model/{model_name}_{times}.txt","a+") as f:
                f.write("Accuracy (Test)：" + str(round(r_acc, 2) * 100) + "%")
                f.write("Recall (Test)：" + str(round(recall, 2) * 100) + "%")
                f.write("precision (Test)：" + str(round(precision, 2) * 100) + "%")

            model.train()

        scheduler.step()


if __name__ == '__main__':
    # model = F3Net()
    # model.load_state_dict(torch.load(r"C:\Users\ethanyi\Desktop\security_competition\models\F3net_effi_srm\model1.pth"))
    # model.to("cuda:0")
    # r_acc, auc = evaluate(model, config.normal_root, config.malicious_root, config.csv_root, "valid")
    # print("本次epoch的acc为：" + str(r_acc))
    # print("本次epoch的auc为：" + str(auc))

    f3net_training(iftrained=False)




