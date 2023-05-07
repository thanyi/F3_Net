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
from utils.utils import evaluate
import utils.f3net_conf as config

from torch.utils.tensorboard import SummaryWriter
import os
from utils.utils import AMSoftmax,FocalLoss

def record(r_acc, auc ,con_mat ,recall, precision,epoch,data_name):
    print("本次epoch的acc为：" + str(r_acc))
    print("本次epoch的auc为：" + str(auc))
    print(f"{con_mat}为本次epoch的混淆矩阵" )

    valid_writer.add_scalar(f'Accuracy ({data_name})', r_acc , epoch+1)
    recall_writer.add_scalar(f'precision and Recall ({data_name})', recall , epoch+1)
    precision_writer.add_scalar(f'precision and Recall ({data_name})', precision , epoch+1)

    with open (f"/hy-nas/model/{model_name}_{epoch}.txt","a+") as f:
        f.write(f"Accuracy ({data_name})：" + "{:.2f}%\n".format(round(r_acc, 4) * 100) )
        f.write(f"Recall ({data_name})：" + "{:.2f}%\n".format(round(recall, 4) * 100) )
        f.write(f"precision ({data_name})：" + "{:.2f}%\n".format(round(precision, 4) * 100) )
        f.write(f"AUC ({data_name})：" + "{:.2f}%\n".format(round(auc, 4) * 100) )
        f.write("\n")



def f3net_training(iftrained=False):
    global test_writer, valid_writer, loss_writer, recall_writer, precision_writer,model_name
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

    # 初始化gpu设备
    device = config.device

    # 载入dataset和loader
    bz = 4
    train_data = DeepfakeDataset(normal_root=config.dfdc_real_root, malicious_root=config.dfdc_syn_root, mode='train',
                                 resize=380,
                                 csv_root=config.dfdc_csv_root)
    train_data_size = len(train_data)
    print('train_data_size:', train_data_size)
    train_loader = DataLoader(train_data, bz, shuffle=True)

    # 初始化model
    model_name = input("请输入model_name: ")
    model_loss = input("请输入model_loss(logits or AM  or Focal): ")
    model = F3Net(mode="Both",loss_mode = model_loss)

    if iftrained == True:
        model_to_load = input("载入的模型为：")
        model.load_state_dict(torch.load(f"/hy-nas/model/{model_to_load}"))
        print(f"{model_to_load}载入成功")

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
    for param in model.eff.fc.parameters():
        param.requires_grad = True

    # 初始化loss函数
    if model_loss =='logits':
        loss_fn = nn.BCEWithLogitsLoss().to(device)
    elif model_loss =='AM' :
        loss_fn = AMSoftmax(2, 2).to(device)
    else:
        loss_fn = FocalLoss().to(device)


    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.004)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=4, eta_min=5e-6)

    running_loss = 0.0
    running_loss_rate = 0

    for epoch in range(1,50):
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
                loss_writer.add_scalar('训练时Loss值的变化', running_loss/running_loss_rate  , (epoch-1) * len(train_loader) + i)
                running_loss = 0.0
                running_loss_rate = 0

        print("epoch训练次数：{}, Loss: {}".format(epoch, loss.item()))
        torch.save(model.state_dict(),
                   f"/hy-nas/model/{model_name}_{epoch}.pth")
        print("模型保存成功")

        if epoch % 1 == 0:
            model.eval()

            with open(f"/hy-nas/model/{model_name}_{epoch}.txt", "a+") as f:
                f.write(f"this is the {epoch}'s epoch\n")

            r_acc, auc, con_mat, recall, precision = evaluate(model, config.ff_real_root, config.ff_syn_root,config.ff_csv_root, "test",loss_mode=model_loss)
            record(r_acc, auc, con_mat, recall, precision,epoch, data_name='ff_test')

            r_acc, auc ,con_mat, recall ,precision= evaluate(model, config.celeb_real_root, config.celeb_syn_root, config.celeb_csv_root, "test",loss_mode=model_loss)
            record(r_acc, auc ,con_mat, recall ,precision,epoch, data_name='celeb_test')


            r_acc, auc ,con_mat ,recall, precision = evaluate(model, config.dfdc_root, config.dfdc_syn_root, config.dfdc_csv_root, "test",loss_mode=model_loss)
            record(r_acc, auc, con_mat, recall, precision, epoch, data_name='dfdc_test')
            model.train()

        scheduler.step()


if __name__ == '__main__':
    f3net_training(iftrained=False)




