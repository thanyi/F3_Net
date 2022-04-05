'''
这个文件是我们项目自己准备使用的模型训练脚本文件
'''


import torch
from sklearn.metrics import roc_curve
from torch import nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import auc as cal_auc
from dataset.dataset import DeepfakeDataset
from models.models_effi import *
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score
from utils.utils import evaluate,CenterLoss
import utils.f3net_conf as config
from trainer import Trainer



def modelsize(model, input, type_size=4):
    para = sum([np.prod(list(p.size())) for p in model.parameters()])
    print('Model {} : params: {:4f}M'.format(model._get_name(), para * type_size / 1000 / 1000))

    input_ = input.clone()
    input_.requires_grad_(requires_grad=False)

    mods = list(model.modules())
    out_sizes = []

    for i in range(1, len(mods)):
        m = mods[i]
        if isinstance(m, nn.ReLU):
            if m.inplace:
                continue
        out = m(input_)
        out_sizes.append(np.array(out.size()))
        input_ = out

    total_nums = 0
    for i in range(len(out_sizes)):
        s = out_sizes[i]
        nums = np.prod(np.array(s))
        total_nums += nums


    print('Model {} : intermedite variables: {:3f} M (without backward)'
          .format(model._get_name(), total_nums * type_size / 1000 / 1000))
    print('Model {} : intermedite variables: {:3f} M (with backward)'
          .format(model._get_name(), total_nums * type_size*2 / 1000 / 1000))

def f3net_training():
    device = torch.device('cuda')

    train_data = DeepfakeDataset(normal_root=config.normal_root, malicious_root=config.malicious_root, mode='train',
                                 resize=380,
                                 csv_root=config.csv_root)
    train_data_size = len(train_data)
    print('train_data_size:', train_data_size)

    train_loader = DataLoader(train_data, 16, shuffle=True)

    # train

    model = F3Net()
    model.to(device)

    for param in model.FAD_eff.parameters():
        param.requires_grad = False
    for param in model.LFS_eff.parameters():
        param.requires_grad = False



    loss_fn = nn.BCEWithLogitsLoss().to(device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0002, betas=(0.9, 0.999))
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    times = 0
    for epoch in range(1, 10):
        print("第{}个epoch".format(epoch))
        train_step = 0

        for i, (X, y) in enumerate(train_loader):
            model.train()

            X = X.to(device)
            print(X.shape)
            y = y.to(device)

            output = model(X)
            loss = loss_fn(output, y)

            if i % 5 ==0:
                optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            train_step += 1
            if train_step % 10 == 0:
                print("第{}个batch训练完成".format(train_step))
                print("Loss: {}".format(loss.item()))

        print("epoch训练次数：{}, Loss: {}".format(model.total_steps, model.loss.item()))

        if epoch % 2 == 0:
            times += 1
            torch.save(model.model.state_dict(),
                       "/content/drive/MyDrive/models/F3/test_6(git_version)/model{}.pth".format(times))

            model.eval()

            r_acc, auc = evaluate(model, config.normal_root, config.malicious_root, config.csv_root, "valid")
            print("本次epoch的acc为：" + str(r_acc))
            print("本次epoch的auc为：" + str(auc))
            model.train()

        scheduler.step()



if __name__ == '__main__':
    # model = F3Net()
    # input = torch.ones(16, 3, 380, 380)
    # print(input.shape)
    # modelsize(model,input)

    f3net_training()




