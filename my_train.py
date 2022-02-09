
import torch
from sklearn.metrics import roc_curve
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import auc as cal_auc
from dataset.dataset import DeepfakeDataset
from trainer import Trainer
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score
from utils.utils import evaluate,CenterLoss
import utils.f3net_conf as config


def f3net_training():
    device = torch.device('cuda')

    train_data = DeepfakeDataset(normal_root=config.normal_root, malicious_root=config.malicious_root, mode='train', resize=299,
                                 csv_root=config.csv_root)
    train_data_size = len(train_data)
    print('train_data_size:', train_data_size)

    train_loader = DataLoader(train_data, 16, shuffle=True)

    # train
    model = Trainer(config.gpu_ids, config.mode, config.pretrained_path)
    model.model.to(device)
    model.total_steps = 0

    times = 0
    for epoch in range(1, 10):
        print("第{}个epoch".format(epoch))
        train_step = 0

        for i, (X, y) in enumerate(train_loader):
            model.model.train()

            X = X.to(device)
            y = y.to(device)

            model.set_input(X, y)
            loss = model.optimize_weight()

            train_step += 1
            if train_step %10 ==0:
                print("第{}个batch训练完成".format(train_step))
                print("Loss: {}".format(model.loss.item()))

        model.total_steps += 1
        print("epoch训练次数：{}, Loss: {}".format(model.total_steps, model.loss.item()))

        if model.total_steps % 1 == 0:
            times += 1
            torch.save(model.model.state_dict(),
                       "/content/drive/MyDrive/models/F3/test_6(git_version)/model{}.pth".format(times))

        if epoch % 1 == 0:
            model.model.eval()

            r_acc, auc = evaluate(model, config.normal_root, config.malicious_root, config.csv_root, "valid")
            print("本次epoch的acc为：" + str(r_acc))
            print("本次epoch的auc为：" + str(auc))
            model.model.train()





if __name__ == '__main__':
    f3net_training()





