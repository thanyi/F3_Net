from datetime import time

from utils.utils import evaluate
from kaggle_dfdc_model.wsdan import WSDAN
from kaggle_dfdc_model import xception_conf as config
import torch
from sklearn.metrics import roc_curve
from torch import nn
from torch.utils.data import DataLoader
from dataset.dataset import DeepfakeDataset
from kaggle_dfdc_model.wsdan_utils import CenterLoss, AverageMeter, TopKAccuracyMetric, batch_augment
import torch.nn.functional as F
import logging
import os

logs = {}
normal_root = r"/content/data/normal_dlib"
malicious_root = r"/content/data/Deepfakes_dlib"
csv_root = r"/content/data/csv"


# loss and metric
loss_container = AverageMeter(name='loss')
raw_metric = TopKAccuracyMetric(topk=(1,))
crop_metric = TopKAccuracyMetric(topk=(1,))
drop_metric = TopKAccuracyMetric(topk=(1,))

logging.basicConfig(
            filename=os.path.join(config.save_dir, config.log_name),
            filemode='w',
            format='%(asctime)s: %(levelname)s: [%(filename)s:%(lineno)d]: %(message)s',
            level=logging.INFO)

def kaggle_training():
    #preparing_dataset
    device = torch.device('cuda')

    train_data = DeepfakeDataset(normal_root=normal_root, malicious_root=malicious_root, mode='train', resize=299,
                                 csv_root=csv_root)
    train_data_size = len(train_data)
    print('train_data_size:', train_data_size)

    train_loader = DataLoader(train_data, 16, shuffle=True)
    # preparing_model
    model = WSDAN(num_classes=2, M=config.num_attentions, net=config.net,pretrained=config.pretrained)
    model.load_state_dict(
        torch.load(r"D:\DeepFakeProject_in_D\deepfake_project\our_code\f3net\kaggle_dfdc_model\ckpt_x.pth"))
    model.to(device)
    model.train()
    num_features = model.num_features

    # preparing_loss
    cross_entropy_loss = nn.CrossEntropyLoss().to(device)
    center_loss = CenterLoss().to(device)

    # preparing_optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

    # preparing_feature_center
    feature_center = torch.zeros(2, config.num_attentions * num_features).to(device)

    model.total_steps = 0
    times = 0
    for epoch in range(0, config.epochs):
        logs['epoch'] = epoch + 1
        logs['lr'] = optimizer.param_groups[0]['lr']

        print("第{}个epoch".format(epoch))
        train_step = 0
        start_time = time.time()

        for i, (X, y) in enumerate(train_loader):
            X = X.to(device)
            y = y.to(device)

            y_pred_raw, feature_matrix, attention_map = model(X, dropout=True)
            # Update Feature Center
            feature_center_batch = F.normalize(feature_center[y], dim=-1)
            feature_center[y] += config.beta * (feature_matrix.detach() - feature_center_batch)
            # dist.all_reduce(feature_center, op=dist.ReduceOp.SUM)
            # feature_center /= ngpus_per_node

            ##################################
            # Attention Cropping
            ##################################
            with torch.no_grad():
                crop_images = batch_augment(X, attention_map[:, :1, :, :], mode='crop', theta=(0.4, 0.6),
                                            padding_ratio=0.1)
            # crop images forward
            y_pred_crop, _, _ = model(crop_images)

            ##################################
            # Attention Dropping
            ##################################
            with torch.no_grad():
                drop_images = batch_augment(X, attention_map[:, 1:, :, :], mode='drop', theta=(0.4, 0.7))

            # drop images forward
            y_pred_drop, _, _ = model(drop_images)

            # loss
            batch_loss = cross_entropy_loss(y_pred_raw, y) + \
                         cross_entropy_loss(y_pred_crop, y) / 3. + \
                         cross_entropy_loss(y_pred_drop, y) / 2. + \
                         center_loss(feature_matrix, feature_center_batch)
            # backward
            batch_loss.backward()
            optimizer.step()

            # metrics: loss and top-1,5 error
            with torch.no_grad():
                epoch_loss = loss_container(batch_loss.item())
                epoch_raw_acc = raw_metric(y_pred_raw, y)
                epoch_crop_acc = crop_metric(y_pred_crop, y)
                epoch_drop_acc = drop_metric(y_pred_drop, y)

            # end of this batch
            epoch_loss = torch.tensor(epoch_loss).cuda()
            epoch_raw_acc = torch.tensor(epoch_raw_acc).cuda()
            epoch_crop_acc = torch.tensor(epoch_crop_acc).cuda()
            epoch_drop_acc = torch.tensor(epoch_drop_acc).cuda()

            batch_info = 'Loss {:.4f}, Raw Acc ({:.2f}), Crop Acc ({:.2f}), Drop Acc ({:.2f})'.format(
                epoch_loss, epoch_raw_acc[0],
                epoch_crop_acc[0], epoch_drop_acc[0])


            print('Train: {}'.format(batch_info))

        model.total_steps += 1


        if model.total_steps % 1 == 0:
            times += 1
            torch.save(model.model.state_dict(),
                       "/content/drive/MyDrive/models/kaggle-dfdc/test_1/model{}.pth".format(times))

        if epoch % 1 == 0:
            model.model.eval()

            r_acc, auc = evaluate(model, normal_root, malicious_root, csv_root, "valid")
            print("本次epoch的acc为：" + str(r_acc))
            print("本次epoch的auc为：" + str(auc))
            model.model.train()


if __name__ == '__main__':
    kaggle_training()