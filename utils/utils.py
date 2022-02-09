import torch
import os
import numpy as np
import random
from torch.utils import data
from torchvision import transforms as trans
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc as cal_auc
from PIL import Image
import sys
import logging
from kaggle_dfdc_model.wsdan import WSDAN
from torch import nn

from dataset.dataset import DeepfakeDataset
from trainer import Trainer
import kaggle_conf as kaggle_config

class FFDataset(data.Dataset):

    def __init__(self, dataset_root, frame_num=300, size=299, augment=True):
        self.data_root = dataset_root
        self.frame_num = frame_num
        self.train_list = self.collect_image(self.data_root)   #将文件夹中的每一个图像（绝对）路径加到train_list中
        if augment:
            self.transform = trans.Compose([
                                            trans.RandomHorizontalFlip(p=0.5),
                                            trans.ToTensor()])
            print("Augment True!")
        else:
            self.transform = trans.ToTensor()
        self.max_val = 1.
        self.min_val = -1.
        self.size = size

    def collect_image(self, root):
        image_path_list = []
        for split in os.listdir(root):
            split_root = os.path.join(root, split)
            img_list = os.listdir(split_root)
            random.shuffle(img_list)
            img_list = img_list if len(img_list) < self.frame_num else img_list[:self.frame_num]
            for img in img_list:
                img_path = os.path.join(split_root, img)
                image_path_list.append(img_path)
        return image_path_list

    def read_image(self, path):
        img = Image.open(path)
        return img

    def resize_image(self, image, size):
        img = image.resize((size, size))
        return img

    def __getitem__(self, index):
        image_path = self.train_list[index]
        img = self.read_image(image_path)
        img = self.resize_image(img,size=self.size)
        img = self.transform(img)
        img = img * (self.max_val - self.min_val) + self.min_val
        return img

    def __len__(self):
        return len(self.train_list)



def get_dataset(name = 'train', size=299, root='E:\\Dataset_pre\\ff++_dataset\\', frame_num=300, augment=True):
    root = os.path.join(root, name)
    fake_root = os.path.join(root,'fake')

    fake_list = ['Deepfakes', 'FaceSwap', 'Face2Face', 'NeuralTextures']

    total_len = len(fake_list)
    dset_lst = []
    for i in range(total_len):
        fake = os.path.join(fake_root , fake_list[i])
        dset = FFDataset(fake, frame_num, size, augment)
        dset.size = size
        dset_lst.append(dset)
    return torch.utils.data.ConcatDataset(dset_lst), total_len

def evaluate(model, normal_root,malicious_root,csv_root, mode='valid',):

    my_dataset = DeepfakeDataset(normal_root=normal_root, malicious_root=malicious_root, mode=mode, resize=299,
                                 csv_root=csv_root)
    malicious_name = malicious_root.split('/')[-1]
    print("This is the {} {} dataset!".format(malicious_name,mode))
    print("dataset size:{}".format(len(my_dataset)))

    bz = 64
    # torch.cache.empty_cache()
    with torch.no_grad():
        y_true, y_pred = [], []

        dataloader = torch.utils.data.DataLoader(
            dataset=my_dataset,
            batch_size=bz,
            shuffle=True,
            num_workers=0
        )

        device = torch.device("cuda")
        correct = 0
        total = len(my_dataset)

        for x, y in dataloader:
            x, y = x.to(device), y.to(device)

            output = model.forward(x)
            y_pred.extend(output.sigmoid().flatten().tolist())
            y_true.extend(y.flatten().tolist())

        y_true, y_pred = np.array(y_true), np.array(y_pred)
        fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1)

        AUC = cal_auc(fpr, tpr)

        for i in range(len(y_pred)):
            if y_pred[i] < 0.5:
                y_pred[i] = 0
            else:
                y_pred[i] = 1

        r_acc = accuracy_score(y_true, y_pred)

    return r_acc, AUC

def kaggle_evaluate(normal_root,malicious_root,csv_root, mode='valid',):
    device = torch.device("cuda")
    model = WSDAN(num_classes=2, M=8, net='xception',
                  pretrained=kaggle_config.xception_pretained_path)
    model.load_state_dict(torch.load(kaggle_config.wsdan_pretained_path))

    model.to(device)
    model.eval()

    my_dataset = DeepfakeDataset(normal_root=normal_root, malicious_root=malicious_root, mode=mode, resize=299,
                                 csv_root=csv_root)
    malicious_name = malicious_root.split('/')[-1]
    print("This is the {} {} dataset!".format(malicious_name,mode))
    print("dataset size:{}".format(len(my_dataset)))

    bz = 64
    # torch.cache.empty_cache()
    with torch.no_grad():
        y_true, y_pred = [], []

        dataloader = torch.utils.data.DataLoader(
            dataset=my_dataset,
            batch_size=bz,
            shuffle=True,
            num_workers=0
        )

        device = torch.device("cuda")
        correct = 0
        total = len(my_dataset)

        for x, y in dataloader:
            x, y = x.to(device), y.to(device)

            output, f ,a = model(x)

            y_pred.extend(output.argmax(1).flatten().tolist())
            y_true.extend(y.flatten().tolist())

        y_true, y_pred = np.array(y_true), np.array(y_pred)
        fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1)

        AUC = cal_auc(fpr, tpr)

        for i in range(len(y_pred)):
            if y_pred[i] < 0.5:
                y_pred[i] = 0
            else:
                y_pred[i] = 1

        r_acc = accuracy_score(y_true, y_pred)

    return r_acc, AUC


##############################################
# Center Loss for Attention Regularization
##############################################
class CenterLoss(nn.Module):
    def __init__(self):
        super(CenterLoss, self).__init__()
        self.l2_loss = nn.MSELoss(reduction='sum')

    def forward(self, outputs, targets):
        return self.l2_loss(outputs, targets) / outputs.size(0)


def check_video_acc():
    normal_root = r"/content/data/normal_dlib"
    malicious_root = r"/content/data/FaceSwap_dlib"
    csv_root = r"/content/data/csv"

    device = torch.device("cuda")

    model = Trainer([0], 'test', pretrained_path = '../models/xception-b5690688.pth')
    model.model.load_state_dict(torch.load("/content/drive/MyDrive/models/F3/test_6(git_version)/model2.pth"))

    model.model.to(device)

    model.model.eval()

    for dir in os.listdir(r'C:\Users\ethanyi\Desktop\deepfake_project\数据集\FaceSwap_dlib'):
        os.path.join()

# python 3.7
"""Utility functions for logging."""

__all__ = ['setup_logger']

DEFAULT_WORK_DIR = 'results'

def setup_logger(work_dir=None, logfile_name='log.txt', logger_name='logger'):
    """Sets up logger from target work directory.

    The function will sets up a logger with `DEBUG` log level. Two handlers will
    be added to the logger automatically. One is the `sys.stdout` stream, with
    `INFO` log level, which will print improtant messages on the screen. The other
    is used to save all messages to file `$WORK_DIR/$LOGFILE_NAME`. Messages will
    be added time stamp and log level before logged.

    NOTE: If `logfile_name` is empty, the file stream will be skipped. Also,
    `DEFAULT_WORK_DIR` will be used as default work directory.

    Args:
    work_dir: The work directory. All intermediate files will be saved here.
        (default: None)
    logfile_name: Name of the file to save log message. (default: `log.txt`)
    logger_name: Unique name for the logger. (default: `logger`)

    Returns:
    A `logging.Logger` object.

    Raises:
    SystemExit: If the work directory has already existed, of the logger with
        specified name `logger_name` has already existed.
    """

    logger = logging.getLogger(logger_name)
    if logger.hasHandlers():  # Already existed
        raise SystemExit(f'Logger name `{logger_name}` has already been set up!\n'
                            f'Please use another name, or otherwise the messages '
                            f'may be mixed between these two loggers.')

    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s")
    # Print log message with `INFO` level or above onto the screen.
    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setLevel(logging.DEBUG)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    if not logfile_name:
        return logger

    work_dir = work_dir or DEFAULT_WORK_DIR
    logfile_name = os.path.join(work_dir, logfile_name)
    # if os.path.isfile(logfile_name):
    #   print(f'Log file `{logfile_name}` has already existed!')
    #   while True:
    #     decision = input(f'Would you like to overwrite it (Y/N): ')
    #     decision = decision.strip().lower()
    #     if decision == 'n':
    #       raise SystemExit(f'Please specify another one.')
    #     if decision == 'y':
    #       logger.warning(f'Overwriting log file `{logfile_name}`!')
    #       break

    os.makedirs(work_dir, exist_ok=True)

    # Save log message with all levels in log file.
    fh = logging.FileHandler(logfile_name)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger
