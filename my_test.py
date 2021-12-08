import os
import torch
from sklearn.metrics import roc_curve
from torch import nn
from torch.utils.data import DataLoader

from torchvision import transforms
from PIL import Image
import os


os.environ['CUDA_VISIBLE_DEVICES'] = '0'   #在确保所有gpu可用的前提下，可设置多个gpu，否则torch.cuda.is_availabel()显示为false
osenvs = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
gpu_ids = [*range(osenvs)]
max_epoch = 5
loss_freq = 40
mode = 'FAD' # ['Original', 'FAD', 'LFS', 'Both', 'Mix']
pretrained_path = 'xception-b5690688.pth'
device = torch.device("cuda")

normal_root = r"/content/data/normal_our_imgs"
malicious_root = r"/content/data/malicious_our_imgs"
csv_root = r"/content/data/csv"

train_data = DeepfakeDataset(normal_root=normal_root, malicious_root=malicious_root, mode='train', resize=299,
                             csv_root=csv_root)
val_data = DeepfakeDataset(normal_root=normal_root, malicious_root=malicious_root, mode='val', resize=299,
                           csv_root=csv_root)

train_data_size = len(train_data)
val_data_size = len(val_data)

print('train_data_size:', train_data_size)
print('val_data_size:', val_data_size)

train_loader = DataLoader(train_data, 16, shuffle=True)
val_loader = DataLoader(val_data, 16, shuffle=True)



if __name__ == '__main__':

    # model = Trainer(gpu_ids, mode, pretrained_path)
    model = torch.load("/content/drive/MyDrive/models/F3/test_4/model_3.pth")
    model.model.to(device)
    # img_path = "test_img/"
    # imgs =[]
    # for root,dirs,file_names in os.walk(img_path):
    #     for file_name in file_names:
    #         if ".png" in file_name:
    #             imgs.append(os.path.join(root,file_name))

    # tf = transforms.Compose([
    #     lambda x: Image.open(x).convert("RGB"),  # string path => image data
    #     # transforms.Resize((int(self.resize), int(self.resize))),
    #     # transforms.Resize((int(299 * 1.25), int(299 * 1.25))),
    #     transforms.RandomRotation(30),
    #     transforms.CenterCrop(299),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                          std=[0.229, 0.224, 0.225])
    # ])
    datas,labels = iter(val_loader).next()

    normal = 0
    malicious = 0
    y_true, y_pred = [], []

    output = model.forward(datas.to(device))
    y_pred.extend(output.sigmoid().flatten().tolist())
    y_true.extend(labels.flatten().tolist())
    # if pred>0.5:
    #     malicious+=1
    #     print("This is a malicious picture!")
    # else:
    #     normal+=1
    #     print("This is a normal picture!")
    print(y_pred)
    print(y_true)