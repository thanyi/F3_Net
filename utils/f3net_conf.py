import torch
import os

###########################
# 基本配置
###########################

#在确保所有gpu可用的前提下，可设置多个gpu，否则torch.cuda.is_available()显示为false
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
osenvs = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
gpu_ids = [*range(osenvs)]
max_epoch = 5
loss_freq = 40
mode = 'Both'
# pretrained_path = 'models/xception-b5690688.pth'
device = torch.device("cuda")
resize = 380

###########################
#  路径
###########################
normal_root = r"/home/jiahaozhong/dataset/celeb-Df/celeb_real"
malicious_root = r"/home/jiahaozhong/dataset/celeb-Df/celeb_syn"
csv_root = r"/home/jiahaozhong/dataset/celeb-Df/csv"


xception_pretrained_path =r""
efficient_pretrained_path=r"/home/jiahaozhong/model/f3net/deepware.pt"

dfdc_root = r"/home/jiahaozhong/dataset/Dfdc/Dfdc_real"
dfdc_syn_root = r"/home/jiahaozhong/dataset/Dfdc/Dfdc_syn"
dfdc_csv_root = r"/home/jiahaozhong/dataset/Dfdc/csv"

