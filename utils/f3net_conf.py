import torch
import os

###########################
# 基本配置
###########################

#在确保所有gpu可用的前提下，可设置多个gpu，否则torch.cuda.is_available()显示为false
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
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
normal_root = r"E:\Face_Dataset\celeb-Df\deepware(380_380)\celeb_real(380_380)"
malicious_root = r"E:\Face_Dataset\celeb-Df\deepware(380_380)\celeb_syn(380_380)"
csv_root = r"C:\Users\ethanyi\Desktop\security_competition\csv"

model_path_name = r"G:\我的云端硬盘\models\F3\test_6(git_version)\model2.pth"
xception_pretrained_path =r"D:\DeepFakeProject_in_D\deepfake_project\our_code\f3net\models\xception-b5690688.pth"
efficient_pretrained_path=r"D:\DeepFakeProject_in_D\deepfake_project\eliminate_project\ycy\secruity-eye\screen_shot\face_recognition\deepware.pt"



