import torch



# 路径
normal_root = r"/content/data/normal_dlib"
malicious_root = r"/content/data/Deepfakes_dlib"
csv_root = r"/content/data/csv"

malicious_root_deepfake = r"/content/data/Deepfakes_dlib"
malicious_root_deepfake_compress = r"/content/data/DeepFake++compress_dlib"
malicious_root_faceswap = r"/content/data/FaceSwap_dlib"

malicious_root_dfdc = r"/content/data/FaceSwap_dlib"
malicious_root_celeb = r"/content/data/FaceSwap_dlib"



xception_pretained_path = r"D:\DeepFakeProject_in_D\deepfake_project\our_code\f3net\kaggle_dfdc_model\xception-hg-2.pth"
wsdan_pretained_path = r"D:\DeepFakeProject_in_D\deepfake_project\our_code\f3net\kaggle_dfdc_model\ckpt_x.pth"

# 其他设置
device = torch.device("cuda")
resize = 299