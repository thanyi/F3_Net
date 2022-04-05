'''
处理DFDC的json格式文件
    将real和fake分开为两个文件夹之内的东西
'''

import json
import shutil

import pandas as pd
import os

img_path = r'E:\Face_Dataset\DFDC\img_dfdc\380_380\dfdc_train_part_00'
real_path = r"E:\Face_Dataset\DFDC\img_dfdc\380_380\real"
fake_path = r"E:\Face_Dataset\DFDC\img_dfdc\380_380\fake"





with open(r"C:\Users\ethanyi\Desktop\security_competition\metadata.json") as f:
    json_dict  = json.load(f)


video_dict ={}

for k,v in json_dict.items():

    video_dict[k] = v['label']



img_files = sorted(os.listdir(img_path))
print("开始进行复制整理。。。")
for i,file in enumerate(img_files[1000:]):
    print(str(i)+":"+file+' is proceeding...')
    if file+'.mp4' in video_dict:

        if video_dict[file+'.mp4'] == 'FAKE':

            source_path = img_path+'\\'+file
            target_path = fake_path+'\\'+file

            if os.path.exists(target_path):
                # 如果目标路径存在文件夹的话就删除
                shutil.rmtree(target_path)

            shutil.copytree(source_path, target_path)
        else:
            source_path = img_path +'\\' + file
            target_path = real_path +'\\'+ file

            if os.path.exists(target_path):
                # 如果目标路径存在文件夹的话就删除
                shutil.rmtree(target_path)

            shutil.copytree(source_path, target_path)

