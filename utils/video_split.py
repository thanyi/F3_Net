import os

import cv2
'''
这是opencv中用于给视频切片的py文件,保存的文件名是以
视频数-视频被抽帧的图片序号-本帧图片的人脸序号
'''
def normal_split():
    videodir = r'F:\dataset\faceforensics++\original_sequences\youtube\c23\videos'  # 视频文件路径
    # path = r'F:\dataset\faceforensics++\original_sequences\youtube\c23\our_imgs_face'  # 存储视频的子目录
    path = r'C:\Users\ethanyi\Desktop\deepfake_project\our_code\f3net\data\normal_our_imgs'  # 存储视频的子目录

    # video_path = r"C:\Users\Y\Desktop\test\temp.mp4"
    # save_path = r"C:\Users\Y\Desktop\test\out"
    classifier_path = r"C:\Users\ethanyi\anaconda3\envs\pytorch\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml"
    videonames = sorted(os.listdir(videodir))

    for videoname in videonames[240:360]:
        videoname = os.path.join(videodir, videoname)
        img_dir = os.path.join(path, videoname.split('\\')[-1].split('.')[0])
        vc = cv2.VideoCapture(videoname)

        faceCascade = cv2.CascadeClassifier(classifier_path)
        if not os.path.exists(img_dir):  # 如果不存在就创建文件夹
            os.mkdir(img_dir)
        if vc.isOpened():
            rval, frame = vc.read()
        else:
            rval = False
        timeF = 10  # 帧数间隔
        image_count = 1  # 图片计数
        frame_count = 1  # 帧数计数
        while rval:
            rval, frame = vc.read()  # 分帧读取视频
            if not rval:
                break
            if frame_count % timeF == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                locations = faceCascade.detectMultiScale(gray, 1.3, 5)
                face_count = 0
                for (x, y, w, h) in locations:
                    face_count += 1
                    # img = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    save_img = frame[int(y):int(y + h), int(x):int(x + w)]
                    save_img = cv2.resize(save_img, (299, 299))
                    cv2.imwrite(img_dir + '\\' + img_dir.split('\\')[-1] + '-' + str(image_count) + '-' + str(
                        face_count) + '.png', save_img)  # 保存路径
                    # cv2.imshow('result', img)
                image_count += 1
            frame_count += 1

        print(videoname.split('\\')[-1].split('.')[0]+"完成!")

        vc.release()


def malicious_split():
    videodir = r'F:\dataset\faceforensics++\manipulated_sequences\Deepfakes\c23\videos'  # 视频文件路径
    # path = r'F:\dataset\faceforensics++\manipulated_sequences\Deepfakes\c23\our_imgs_face'  # 存储视频的子目录
    path = r'C:\Users\ethanyi\Desktop\deepfake_project\our_code\f3net\data\malicious_our_imgs'  # 存储视频的子目录

    classifier_path = r"C:\Users\ethanyi\anaconda3\envs\pytorch\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml"
    videonames = sorted(os.listdir(videodir))

    for videoname in videonames[240:360]:
        videoname = os.path.join(videodir, videoname)
        img_dir = os.path.join(path, videoname.split('\\')[-1].split('.')[0])
        vc = cv2.VideoCapture(videoname)

        faceCascade = cv2.CascadeClassifier(classifier_path)
        if not os.path.exists(img_dir):  # 如果不存在就创建文件夹
            os.mkdir(img_dir)
        if vc.isOpened():
            rval, frame = vc.read()
        else:
            rval = False
        timeF = 10  # 帧数间隔
        image_count = 1  # 图片计数
        frame_count = 1  # 帧数计数
        while rval:
            rval, frame = vc.read()  # 分帧读取视频
            if not rval:
                break
            if frame_count % timeF == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                locations = faceCascade.detectMultiScale(gray, 1.3, 5)
                face_count = 0
                for (x, y, w, h) in locations:
                    face_count += 1
                    # img = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    save_img = frame[int(y):int(y + h), int(x):int(x + w)]
                    save_img = cv2.resize(save_img, (299, 299))
                    cv2.imwrite(img_dir + '\\' + img_dir.split('\\')[-1] + '-' + str(image_count) + '-' + str(
                        face_count) + '.png', save_img)  # 保存路径
                    # cv2.imshow('result', img)
                image_count += 1
            frame_count += 1

        print(videoname.split('\\')[-1].split('.')[0] + "完成!")
        vc.release()

if __name__ == '__main__':
    malicious_split()
