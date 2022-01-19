import cv2
import dlib
import os
'''
这个模块使用dlib进行人脸识别和切脸
使用了一个dlib模块的模型，位置在本文件夹中的 shape_predictor_68_face_landmarks.dat 文件
'''

if __name__ == '__main__':
    videodir = r'F:\dataset\DFDC\video_dfdc\dfdc_train_part_00\dfdc_train_part_0'
    path = r'F:\dataset\DFDC\img_dfdc\dfdc_train_part_00'
    dlib_classifier_path = r"shape_predictor_68_face_landmarks.dat"
    videonames = sorted(os.listdir(videodir))
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(dlib_classifier_path)
    for videoname in videonames[201:250]:
        videoname = os.path.join(videodir, videoname)
        img_dir = os.path.join(path, videoname.split('\\')[-1].split('.')[0])
        vc = cv2.VideoCapture(videoname)
        if not os.path.exists(img_dir):  # 如果不存在就创建文件夹
            print(img_dir)
            os.mkdir(img_dir)

        if vc.isOpened():
            rval, frame = vc.read()
        else:
            rval = False
        timeF = 7  # 帧数间隔
        image_count = 1  # 图片计数
        frame_count = 1  # 帧数计数
        face_id = 0
        face_tracker = {}
        location_pre = {}

        while rval:
            rval, frame = vc.read()  # 分帧读取视频
            if not rval:
                break
            faceIDtoDelete = []
            for faceID in face_tracker.keys():
                trackingQuality = face_tracker[faceID].update(img)
                if trackingQuality < 7:
                    faceIDtoDelete.append(faceID)
            for faceID in faceIDtoDelete:
                face_tracker.pop(faceID, None)

            if frame_count % timeF == 0:
                img = frame
                dets = detector(img, 1)
                test = 1
                for k, d in enumerate(dets):
                    x = d.left()
                    y = d.top()
                    w = d.width()
                    h = d.height()

                    x_bar = d.left() + 0.5 * d.width()
                    y_bar = d.top() + 0.5 * d.height()

                    matched_face_id = None
                    for faceID in face_tracker.keys():
                        trackedPosition = face_tracker[faceID].get_position()

                        t_x = int(trackedPosition.left())
                        t_y = int(trackedPosition.top())
                        t_w = int(trackedPosition.width())
                        t_h = int(trackedPosition.height())

                        t_x_bar = t_x + 0.5 * t_w
                        t_y_bar = t_y + 0.5 * t_h

                        if ((t_x <= x_bar <= (t_x + t_w)) and (t_y <= y_bar <= (t_y + t_h)) and (
                                x <= t_x_bar <= (x + w)) and (y <= t_y_bar <= (y + h))):
                            matched_face_id = faceID

                    if matched_face_id is None:
                        tracker = dlib.correlation_tracker()
                        tracker.start_track(img, dlib.rectangle(x, y, x + w, y + h))
                        face_tracker[face_id] = tracker
                        location_pre[face_id] = predictor(img, d)
                        face_id += 1

                if frame_count != timeF:
                    for faceID in face_tracker.keys():
                        trackedPosition = face_tracker[faceID].get_position()
                        shape = predictor(img, dlib.rectangle(trackedPosition))

                        t_x = int(trackedPosition.left())
                        t_y = int(trackedPosition.top())
                        t_w = int(trackedPosition.width())
                        t_h = int(trackedPosition.height())
                        same_count = 0

                        for p_pt, n_pt in zip(location_pre[faceID].parts(), shape.parts()):
                            if p_pt.x == n_pt.x and p_pt.y == n_pt.y:
                                same_count += 1
                        if same_count < 10:
                            save_img = img[int(t_y):int(t_y + t_h), int(t_x):int(t_x + t_w)]
                            try:
                                save_img = cv2.resize(save_img, (299, 299))
                                cv2.imwrite(img_dir+'\\'+img_dir.split('\\')[-1] + '-'+ str(image_count) + '-' + str(faceID) + '.png',
                                    save_img)  # 保存路径
                            except Exception:
                                pass
                        location_pre[faceID] = shape
                    image_count += 1
            frame_count += 1
        vc.release()
