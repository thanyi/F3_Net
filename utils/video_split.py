import cv2
import dlib
import os
import datetime
import time
import threading

dlib_classifier_path = "shape_predictor_68_face_landmarks.dat"  # 人脸识别模型路径
LOCAL_VIDEO = 0
CAMERA_STREAM = 1


class dealImgThread(threading.Thread):
    image_count = 0
    face_details = []
    details_lock = threading.Lock()
    count_lock = threading.Lock()

    def __init__(self, frame, outPath, detector, predictor, timeF, frame_count):
        threading.Thread.__init__(self)
        self.frame = frame
        self.outPath = outPath
        self.detector = detector
        self.predictor = predictor
        self.timeF = timeF
        self.frame_count = frame_count

    def run(self):
        dots = self.detector(self.frame, 1)
        backup = dealImgThread.face_details[:]
        for k, d in enumerate(dots):
            shape = self.predictor(self.frame, d)
            # 排除静态人脸,如画像
            isSame = False
            for i in range(len(backup)):
                same_count = 0
                for p_pt, n_pt in zip(backup[i].parts(), shape.parts()):
                    if p_pt.x == n_pt.x and p_pt.y == n_pt.y:
                        same_count += 1
                if same_count >= 10:
                    isSame = True
                    break
            if self.frame_count == self.timeF:
                dealImgThread.details_lock.acquire()
                dealImgThread.face_details.append(shape)
                dealImgThread.details_lock.release()

            if not isSame and self.frame_count != self.timeF:
                height, width = self.frame.shape[:2]

                multiple = max(d.height(), d.width()) / 1.9
                need_adjust_height = 0
                need_adjust_width = 0
                difference = abs(d.height() - d.width())
                if d.height() > d.width():
                    need_adjust_width = 1
                else:
                    need_adjust_height = 1

                height_overflow = False
                width_overflow = False
                if height < width and d.height() + multiple * 2 + need_adjust_height * difference > height:
                    height_overflow = True

                if width < height and d.width() + multiple * 2 + need_adjust_width * difference > width:
                    width_overflow = True

                if width_overflow:
                    backup = multiple
                    multiple = (height - d.width()) / 2
                    need_adjust_height = 0
                top_cross = max(0 - d.top() + multiple + need_adjust_height * difference / 2, 0)
                bottom_cross = max(d.bottom() + multiple + need_adjust_height * difference / 2 - height, 0)
                top = max(d.top() - multiple - need_adjust_height * difference / 2 - bottom_cross, 0)
                bottom = min(d.bottom() + multiple + need_adjust_height * difference / 2 + top_cross, height)
                if width_overflow:
                    multiple = backup
                if height_overflow:
                    multiple = (height - d.width()) / 2
                    need_adjust_width = 0
                left_cross = max(0 - d.left() + multiple + need_adjust_width * difference / 2, 0)
                right_cross = max(d.right() + multiple + need_adjust_width * difference / 2 - width, 0)
                left = max(d.left() - multiple - need_adjust_width * difference / 2 - right_cross, 0)
                right = min(d.right() + multiple + need_adjust_width * difference / 2 + left_cross, width)
                # print(bottom - top, right - left)
                save_img = self.frame[int(top):int(bottom), int(left):int(right)]
                try:
                    save_img = cv2.resize(save_img, (380, 380))
                    dealImgThread.count_lock.acquire()
                    dealImgThread.image_count += 1
                    cv2.imwrite(self.outPath + "/" + str(dealImgThread.image_count) + '.png', save_img)
                    dealImgThread.count_lock.release()
                except Exception:
                    pass


def setModelPath(path):
    global dlib_classifier_path
    dlib_classifier_path = path


def loadModel():
    global dlib_classifier_path
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(dlib_classifier_path)
    return detector, predictor


def recognition(vc, outPath, detector, predictor, timeF, mode, timeout=60, total=100, multiThread=False):
    # 加载模型
    if mode:
        begin_time = time.time()
    if not os.path.exists(outPath):  # 如果不存在就创建文件夹
        os.mkdir(outPath)
    if vc.isOpened():
        res = True
    else:
        res = False
    image_count = 0  # 图片计数
    frame_count = 0  # 帧数计数
    face_details = []  # 辅助排除静态人脸
    while res:
        if mode:
            now_time = time.time()
            if now_time - begin_time > timeout:
                break
        res, frame = vc.read()  # 分帧读取视频
        if not res:
            break
        # cv2.imshow("img", frame)
        frame_count += 1
        if frame_count % timeF == 0:
            if multiThread:
                dealImgThread(frame, outPath, detector, predictor, timeF, frame_count).start()
            else:
                dots = detector(frame, 1)
                backup = face_details[:]
                face_details.clear()
                for k, d in enumerate(dots):
                    shape = predictor(frame, d)

                    # 排除静态人脸,如画像
                    isSame = False
                    for i in range(len(backup)):
                        same_count = 0
                        for p_pt, n_pt in zip(backup[i].parts(), shape.parts()):
                            if p_pt.x == n_pt.x and p_pt.y == n_pt.y:
                                same_count += 1
                        if same_count >= 10:
                            isSame = True
                            break
                    face_details.append(shape)

                    if not isSame and frame_count != timeF:

                        height, width = frame.shape[:2]

                        multiple = max(d.height(), d.width()) / 1.9
                        need_adjust_height = 0
                        need_adjust_width = 0
                        difference = abs(d.height() - d.width())
                        if d.height() > d.width():
                            need_adjust_width = 1
                        else:
                            need_adjust_height = 1

                        height_overflow = False
                        width_overflow = False
                        if height < width and d.height() + multiple * 2 + need_adjust_height * difference > height:
                            height_overflow = True

                        if width < height and d.width() + multiple * 2 + need_adjust_width * difference > width:
                            width_overflow = True

                        if width_overflow:
                            backup = multiple
                            multiple = (height - d.width()) / 2
                            need_adjust_height = 0
                        top_cross = max(0 - d.top() + multiple + need_adjust_height * difference / 2, 0)
                        bottom_cross = max(d.bottom() + multiple + need_adjust_height * difference / 2 - height, 0)
                        top = max(d.top() - multiple - need_adjust_height * difference / 2 - bottom_cross, 0)
                        bottom = min(d.bottom() + multiple + need_adjust_height * difference / 2 + top_cross, height)
                        if width_overflow:
                            multiple = backup
                        if height_overflow:
                            multiple = (height - d.width()) / 2
                            need_adjust_width = 0
                        left_cross = max(0 - d.left() + multiple + need_adjust_width * difference / 2, 0)
                        right_cross = max(d.right() + multiple + need_adjust_width * difference / 2 - width, 0)
                        left = max(d.left() - multiple - need_adjust_width * difference / 2 - right_cross, 0)
                        right = min(d.right() + multiple + need_adjust_width * difference / 2 + left_cross, width)

                        save_img = frame[int(top):int(bottom), int(left):int(right)]
                        # cv2.imshow("save", save_img)
                        try:
                            save_img = cv2.resize(save_img, (380, 380))
                            image_count += 1
                            cv2.imwrite(outPath + "/" + str(image_count) + '.png', save_img)
                        except Exception:
                            pass
        if mode and image_count >= total:
            break
        if multiThread and mode and dealImgThread.image_count >= total:
            break
        # if cv2.waitKey(33) == 27:
        #     break
    vc.release()


def dealLocalVideos(inPath, outPath, timeF=7, multiThread=False):
    # 预处理
    if not os.path.exists(outPath):
        os.mkdir(outPath)

    # 加载模型
    detector, predictor = loadModel()

    # 视频处理
    videos = sorted(os.listdir(inPath))
    start =13
    for video in videos[start:]:

        print(f"第{start}组开始。。。")
        video = os.path.join(inPath, video)
        img_dir = os.path.join(outPath, video.split('\\')[-1].split('.')[0])
        vc = cv2.VideoCapture(video)
        recognition(vc, img_dir, detector, predictor, timeF, LOCAL_VIDEO, multiThread=multiThread)
        start += 1






if __name__ == '__main__':
    in_dir = r'E:\bilibili\video'
    out_dir = r'E:\bilibili\img'  # 处理后图片存放位置
    dealLocalVideos(in_dir, out_dir, multiThread=True)

