import cv2
import dlib

if __name__ == '__main__':
    video_path = r"C:\Users\Y\Desktop\test\011.mp4"
    video_name = "011"
    save_path = r"C:\Users\Y\Desktop\test\out"
    dlib_classifier_path = r"C:\Users\Y\Desktop\test\shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(dlib_classifier_path)

    vc = cv2.VideoCapture(video_path)
    if vc.isOpened():
        rval, frame = vc.read()
    else:
        rval = False
    timeF = 30  # 帧数间隔
    image_count = 1  # 图片计数
    frame_count = 1  # 帧数计数
    face_id = 0
    face_tracker = {}
    location_pre = {}
    while rval:
        rval, frame = vc.read()  # 分帧读取视频
        if not rval:
            break
        if frame_count % timeF == 0:
            for face_id in face_tracker.keys():
                trackingQuality = face_tracker[face_id].update(frame)
                if trackingQuality < 7:
                    face_tracker.pop(face_id, None)

            img = frame.copy()
            dets = detector(frame, 1)
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
                        save_img = cv2.resize(save_img, (299, 299))
                        cv2.imwrite(save_path + '\\' + video_name + '-' + str(image_count) + '-' + str(faceID) + '.jpg',
                                    save_img)  # 保存路径
                    location_pre[faceID] = shape
                image_count += 1
        frame_count += 1
    vc.release()
