import cv2
import learn
import mediapipe as mp
import numpy as np
import time

from pycaw.api.endpointvolume import IAudioEndpointVolume

import  HandTrackingModule as htm
from ctypes import  cast, POINTER
from  comtypes import  CLSCTX_ALL
# from  pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import pickle
import copy
import itertools

from pycaw.pycaw import AudioUtilities, ISimpleAudioVolume

'''
# 获取默认音频渲染设备
devices = AudioUtilities.GetSpeakers()
# 激活音频终端音量接口
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
# 转换为指针类型
volume = cast(interface, POINTER(IAudioEndpointVolume))
# 获取音量范围
volume_range = volume.GetVolumeRange()


def vol_tansfer(x):
    dict = {0: -65.25, 1: -56.99, 2: -51.67, 3: -47.74, 4: -44.62, 5: -42.03, 6: -39.82, 7: -37.89, 8: -36.17,
            9: -34.63, 10: -33.24,
            11: -31.96, 12: -30.78, 13: -29.68, 14: -28.66, 15: -27.7, 16: -26.8, 17: -25.95, 18: -25.15, 19: -24.38,
            20: -23.65,
            21: -22.96, 22: -22.3, 23: -21.66, 24: -21.05, 25: -20.46, 26: -19.9, 27: -19.35, 28: -18.82, 29: -18.32,
            30: -17.82,
            31: -17.35, 32: -16.88, 33: -16.44, 34: -16.0, 35: -15.58, 36: -15.16, 37: -14.76, 38: -14.37, 39: -13.99,
            40: -13.62,
            41: -13.26, 42: -12.9, 43: -12.56, 44: -12.22, 45: -11.89, 46: -11.56, 47: -11.24, 48: -10.93, 49: -10.63,
            50: -10.33,
            51: -10.04, 52: -9.75, 53: -9.47, 54: -9.19, 55: -8.92, 56: -8.65, 57: -8.39, 58: -8.13, 59: -7.88,
            60: -7.63,
            61: -7.38, 62: -7.14, 63: -6.9, 64: -6.67, 65: -6.44, 66: -6.21, 67: -5.99, 68: -5.76, 69: -5.55, 70: -5.33,
            71: -5.12, 72: -4.91, 73: -4.71, 74: -4.5, 75: -4.3, 76: -4.11, 77: -3.91, 78: -3.72, 79: -3.53, 80: -3.34,
            81: -3.15, 82: -2.97, 83: -2.79, 84: -2.61, 85: -2.43, 86: -2.26, 87: -2.09, 88: -1.91, 89: -1.75,
            90: -1.58,
            91: -1.41, 92: -1.25, 93: -1.09, 94: -0.93, 95: -0.77, 96: -0.61, 97: -0.46, 98: -0.3, 99: -0.15, 100: 0.0}
    return dict[x]

# 设置声音
vo = 20
volume.SetMasterVolumeLevel(vol_tansfer(vo), None)
'''

#设置摄像头画面大小
WCam = 1280
HCam = 640

#获取摄像头，参数因设备而异同，默认为1，若苹果电脑为2
cap = cv2.VideoCapture(0)
cap.set(3, WCam)
cap.set(4, HCam)

# 加载Haar级联分类器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

pTime = 0

detector = htm.handDetector(detectionCon=0.7)

#读入训练好的model.p文件
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

#骨架坐标映射为像素点
def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point

#骨架坐标归一化
def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # 归一化
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list

#分类标签
labels_dict = {0: '0', 1: '1', 2: '2', 3:'3'}

results = 0

#调用Mediapipe手势识别工具
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.7)


photo_count = 0

wait = 0
pre_time = 0

while True:
    # 读取摄像头视频帧
    success, frame = cap.read()
    cv2.putText(frame, "Pose", (440, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    # 将图像转为灰度
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 检测人脸
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    x_ = []
    y_ = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            landmark_list = calc_landmark_list(frame, hand_landmarks)

            # 归一化
            pre_processed_landmark_list = pre_process_landmark(
                landmark_list)
            
            # 检测手的关键点，draw=False
            frame = detector.findHands(frame, draw=False)
    
        rx1 = int(min(x_) * W) - 10
        ry1 = int(min(y_) * H) - 10

        rx2 = int(max(x_) * W) - 10
        ry2 = int(max(y_) * H) - 10

        prediction = model.predict([np.array(pre_processed_landmark_list)])
        
        confidence_scores = model.decision_function([np.array(pre_processed_landmark_list)])
        
        predicted_character = labels_dict[int(prediction[0])]
        #print(predicted_character)

        # 相邻两张照片时间间隔至少为0.5秒
        if wait==1 :
            tm = time.time()
            derta = tm-pre_time
            if derta >= 0.5:
                wait = 0

        if wait==0 and predicted_character in ['0', '1', '2'] and len(faces) > 0:
            for (x, y, w, h) in faces:
                cv2.imwrite(f'photos/photo_{photo_count}.jpg', frame)       
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                photo_count += 1
                '''
                if (predicted_character=='0') and (vo>=1):
                    vo -= 1
                elif (predicted_character=='2') and (vo<=99):
                    vo += 1
                '''
                cv2.putText(frame, f"Photo Count: {photo_count}, pose: {predicted_character}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                wait = 1
                pre_time = time.time()
                # volume.SetMasterVolumeLevel(vol_tansfer(vo), None)

        

    cv2.imshow('PICS capture', frame)
    key = cv2.waitKey(1)
    # 如果按下 'q' 键，退出循环
    if key == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()