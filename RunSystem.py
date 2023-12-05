import cv2
import mediapipe as mp
import numpy as np
import time
import  HandTrackingModule as htm
from ctypes import  cast, POINTER
from  comtypes import  CLSCTX_ALL
from  pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import pickle
import copy
import itertools

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
labels_dict = {0: '0', 1: '1', 2: '2'}

results = 0

#调用Mediapipe手势识别工具
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.7)

  
mode = 0
fetch = 0

photo_count = 0

last_capture_time = time.time()

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
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

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
            
            # 检测并绘制手的关键点
            frame = detector.findHands(frame, draw=True)
    
        rx1 = int(min(x_) * W) - 10
        ry1 = int(min(y_) * H) - 10

        rx2 = int(max(x_) * W) - 10
        ry2 = int(max(y_) * H) - 10

        prediction = model.predict([np.array(pre_processed_landmark_list)])
        
        confidence_scores = model.decision_function([np.array(pre_processed_landmark_list)])
        
        predicted_character = labels_dict[int(prediction[0])]
        #print(predicted_character)

        if predicted_character in ['0', '1', '2'] and len(faces) > 0:
            for (x, y, w, h) in faces:
                cv2.imwrite(f'photos/photo_{photo_count}.jpg', frame)       
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                photo_count += 1
                cv2.putText(frame, f"Photo Count: {photo_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                      

        

    cv2.imshow('PICS capture', frame)
    key = cv2.waitKey(1)
    # 如果按下 'q' 键，退出循环
    if key == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()