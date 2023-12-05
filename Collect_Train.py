import os
import cv2
import csv
import mediapipe as mp
import copy
import itertools
import numpy as np
import pickle
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pycaw.pycaw as pycaw

#创建imgs文件夹
DATA_DIR = './imgs'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
    
#设置手势类别数量
number_of_classes = 3

#设置每个类别样本数
dataset_size = 100



#收集自定义微数据集并写入imgs文件夹
cap = cv2.VideoCapture(0)
for j in range(number_of_classes):
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))
    print('Collecting data for class {}'.format(j))
    done = False
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, 'Ready to pose? Press "R" !', (100, 50), cv2.FONT_ITALIC, 0.9, (0, 255, 0), 3,
                    cv2.LINE_AA)
        cv2.putText(frame, 'Remember to twist and turn a little', (80, 90), cv2.FONT_ITALIC, 0.8, (255, 25, 0), 3,
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('r'):
            break
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)), frame)
        counter += 1
cap.release()
cv2.destroyAllWindows()




#将收集到的数据集中提取骨架并归一化整理到DATASET.csv文件中
mp_hands = mp.solutions.hands.Hands()

#计算手部框
def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_array = np.empty((0, 2), int)
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point = [np.array((landmark_x, landmark_y))]
        landmark_array = np.append(landmark_array, landmark_point, axis=0)
    x, y, w, h = cv2.boundingRect(landmark_array)
    return [x, y, x + w, y + h]

#骨架坐标映射到像素点
def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])
    return landmark_point

#归一化骨架坐标
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
    max_value = max(list(map(abs, temp_landmark_list)))
    def normalize_(n):
        return n / max_value
    temp_landmark_list = list(map(normalize_, temp_landmark_list))
    return temp_landmark_list
def logging_csv(number, mode, landmark_list):
    if mode == 1 and (0 <= number <= 9):
        csv_path = './DATASET.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
    return

# 获取当前目录下的子文件夹
current_dir = os.getcwd()+'./imgs/'
directories = [d for d in os.listdir(current_dir) if os.path.isdir(os.path.join(current_dir, d))]
# 创建CSV文件DATASET.csv
with open("DATASET.csv", mode="w", newline="") as csv_file:
    csv_writer = csv.writer(csv_file)
# 循环遍历每个子文件夹中的照片并处理
for i, directory in enumerate(sorted(directories)):
    for filename in os.listdir('./imgs/'+directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image = cv2.imread(os.path.join('./imgs/'+directory, filename))            
            # 获取Landmark坐标并保存到CSV文件中
            results = mp_hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))            
            with open("DATASET.csv", mode="a", newline="") as csv_file:
                csv_writer = csv.writer(csv_file)
                if results.multi_hand_landmarks:
                    for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                    results.multi_handedness):
                        brect = calc_bounding_rect(image, hand_landmarks)
                        # 计算归一化坐标点
                        landmark_list = calc_landmark_list(image, hand_landmarks)
                        pre_processed_landmark_list = pre_process_landmark(
                            landmark_list)
                        #写入csv文件
                        logging_csv(i, 1, pre_processed_landmark_list)

mp_hands.close()



#根据DATASET.csv文件训练一个SVM模型，并导出为model.p文件

#读入数据集与标签
dataset = 'DATASET.csv'
data = np.loadtxt(dataset, delimiter=',', dtype='float32', usecols=list(range(1, (21 * 2) + 1)))
labels = np.loadtxt(dataset, delimiter=',', dtype='int32', usecols=(0))

#将数据集打乱并分为测试集与验证集
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

#创建SVM模型
model = SVC(kernel='linear', C=1, random_state=42)
model.fit(x_train, y_train)

#验证模型性能
y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)
print('{}% of samples were classified correctly!'.format(score * 100))

#将模型导出
f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()