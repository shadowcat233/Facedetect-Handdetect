import cv2

# 加载Haar级联分类器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 设置摄像头
cap = cv2.VideoCapture(0)

photo_count = 0

while True:
    # 读取摄像头视频帧
    ret, frame = cap.read()

    # 将图像转为灰度
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 检测人脸
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5)

    # 如果检测到人脸，显示矩形框，并拍照
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            cv2.imwrite(f'photos/photo_{photo_count}.jpg', frame)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            photo_count += 1
            cv2.putText(frame, f"Photo Count: {photo_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            

    # 显示结果
    cv2.imshow('Face Recognition', frame)

    # 检测按键，如果按下 'q' 键，退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头资源
cap.release()

# 关闭窗口
cv2.destroyAllWindows()
