# import cv2
# from deepface import DeepFace
# import numpy as np

# face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


# video = cv2.VideoCapture(0)

# while video.isOpened():
#     _,frame = video.read()

#     gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
#     face = face_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5)

#     for x,y,w,h in face:
#         img=cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),1)
#         try:
#             analyze = DeepFace.analyze(frame,actions=['emotion'])
#             print(analyze[0]['dominant_emotion'])
#         except:
#             print("no face")

#     cv2.imshow('video',frame)
#     key = cv2.waitKey(1)
#     if key==ord('q'):
#         break

# video.release()


import cv2
from deepface import DeepFace
import numpy as np
import argparse

# 初始化人脸检测器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# 设置参数解析器
parser = argparse.ArgumentParser(description="Emotion detection in video.")
parser.add_argument("--video", type=str, help="Path to the video file. If not provided, the webcam will be used.")
args = parser.parse_args()

# 根据参数决定使用视频文件还是摄像头
if args.video:
    video = cv2.VideoCapture(args.video)
else:
    video = cv2.VideoCapture(0)

while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        img = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
        try:
            analyze = DeepFace.analyze(frame, actions=['emotion'])
            print(analyze[0]['dominant_emotion'])
        except Exception as e:
            print(f"No face detected: {e}")

    cv2.imshow('video', frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
