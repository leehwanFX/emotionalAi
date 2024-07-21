import warnings
warnings.filterwarnings('ignore')

# 데이터 확인
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Dataset 만들기
import keras
from keras.utils import to_categorical

# Detect Face
import cv2
from scipy.ndimage import zoom
from scipy.fftpack import fft2, ifft2

# Model
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import BatchNormalization
from keras.models import Model

import tensorflow as tf

shape_x = 48
shape_y = 48

# 전체 이미지에서 얼굴을 찾아내는 함수
def detect_face_hybrid(frame):
    # Haar Cascade를 사용한 얼굴 감지
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detected_faces = face_cascade.detectMultiScale(gray,
                                                   scaleFactor=1.1,
                                                   minNeighbors=6,
                                                   minSize=(48, 48),
                                                   flags=cv2.CASCADE_SCALE_IMAGE)
    
    coord = []
    for x, y, w, h in detected_faces:
        if w > 100:
            coord.append([x, y, w, h])
    
    return gray, detected_faces, coord

# 전체 이미지에서 찾아낸 얼굴을 추출하는 함수
def extract_face_features_hybrid(gray, coord, shape_x=48, shape_y=48):
    new_face = []
    for x, y, w, h in coord:
        # 얼굴 영역 추출
        face = gray[y:y+h, x:x+w]
        
        # 2D 푸리에 변환 적용
        f_transform = fft2(face)
        f_shift = np.fft.fftshift(f_transform)
        
        # 고주파 성분 제거 (저역 통과 필터)
        rows, cols = face.shape
        crow, ccol = rows//2, cols//2
        mask = np.zeros((rows, cols), np.uint8)
        mask[crow-15:crow+15, ccol-15:ccol+15] = 1
        f_shift_filtered = f_shift * mask
        
        # 역 푸리에 변환
        f_ishift = np.fft.ifftshift(f_shift_filtered)
        face_filtered = np.abs(ifft2(f_ishift))
        
        # 크기 조정
        face_resized = cv2.resize(face_filtered, (shape_x, shape_y))
        
        # 정규화
        face_normalized = cv2.normalize(face_resized, None, 0, 1, cv2.NORM_MINMAX)
        
        new_face.append(face_normalized)
    
    return new_face

# 예측 결과를 출력하는 함수 추가
def print_prediction(pred):
    emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    print("\nCurrent Emotion Predictions:")
    for emotion, probability in zip(emotions, pred[0]):
        print(f"{emotion}: {probability:.3f}")
    print(f"Predicted Emotion: {emotions[np.argmax(pred)]}")

# 모델 불러오기
model = keras.models.load_model('./face-emotion-model/model.h5')

# 인덱스번호로 웹캠연결 대부분 시스템적으로 0번부터 부여됨
video_capture = cv2.VideoCapture(0)

# 프레임 단위로 영상 캡쳐
while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to grab frame")
        break
    # ret: 비디오를 성공적으로 읽어왔는지 확인 True/False
    # frame: 각 픽셀의 색상을 포함한 프레임 정보 Numpy
    
    face_index = 0
    gray, detected_faces, coord = detect_face_hybrid(frame)
    
    try:
        face_zoom = extract_face_features_hybrid(gray, coord)
        face_zoom = np.reshape(face_zoom[0].flatten(), (1, 48, 48, 1))
        x, y, w, h = coord[face_index]
        
        # 머리 둘레에 직사각형 그리기: (0, 255, 0)을 통해 녹색으로 선두께는 2
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # 감정 예측
        pred = model.predict(face_zoom)
        pred_result = np.argmax(pred)
        
        # 각 라벨별 예측 정도 표시
        cv2.putText(frame,                                   # 텍스트를 표시할 프레임
                    "Angry: " + str(round(pred[0][0], 3)),   # 텍스트 표시 "감정: 예측 probablity", 소수점 아래 3자리
                    (10, 50),                                # 텍스트 위치
                    cv2.FONT_HERSHEY_SIMPLEX,                # 폰트 종류
                    1,                                       # 폰트 사이즈
                    (0, 0, 255),                             # 폰트 색상
                    2                                        # 폰트 두께
                   )
        cv2.putText(frame, "Disgust: " + str(round(pred[0][1], 3)), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, "Fear: " + str(round(pred[0][2], 3)), (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, "Happy: " + str(round(pred[0][3], 3)), (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, "Sad: " + str(round(pred[0][4], 3)), (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, "Surprise: " + str(round(pred[0][5], 3)), (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, "Neutral: " + str(round(pred[0][6], 3)), (10, 290), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # 예측값이 높은 라벨 하나만 프레임 옆에 표시
        if pred_result == 0:
            cv2.putText(frame, "Angry " + str(round(pred[0][0], 2)), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5)
        elif pred_result == 1:
            cv2.putText(frame, "Disgust " + str(round(pred[0][1], 2)), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5)
        elif pred_result == 2:
            cv2.putText(frame, "Fear " + str(round(pred[0][2], 2)), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5)
        elif pred_result == 3:
            cv2.putText(frame, "Happy " + str(round(pred[0][3], 2)), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5)
        elif pred_result == 4:
            cv2.putText(frame, "Sad " + str(round(pred[0][4], 2)), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5)
        elif pred_result == 5:
            cv2.putText(frame, "Surprise " + str(round(pred[0][5], 2)), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5)
        else:
            cv2.putText(frame, "Neutral " + str(round(pred[0][6], 2)), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5)
    
    except:
        continue
        
    # 결과 표시
    cv2.imshow('Video', frame)
    
    # 키 입력 확인
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('e'):
        if 'pred' in locals():
            print_prediction(pred)
        else:
            print("No prediction available yet.")

        
# 웹캠 해지
video_capture.release()

# 창 닫기: 창이 안닫히는 경우 쥬피터 닫기
cv2.destroyAllWindows()