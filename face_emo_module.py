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
def detect_face(frame):
    
    # cascade pre-trained 모델 불러오기
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # RGB를 gray scale로 바꾸기
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # cascade 멀티스케일 분류
    detected_faces = face_cascade.detectMultiScale(gray,
                                                   scaleFactor = 1.1,
                                                   minNeighbors = 6,
                                                   minSize = (shape_x, shape_y),
                                                   flags = cv2.CASCADE_SCALE_IMAGE
                                                  )
    
    coord = []
    for x, y, w, h in detected_faces:
        if w > 100:
            sub_img = frame[y:y+h, x:x+w]
            coord.append([x, y, w, h])
            
    return gray, detected_faces, coord

# 전체 이미지에서 찾아낸 얼굴을 추출하는 함수
def extract_face_features(gray, detected_faces, coord, offset_coefficients=(0.075, 0.05)):
    new_face = []
    for det in detected_faces:
        
        # 얼굴로 감지된 영역
        x, y, w, h = det
        
        # 이미지 경계값 받기
        horizontal_offset = np.int(np.floor(offset_coefficients[0] * w))
        vertical_offset = np.int(np.floor(offset_coefficients[1] * h))
        
        # gray scacle 에서 해당 위치 가져오기
        extracted_face = gray[y+vertical_offset:y+h, x+horizontal_offset:x-horizontal_offset+w]
        
        # 얼굴 이미지만 확대
        new_extracted_face = zoom(extracted_face, (shape_x/extracted_face.shape[0], shape_y/extracted_face.shape[1]))
        new_extracted_face = new_extracted_face.astype(np.float32)
        new_extracted_face /= float(new_extracted_face.max()) # sacled
        new_face.append(new_extracted_face)
        
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

def read_emotion(ret, frame):
    ret, frame = ret, frame
    if not ret:
        print("Failed to grab frame")
    # ret: 비디오를 성공적으로 읽어왔는지 확인 True/False
    # frame: 각 픽셀의 색상을 포함한 프레임 정보 Numpy
    
    face_index = 0
    gray, detected_faces, coord = detect_face(frame)
    
    try :
        face_zoom = extract_face_features(gray, detected_faces, coord)
        face_zoom = np.reshape(face_zoom[0].flatten(), (1, 48, 48, 1))
    except :
        print("\nModule Error: face_emo_module, face_zoom\n")
        return ""
    try:
        x, y, w, h = coord[face_index]
    except:
        print("\nModule Error: list index out of range\n")
    
    # 감정 예측
    pred = model.predict(face_zoom)

    emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    result = ""
    # for emotion, probability in zip(emotions, pred[0]):
    #     result += f"{emotion}: {probability:.3f}\n"
    result += f"facial expressions: {emotions[np.argmax(pred)]}"

    return result