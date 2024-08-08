#!/usr/bin/env python3

import warnings
warnings.filterwarnings('ignore')
import requests
from time import sleep
from playsound import playsound
import os

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
# ESP32 URL
URL = "http://192.168.0.129"
AWB = True



cap = cv2.VideoCapture(URL + ":81/stream")
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml') # insert the full path to haarcascade file if you encounter any problem

def set_resolution(url: str, index: int=1, verbose: bool=False):
    try:
        if verbose:
            resolutions = "10: UXGA(1600x1200)\n9: SXGA(1280x1024)\n8: XGA(1024x768)\n7: SVGA(800x600)\n6: VGA(640x480)\n5: CIF(400x296)\n4: QVGA(320x240)\n3: HQVGA(240x176)\n0: QQVGA(160x120)"
            print("available resolutions\n{}".format(resolutions))

        if index in [10, 9, 8, 7, 6, 5, 4, 3, 0]:
            requests.get(url + "/control?var=framesize&val={}".format(index))
        else:
            print("Wrong index")
    except:
        print("SET_RESOLUTION: something went wrong")

def set_quality(url: str, value: int=1, verbose: bool=False):
    try:
        if value >= 10 and value <=63:
            requests.get(url + "/control?var=quality&val={}".format(value))
    except:
        print("SET_QUALITY: something went wrong")

def set_awb(url: str, awb: int=1):
    try:
        awb = not awb
        requests.get(url + "/control?var=awb&val={}".format(1 if awb else 0))
    except:
        print("SET_QUALITY: something went wrong")
    return awb

def set_V_flip(url: str, awb: int=1):
    try:
        vflip = not vflip
        requests.get(url + "/control?var=vflip&val={}".format(1 if awb else 0))
    except:
        print("SET_QUALITY: something went wrong")
    return awb


def set_face_detect(url: str, face_detect: int=1):
    try:
        requests.get(url + "/control?var=face_detect&val={}".format(1 if face_detect else 0))
    except:
        print("SET_FACE_DETECT: something went wrong")
    return face_detect

i=0

emotion = ["angry", "disgust", "fear", "happy", "sad", "surprise"]

if __name__ == '__main__':
    set_resolution(URL, index=8)
    set_quality(URL, 63)
    # set_face_detect(URL, 1)
    set_V_flip(URL, 1)
# 프레임 단위로 영상 캡쳐
    while True:
        ret, frame = cap.read()
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
                
            #     # 머리 둘레에 직사각형 그리기: (0, 255, 0)을 통해 녹색으로 선두께는 2
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
            #     # 감정 예측
            pred = model.predict(face_zoom)
            pred_result = np.argmax(pred)

            if pred_result == 0:
                cv2.putText(frame, "Angry" + str(round(pred[0][0], 2)), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5)
            elif pred_result == 1:
                cv2.putText(frame, "Disgust " + str(round(pred[0][1], 2)), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5)
            elif pred_result == 2:
                cv2.putText(frame, "Fear " + str(round(pred[0][2], 2)), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5)
            elif pred_result == 3:
                cv2.putText(frame, "Happy " + str(round(pred[0][3], 2)), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5)
            elif pred_result == 4:
                cv2.putText(frame, "Sad " + str(round(pred[0][4], 2)), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5)
            elif pred_result == 5:
                cv2.putText(frame, "Suprise " + str(round(pred[0][5], 2)), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5)
            else:
                cv2.putText(frame, "Neutral " + str(round(pred[0][6], 2)), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5)
        except:
            continue
        
        # 결과 표시
        cv2.imshow('Video', frame)

        if (i%50==0):
            print("interval")
            if not(pred_result>5):
                file_path = f"./tts/{emotion[pred_result]}.mp3"
                if os.path.exists(file_path):
                    playsound(file_path)
                    print("play")
                else:
                    print(f"파일을 찾을 수 없습니다: {file_path}")

        
        i = i + 1    
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
cap.release()

# 창 닫기: 창이 안닫히는 경우 쥬피터 닫기
cv2.destroyAllWindows()
