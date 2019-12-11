import sys
import time
import dlib
import pickle
import mysql.connector

import numpy as np
import pandas as pd

from cv2 import cv2
from copy import deepcopy
from sort.sort import Sort
from sklearn.svm import SVC
from threading import Thread
from tqdm import tqdm, trange
from datetime import datetime
from keras.models import load_model
from sklearn.preprocessing import Normalizer

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  passwd="",
  database="stream"
)
cursor = mydb.cursor()
cursor.execute('truncate log')
mydb.commit()

dlib_detector = dlib.get_frontal_face_detector()
x_encoder = Normalizer(norm = 'l2')

model = load_model('model/facenet_keras.h5')

label_encoding_file = open('label_encoding.pkl', 'rb')
label_encoding = pickle.load(label_encoding_file)
label_encoding_file.close()

x_train_normalized = np.load('x_train_normalized.npy')
y_train_encoded = np.load('y_train_encoded.npy')
classifier = SVC(kernel='linear', probability=True)
classifier.fit(x_train_normalized, y_train_encoded)
classifier.predict_proba(x_train_normalized[0:5])

tracker = Sort()

result = []
frame_time = []
counter = 0
keluar = []
konstant = 30

def check_out():
    global result

    cloned = deepcopy(result)

    for item in cloned:
        mark = 0
        x = (item[2] + item[0])/2

        if x > 975:
            temp = (item[-1], datetime.now())
            if len(keluar) == 0:
                keluar.append(temp)
                sql = "insert into log (nama, timestamp) values (%s, %s)"
                cursor.execute(sql, temp)
                mydb.commit()
                continue
            counter = 0
            for i in range(len(keluar)-1,-1,-1):
                if keluar[i][0] == temp[0] and (temp[1] - keluar[i][1]).seconds < 60:
                    mark = 1
                if counter >= 10:
                    break
                counter += 1
            if mark == 0:
                keluar.append(temp)
                sql = "insert into log (nama, timestamp) values (%s, %s)"
                cursor.execute(sql, temp)
                mydb.commit()

def worker(frame):
    global result, model
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = dlib_detector(gray, 0)

    result = []

    for item in rects:
        try:
            x1 = item.left()
            y1 = item.top()
            x2 = item.right()
            y2 = item.bottom()

            segmented_face = frame[y1:y2, x1:x2].copy()
            segmented_face = cv2.resize(segmented_face, (160, 160))
            x = segmented_face.astype('float32')
            mean, std = x.mean(), x.std()
            x = (x-mean) / std
            x = np.expand_dims(x, axis = 0)
            feature = model.predict(x)
            encoded_face = x_encoder.transform(np.array([feature[0]]))
            predicted = classifier.predict_proba(encoded_face)
            score = round(max(predicted[0])*100, 2)
            pred_class = label_encoding[(predicted[0].tolist()).index(max(predicted[0]))] if score > 50 else 'unknown'
            result.append((x1, y1, x2, y2, score, pred_class))
        except Exception as e:
            print(str(e))

video_path = 'data/data.mp4'
capturer = cv2.VideoCapture(video_path)
output = cv2.VideoWriter('data/result/data_result.mp4', -1, 30.0, (1280, 720))
while True:
    now = time.time()
    ret, frame = capturer.read()
    top = 30

    if ret:
        frame = cv2.resize(frame, (1280, 720))

        frame[:, 1077:] = 0

        # frame = frame[:, 0:1077]

        if counter % 5 == 0:
            worker(frame.copy())
            check_out()
            # print(keluar)
            # print('==========================================')

        counter += 1

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        cv2.rectangle(frame, (975, 0), (1077, 720), (0,0,0), 1)
        
        for item in result:
            cv2.rectangle(frame, (item[0], item[1]), (item[2], item[3]), (255, 255, 255), 2)
            cv2.putText(frame, str(item[4]), (item[0], item[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, item[5], (item[0], item[1] - 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        for i in range(len(keluar)-1, -1, -1):
            cv2.putText(frame, keluar[i][0], (1080, top), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            top += 30
        if len(frame_time) != 0:
            total_time = sum(frame_time)
            # cv2.putText(frame, str(counter), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, str(int(len(frame_time)/total_time)) + ' fps', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        sys.stdout.buffer.write(frame.tobytes())

        # cv2.imshow('display', frame)
        # output.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

capturer.release()
output.release()
cv2.destroyAllWindows()