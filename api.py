from flask import Flask, jsonify, request, abort

app = Flask(__name__)

from os.path import expanduser
from cv2 import cv2
import numpy as np
import time
import dlib
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm, trange
from threading import Thread
from keras.models import load_model
from sklearn.preprocessing import Normalizer

class MainDetection():

    def __init__():
        self.frame = None
        

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
                x = np.expand_dims(x, axis=0)
                feature = model.predict(x)
                encoded_face = x_encoder.transform(np.array([feature[0]]))
                predicted = classifier.predict(encoded_face)
                pred_class = label_encoding[predicted[0]]
                result.append(((x1, y1, x2, y2), pred_class))
            except Exception as e:
                print(str(e))

    def get_vis(data):
        video = data
        capturer = cv2.VideoCapture(video)
        capturer.set(1, 19500)
        output = cv2.VideoWriter('data/result/data_result.mp4', -1, 15.0, (1077, 720))
        frame_time = []
        counter = 0
        while True:
            try:
                now = time.time()
                ret, frame = capturer.read()
                if not ret:
                    break
                frame = cv2.resize(frame, (1280, 720))

                frame = frame[:,0:1077]

                if counter % 5 == 0:
                    worker(frame.copy())
                
                counter += 1
                time.sleep(1/30)
                for item in result:
                    cv2.rectangle(frame, (item[0][0], item[0][1]), (item[0][2], item[0][3]), (255, 255, 255), 2)
                    cv2.putText(frame, item[1], (item[0][0], item[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                if len(frame_time) != 0:
                    total_time = sum(frame_time)
                    cv2.putText(frame, str(int(len(frame_time)/total_time)) + ' fps', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                output.write(frame)
                cv2.imshow('video', frame)
                frame_time.append(time.time()-now)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except Exception as e:
                print(str(e))
                break

        cv2.destroyAllWindows()
        capturer.release()
        output.release()

@app.route('/post', methods='POST') # EXAMPLE - with file and roi
def detect(stream):
    frame = MainDetection()
    Thread(target=frame.get_vis(stream))

    return frame

@app.route('/get', methods='GET')
def get_detection_video():
    frame = None # ambil dari video yg udah di detect
    
    return frame

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
