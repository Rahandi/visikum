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

res_frame = None
class MainDetection():

    def __init__(self):
        self.frame = None
        self.dlib_detector = dlib.get_frontal_face_detector()
        self.x_encoder = Normalizer(norm='l2')
        self.result = []
        self.model = load_model('model/facenet_keras.h5')
        self.model.summary()
        self.classifier_file = open('model/classifier.pkl', 'rb')
        self.classifier = pickle.load(self.classifier_file)
        self.classifier_file.close()
        print(self.classifier)
        self.label_encoding_file = open('label_encoding.pkl', 'rb')
        self.label_encoding = pickle.load(self.label_encoding_file)
        self.label_encoding_file.close()
        print(self.label_encoding)    

    def worker(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = self.dlib_detector(gray, 0)
        
        self.result = []
        
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
                feature = self.model.predict(x)
                encoded_face = self.x_encoder.transform(np.array([feature[0]]))
                predicted = self.classifier.predict(encoded_face)
                pred_class = self.label_encoding[predicted[0]]
                self.result.append(((x1, y1, x2, y2), pred_class))
            except Exception as e:
                print(str(e))

    def preprocessing(self, capturer, counter, frame_time):
        global res_frame
        while True:
            try:
                ret, frame = capturer.read()
                if not ret:
                    break
                frame = cv2.resize(frame, (1280, 720))

                frame = frame[:,0:1077]

                if counter % 5 == 0:
                    self.worker(frame.copy())
                
                counter += 1
                time.sleep(1/30)
                for item in self.result:
                    cv2.rectangle(frame, (item[0][0], item[0][1]), (item[0][2], item[0][3]), (255, 255, 255), 2)
                    cv2.putText(frame, item[1], (item[0][0], item[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                if len(frame_time) != 0:
                    total_time = sum(frame_time)
                    cv2.putText(frame, str(int(len(frame_time)/total_time)) + ' fps', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                res_frame = frame

            except Exception as e:
                print(str(e))
                break
        cv2.destroyAllWindows()
        capturer.release()

    def get_detect(self, data):
        capturer = cv2.VideoCapture(data)
        capturer.set(1, 19500)
        # output = cv2.VideoWriter('data/result/data_result.mp4', -1, 15.0, (1077, 720))
        frame_time = []
        
        t = Thread(target=self.preprocessing, args=(capturer, 0, frame_time))
        t.start()

@app.route('/get', methods=["GET"])
def get_visualization():
    return res_frame

if __name__ == '__main__':
    video = 'data/data.mp4'
    detector = MainDetection() # ambil dari video yg udah di detect
    detector.get_detect(video)
    app.run(host='0.0.0.0', port=5000, debug=True)