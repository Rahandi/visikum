import sys
import os
import dlib
import time

from threading import Thread
from multiprocessing import Process
from glob import glob
from cv2 import cv2
from tqdm import tqdm

BASE_PATH = 'data/'

data = []
videos = glob('data/*')
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
detector = dlib.get_frontal_face_detector()

try:
    os.makedirs('data/segmented')
except:
    pass

def worker_haar(identity, frame, filename, folder):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bounding_boxes = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    count = 0
    for (x, y, w, h) in bounding_boxes:
        segmented_image = frame[y:y+h, x:x+w]
        segmented_filename = folder + filename + '_' + str(identity) + '_' + str(count) + '.png'
        cv2.imwrite(segmented_filename, segmented_image)
        data.append({
            'origin': filename,
            'segmented': segmented_filename,
            'bbs_origin': (x, y, w, h)
        })
        count += 1

def worker_dlib(identity, frame, filename, folder):
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)

        count = 0

        for item in rects:
            x = item.left()
            y = item.top()
            w = item.right() - x
            h = item.bottom() - y

            segmented_image = frame[y:y+h, x:x+w]
            segmented_filename = folder + filename + '_' + str(identity) + '_' + str(count) + '.png'
            cv2.imwrite(segmented_filename, segmented_image)
            data.append({
                'origin': filename,
                'segmented': segmented_filename,
                'bbs_origin': (x, y, w, h)
            })
            count += 1
    except:
        pass

for video_path in tqdm(videos):
    filename = video_path.replace('data\\', '')
    filename = filename.split('.')[0]
    folder_path = 'data/segmented/' + filename

    try:
        os.makedirs(folder_path)
    except:
        pass
    
    folder_path += '/'

    video_capture = cv2.VideoCapture(video_path)
    # video_capture.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)

    pool = []
    counter = 0

    with tqdm(total=int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT) - 1), leave=False) as pbar:
        while True:
            ret, frame = video_capture.read()

            if not ret:
                break
            pbar.update(1)
            if counter % 20 == 0:
                # t = Process(target=worker_dlib, args=(counter, frame, filename, folder_path))
                t = Thread(target=worker_dlib, args=(counter, frame, filename, folder_path))
                # t = Thread(target=worker_haar, args=(counter, frame, filename, folder_path))
                t.start()
                pool.append(t)
            counter += 1

            if len(pool) == 100:
                # time.sleep(1)
                # for t in tqdm(pool, leave=False):
                #     t.start()
                for t in pool:
                    t.join()
                pool = []