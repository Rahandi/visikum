from cv2 import cv2
from threading import Thread
import sys
import dlib

# cascPath = 'haarcascade_frontalface_default.xml'
# faceCascade = cv2.CascadeClassifier(cascPath)
detector = dlib.get_frontal_face_detector()

# video_capture = cv2.VideoCapture('output1.mp4')
video_capture = cv2.VideoCapture('rtsp://admin:12345hik@192.168.0.101:554/h264/ch1/main/av_stream')

data = []

def worker(frame):
    global data
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    # print(len(rects))
    temp = []
    for item in rects:
        temp.append((item.left(), item.top(), item.right(), item.bottom()))
    data = temp

count = 0
while True:
    try:
        # Capture frame-by-frame
        ret, frame = video_capture.read()

        if not ret:
            break

        # if count % 5 == 0:
        #     t = Thread(target=worker, args=(frame,))
        #     t.start()
        #     pass

        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # rects = detector(gray, 1)
        # print(len(rects))
        # temp = []
        # for item in rects:
        #     temp.append((item.left(), item.top(), item.right(), item.bottom()))
        # data = temp
        
        count += 1

        # Draw a rectangle around the faces
        for (x, y, w, h) in data:
            # cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except:
        pass 

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()