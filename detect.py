from cv2 import cv2
from threading import Thread
import sys

cascPath = 'haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture('rtsp://admin:12345hik@192.168.0.101:554/h264/ch1/main/av_stream')

data = []

def worker(frame):
    global data
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
    data = faces

count = 0
while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    if count % 5 == 0:
        t = Thread(target=worker, args=(frame,))
        t.start()
    
    count += 1

    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # faces = faceCascade.detectMultiScale(
    #     gray,
    #     scaleFactor=1.1,
    #     minNeighbors=5,
    #     minSize=(30, 30),
    #     flags=cv2.CASCADE_SCALE_IMAGE
    # )

    # Draw a rectangle around the faces
    for (x, y, w, h) in data:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()