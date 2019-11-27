import cv2
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import load_model
import dlib
import os
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm

PATH = 'data/labeled'

# all_images = []
# labels = []
# for d in os.listdir(PATH):
#     # labels.append(d)
#     tmp = []
#     for f in os.listdir(PATH+'/'+d):
#         labels.append(f)
#         tmp.append(f)

#     all_images.append(tmp)


def get_embedding(model, X: np.ndarray):
    X = X.astype('float32')
    mean, std = X.mean(), X.std()
    X = (X - mean) / std
    sample = np.expand_dims(X, axis=0)
    y = model.predict(sample)
    return y[0]


def get_data():
    all_images = []

    labels = []
    for root, dirs, files in os.walk(PATH):
        for name in files:
            all_images.append(os.path.join(root, name))
            labels.append(root.replace(PATH+"/", ""))

    X = np.array(all_images)
    y = np.array(labels)
    return X, y


X, y = get_data()

split = StratifiedShuffleSplit(n_splits=10, test_size=0.3, random_state=1)
for train_index, test_index in split.split(X, y):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

# np.savez_compressed('data/saved_data.npz', X_train, X_test, y_train, y_test)

model = load_model('model/facenet_keras.h5')

newTrainX = []
for face_pixels in tqdm(X_train):
    print(face_pixels)
    img = cv2.imread(face_pixels)
    img = cv2.resize(img, (160, 160))
    embedding = get_embedding(model, img)
    newTrainX.append(embedding)
newTestX = []
for face_pixels in tqdm(X_test):
    img = cv2.imread(face_pixels)
    img = cv2.resize(img, (160, 160))
    embedding = get_embedding(model, img)
    newTestX.append(embedding)
    
newTrainX = np.array(newTrainX)
newTestX = np.array(newTestX)
np.savez_compressed('data/labeled/data_embedded.npz', newTrainX, newTestX)

# Normalize vector data
from sklearn.preprocessing import Normalizer

transformer = Normalizer(norm='l2')
normXTrain = transformer.transform(newTrainX)
normXTest = transformer.transform(newTestX)

# Encoded vector
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
encoder.fit(y_train)
encodedYTrain = encoder.transform(y_train)
encodedYTest = encoder.transform(y_test)

# Create training model
from sklearn.svm import SVC
train_model = SVC(kernel='linear', probability=True)

train_model.fit(normXTrain, encodedYTrain)

# Predict
yhat_train = train_model.predict(normXTrain)
yhat_test = train_model.predict(normXTest)

# Score
from sklearn.metrics import accuracy_score
score_train = accuracy_score(encodedYTrain, yhat_train)
score_test = accuracy_score(encodedYTest, yhat_test)
print('Accuracy: train=%.3f, test=%.3f' % (score_train*100, score_test*100))
