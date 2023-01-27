import numpy as np
import cv2 as cv

haar_cascade = cv.CascadeClassifier(
    'haarcascades\haarcascade_frontalface_default.xml')

people = ['Ben Afflek', 'Elton John',
          'Jerry Seinfield', 'Madonna', 'Mindy Kaling']

# features = np.load('feature_trained.npy')
# labels = np.load('label_trained.npy')

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

img = cv.imread('faceRecogintion\Faces/val/Jerry Seinfield/2.jpg')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# detect
face_rect = haar_cascade.detectMultiScale(
    gray, scaleFactor=1.1, minNeighbors=4)

for (x, y, w, h) in face_rect:
    face_roi = gray[y:y+h, x:x+w]  # region of interest

    label, confidence = face_recognizer.predict(face_roi)
    print(f'label = {people[label]} confidence = {confidence}')

    cv.putText(img, str(people[label]), (20, 20),
               cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), thickness=2)

    cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)

cv.imshow('img', img)

cv.waitKey(0)
