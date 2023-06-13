import numpy as np
import cv2
import pickle

face_cascade = cv2.CascadeClassifier('cascade/haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('cascade/haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('cascade/haarcascade_smile.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

labels = {"person_name" : 1}
with open("labels.pickle", "rb") as f:
    og_labels = pickle.load(f) # load labels.pickle to og_labels
    labels = {v: k for k, v in og_labels.items()} # reverse og_labels to labels

cap = cv2.VideoCapture(0)
cap.set(3, 800) # set lebar frame
cap.set(4, 640) # set tinggi frame

while (True):
    # Capture gambar per frame
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        
        id_, conf = recognizer.predict(roi_gray)
        print("Confidence level : ", conf)
        if conf >= 4 and conf <= 85:
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (0, 0, 255)
            stroke = 2
            cv2.putText(frame, name, (x, y), font, 1, color, stroke, cv2.LINE_AA)

        color = (255, 0, 0)
        stroke = 2
        end_cord_x = x + w
        end_cord_y = y + h 
        cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey),
                          (ex + ew, ey + eh), (0, 255, 0), 2)

    # Menampilkan hasil frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
