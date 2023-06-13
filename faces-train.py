import cv2
import os
import numpy as np
import pickle
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, "images")

face_cascade = cv2.CascadeClassifier('cascade/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create() # Local Binary Patterns Histograms, untuk training recognizer

current_id = 0
label_ids = {}
y_labels = []
x_train = []

for root, dirs, files in os.walk(IMAGE_DIR):
    for file in files:
        if file.endswith("png") or file.endswith("jpg") or file.endswith("jpeg"):
            path = os.path.join(root, file)
            label = os.path.basename(root).replace(" ", "-").lower()

            if not label in label_ids:
                label_ids[label] = current_id
                current_id += 1
            
            id_ = label_ids[label]

            pil_image = Image.open(path).convert("L")  # convert dataset gambar ke grayscale
            image_array = np.array(pil_image, "uint8")
            faces = face_cascade.detectMultiScale(image_array, 1.3, 5)

            for (x, y, w, h) in faces:
                roi = image_array[y:y + h, x:x + w]
                x_train.append(roi)
                y_labels.append(id_)

with open("labels.pickle", "wb") as f:
    pickle.dump(label_ids, f) # simpan label_ids ke labels.pickle

recognizer.train(x_train, np.array(y_labels)) # training recognizer
recognizer.save("trainer.yml") # simpan data training recognizer ke trainner.yml