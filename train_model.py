import cv2
import os
import numpy as np

data_dir = 'datasets'
recognizer = cv2.face.LBPHFaceRecognizer_create()
faces = []
labels = []
label_map = {}
label_id = 0

for person in os.listdir(data_dir):
    person_dir = os.path.join(data_dir, person)
    if not os.path.isdir(person_dir):
        continue

    label_map[label_id] = person
    for image_name in os.listdir(person_dir):
        image_path = os.path.join(person_dir, image_name)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        faces.append(image)
        labels.append(label_id)
    label_id += 1

recognizer.train(faces, np.array(labels))
recognizer.save("trainer.yml")

# Save label map
with open("labels.txt", "w") as f:
    for key, value in label_map.items():
        f.write(f"{key}:{value}\n")

print("Model trained and saved.")
