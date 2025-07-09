import cv2
import os
import argparse

# Parse name from command-line argument
parser = argparse.ArgumentParser(description='Collect face images')
parser.add_argument('--name', type=str, required=True, help='Name of the person')
args = parser.parse_args()

name = args.name.strip()
dataset_path = 'datasets/' + name
os.makedirs(dataset_path, exist_ok=True)

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        count += 1
        face = gray[y:y + h, x:x + w]
        cv2.imwrite(f"{dataset_path}/{count}.jpg", face)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow('Collecting Faces - Press q to stop', frame)

    if cv2.waitKey(1) & 0xFF == ord('q') or count >= 100:
        break

cap.release()
cv2.destroyAllWindows()


#command:python collect_faces.py --name (name of the person)