import cv2

# Load recognizer and labels
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

label_map = {}
with open("labels.txt", "r") as f:
    for line in f:
        key, name = line.strip().split(":")
        label_map[int(key)] = name

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y + h, x:x + w]
        label, confidence = recognizer.predict(face)

        name = label_map.get(label, "Unknown")
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"{name} ({round(confidence, 2)})", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    cv2.imshow("Live Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

