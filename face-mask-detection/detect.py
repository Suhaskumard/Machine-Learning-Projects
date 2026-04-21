import cv2
import numpy as np
from tensorflow.keras.models import load_model

print("[INFO] Loading model...")
model = load_model("mask_model.h5")

# Load Haar cascade face detector (built-in)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        if face.size == 0:
            continue

        face = cv2.resize(face, (224, 224))
        face = face.astype(np.float32) / 255.0
        face = np.expand_dims(face, axis=0)

        (mask, noMask) = model.predict(face, verbose=0)[0]

        label = "Mask" if mask > noMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.rectangle(frame, (x, y),
                      (x+w, y+h), color, 2)

    cv2.imshow("Face Mask Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
