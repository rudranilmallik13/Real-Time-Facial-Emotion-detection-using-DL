import cv2
import numpy as np
from tensorflow.keras.models import load_model
from utils import EMOTION_LABELS

MODEL_PATH = 'emotion_model.h5'
FACE_CASCADE_PATH = 'haarcascade_frontalface_default.xml'

def preprocess_face(face_img):
    # input: gray face image (cropped)
    face = cv2.resize(face_img, (48, 48))
    face = face.astype('float32') / 255.0
    face = np.expand_dims(face, -1)
    face = np.expand_dims(face, 0)  # (1,48,48,1)
    return face

def main():
    model = load_model(MODEL_PATH)
    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            face_in = preprocess_face(roi)
            preds = model.predict(face_in)
            label_idx = np.argmax(preds)
            label = EMOTION_LABELS.get(label_idx, 'Unknown')
            prob = float(np.max(preds))
            text = f'{label} ({prob:.2f})'
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow('Live Emotion Recognition', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()