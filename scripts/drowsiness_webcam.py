import os
import cv2
import numpy as np
import pygame
import time
from collections import deque
from tensorflow.keras.models import load_model

# =========================
# Paths
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "drowsy_mobilenetv2_final.h5")
SOUND_PATH = os.path.join(BASE_DIR, "sounds", "alert.wav")

# =========================
# Load model
# =========================
model = load_model(MODEL_PATH)
print("✅ Model loaded successfully from:", MODEL_PATH)

# =========================
# Initialize pygame for sound
# =========================
pygame.mixer.init()
if os.path.exists(SOUND_PATH):
    alert_sound = pygame.mixer.Sound(SOUND_PATH)
    print("🔊 Alert sound loaded successfully!")
else:
    alert_sound = None
    print("⚠️ Alert sound file not found. Make sure 'alert.wav' is inside /sounds folder.")

# =========================
# Initialize webcam & face detector
# =========================
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# =========================
# Prediction smoothing & stability
# =========================
pred_buffer = deque(maxlen=5)   # smaller buffer = faster reaction
stable_label = "Awake"
stability_counter = 0
STABILITY_FRAMES = 3            # fewer consistent frames to confirm state
last_alert_time = 0             # for repeated sound

# =========================
# Real-time detection loop
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        face = frame[y:y + h, x:x + w]

        # Lighting normalization
        face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face_eq = cv2.equalizeHist(face_gray)
        face_eq = cv2.cvtColor(face_eq, cv2.COLOR_GRAY2BGR)

        # Preprocess for model
        img = cv2.resize(face_eq, (224, 224))
        img = img.astype("float32") / 255.0
        img = np.expand_dims(img, axis=0)

        # Prediction
        pred = model.predict(img, verbose=0)[0][0]
        pred_buffer.append(pred)
        avg_pred = np.mean(pred_buffer)

        # Dynamic thresholding (more sensitive)
        threshold = 0.35
        if 0.3 < avg_pred < 0.5:
            threshold = 0.33

        current_label = "Drowsy" if avg_pred > threshold else "Awake"

        # Stability check
        if current_label == stable_label:
            stability_counter = 0
        else:
            stability_counter += 1
            if stability_counter >= STABILITY_FRAMES:
                stable_label = current_label
                stability_counter = 0

        # Play alert sound repeatedly while drowsy
        if stable_label == "Drowsy" and alert_sound is not None:
            if time.time() - last_alert_time > 3:
                alert_sound.play()
                last_alert_time = time.time()

        # Display
        color = (0, 0, 255) if stable_label == "Drowsy" else (0, 255, 0)
        cv2.putText(frame, f"{stable_label} ({avg_pred:.2f})", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # (Optional) Show raw probability at bottom-left for debugging
        cv2.putText(frame, f"Raw: {pred:.2f}", (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    else:
        cv2.putText(frame, "Face not detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow("Driver Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# =========================
# Cleanup
# =========================
cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
