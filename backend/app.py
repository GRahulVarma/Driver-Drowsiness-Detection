from flask import Flask, request, jsonify, render_template, send_from_directory
import os
import tensorflow as tf
import cv2
import numpy as np
import base64

app = Flask(__name__, template_folder="../frontend", static_folder="../frontend")

# -----------------------------
# MODEL PATH
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "drowsy_mobilenetv2_final.h5")
MODEL_PATH = os.path.abspath(MODEL_PATH)

print("Looking for model at:", MODEL_PATH)
print("Exists?", os.path.exists(MODEL_PATH))

model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!")

# -----------------------------
# Serve frontend
@app.route("/")
def home():
    return render_template("index.html")  # load frontend/index.html

# Serve static assets (JS, CSS, alert.wav)
@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory("../frontend", filename)

# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json["image"]
        img_data = base64.b64decode(data)
        np_arr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        img = cv2.resize(frame, (224, 224))
        img = img.astype("float32") / 255.0
        img = np.expand_dims(img, axis=0)

        prediction = model.predict(img)[0][0]
        status = "drowsy" if prediction > 0.5 else "awake"

        return jsonify({"prediction": float(prediction), "status": status})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
