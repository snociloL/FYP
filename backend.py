"""
Flask backend for the Deepfake Video Validator extension.
Runs locally on http://127.0.0.1:5000

Install dependencies:
    pip install flask flask-cors keras tensorflow opencv-python numpy

Run:
    python backend.py
"""
import base64
import numpy as np
import cv2
import keras
from keras.applications.resnet50 import preprocess_input
from flask import Flask, request, jsonify
from flask_cors import CORS

MODEL_PATH = r"D:\Codes\Code files\FYP\deepfake_resnet_lstm_v2.h5"
SEQ_LEN = 30
SIZE = 224

app = Flask(__name__)
CORS(app)   # allow chrome-extension:// origin

print("Loading model …")
model = keras.models.load_model(MODEL_PATH, compile=False)
model.predict(np.zeros((1, SEQ_LEN, SIZE, SIZE, 3), np.float32), verbose=0)
print("[OK] Model ready — listening on http://127.0.0.1:5000")


def decode_frame(b64: str) -> np.ndarray:
    # strip optional data-URL prefix ("data:image/jpeg;base64,...")
    if "," in b64:
        b64 = b64.split(",", 1)[1]
    raw = base64.b64decode(b64)
    arr = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)   # BGR uint8
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (SIZE, SIZE))
    return img.astype(np.float32)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    frames_b64 = data.get("frames", [])

    if len(frames_b64) != SEQ_LEN:
        return jsonify({"error": f"Expected {SEQ_LEN} frames, got {len(frames_b64)}"}), 400

    frames = np.stack([decode_frame(f) for f in frames_b64])   # (30, 224, 224, 3)
    frames = preprocess_input(frames)
    x = frames[np.newaxis, ...]                                  # (1, 30, 224, 224, 3)

    score = float(model.predict(x, verbose=0).ravel()[0])

    if score > 0.75:
        label = "DEEPFAKE"
        pct   = round(score * 100)
    elif score < 0.45:
        label = "AUTHENTIC"
        pct   = round((1 - score) * 100)
    else:
        label = "UNCERTAIN"
        pct   = round(abs(score - 0.5) / 0.5 * 100)

    return jsonify({"score": score, "label": label, "confidence": pct})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
