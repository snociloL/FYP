"""
Local Flask backend (dev/offline twin of deepfake-space/app.py).
Inference mirrors training: detect + crop the largest face per frame before
running the model; return NO_FACE when too few frames contain a face.

    pip install flask flask-cors keras tensorflow opencv-python numpy
    python backend.py
"""
import os
import json
import base64
import numpy as np
import cv2
import keras
from keras.applications.resnet50 import preprocess_input
from flask import Flask, request, jsonify
from flask_cors import CORS

MODEL_PATH = os.environ.get("MODEL_PATH", r"D:\Codes\Code files\FYP\deepfake_resnet_lstm_v2.h5")
SEQ_LEN = 30
SIZE = 224
MIN_FACE_FRAMES = 8

# Thresholds loaded from calibration.json (written by train_deepfake.py); defaults if absent.
_CALIB = {"deepfake_t": 0.85, "authentic_t": 0.75, "boundary": 0.80, "half_width": 0.20}
for _p in ("calibration.json", os.path.join(os.path.dirname(MODEL_PATH) or ".", "calibration.json")):
    try:
        _CALIB.update({k: v for k, v in json.load(open(_p)).items() if k in _CALIB})
        print(f"[calib] loaded {_p}: {_CALIB}")
        break
    except Exception:
        continue
DEEPFAKE_T  = _CALIB["deepfake_t"]
AUTHENTIC_T = _CALIB["authentic_t"]
BOUNDARY    = _CALIB["boundary"]
HALF_WIDTH  = _CALIB["half_width"]

app = Flask(__name__)
CORS(app)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

print("Loading model …")
model = keras.models.load_model(MODEL_PATH, compile=False)
model.predict(np.zeros((1, SEQ_LEN, SIZE, SIZE, 3), np.float32), verbose=0)
print("[OK] Model ready — http://127.0.0.1:5000")


def decode(b64: str):
    if "," in b64:
        b64 = b64.split(",", 1)[1]
    raw = base64.b64decode(b64)
    return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)


def crop_face(bgr):
    if bgr is None:
        return None
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
    mx, my = int(w * 0.3), int(h * 0.3)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    crop = rgb[max(0, y - my):y + h + my, max(0, x - mx):x + w + mx]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (SIZE, SIZE)).astype(np.float32)


def fill_gaps(crops):
    last = None
    for i in range(len(crops)):
        if crops[i] is None:
            crops[i] = last
        else:
            last = crops[i]
    nxt = None
    for i in range(len(crops) - 1, -1, -1):
        if crops[i] is None:
            crops[i] = nxt
        else:
            nxt = crops[i]
    return crops


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    frames_b64 = data.get("frames", [])
    if len(frames_b64) != SEQ_LEN:
        return jsonify({"error": f"Expected {SEQ_LEN} frames, got {len(frames_b64)}"}), 400

    crops = [crop_face(decode(f)) for f in frames_b64]
    faces_found = sum(c is not None for c in crops)

    if faces_found < MIN_FACE_FRAMES:
        return jsonify({"label": "NO_FACE", "score": None, "confidence": 0,
                        "faces_found": faces_found,
                        "message": "No clear face detected — this tool only analyses faces."})

    seq = preprocess_input(np.stack(fill_gaps(crops)))
    score = float(model.predict(seq[np.newaxis, ...], verbose=0).ravel()[0])

    if score >= DEEPFAKE_T:
        label = "DEEPFAKE"
    elif score <= AUTHENTIC_T:
        label = "AUTHENTIC"
    else:
        label = "UNCERTAIN"

    confidence = int(min(99, max(1, round(abs(score - BOUNDARY) / HALF_WIDTH * 100))))
    return jsonify({"label": label, "score": score, "confidence": confidence,
                    "faces_found": faces_found})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
