"""
Validate the deepfake model in pure Keras (ground truth), independent of TF.js.

Usage:
    python test_model.py                      # load model + run a random dummy input
    python test_model.py path\to\video.mp4    # load model + run a real video
    python test_model.py vid1.mp4 vid2.mp4    # several videos

Confirms (a) the .h5 loads, (b) it produces a score, (c) what it predicts on real clips.
"""
import sys
import numpy as np

MODEL_PATH = r"D:\Codes\Code files\FYP\best_deepfake_model.h5"
SEQ_LEN = 30
SIZE = 224

print("Loading Keras...")
import keras
print("Keras", keras.__version__)

print(f"Loading model: {MODEL_PATH}")
model = keras.models.load_model(MODEL_PATH, compile=False)
print("[OK] Model loaded.")
print("Input shape :", model.input_shape)
print("Output shape:", model.output_shape)
model.summary(line_length=100)


def extract_raw_frames(path):
    """Return (30, 224, 224, 3) uint8 RGB frames, evenly sampled."""
    import cv2
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total <= 0:
        raise RuntimeError(f"Could not read frames from {path}")
    idxs = np.linspace(0, total - 1, SEQ_LEN).astype(int)
    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok:
            frame = np.zeros((SIZE, SIZE, 3), np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (SIZE, SIZE))
        frames.append(frame)
    cap.release()
    return np.asarray(frames, np.uint8)                   # (30, 224, 224, 3)


_FACE = None
def _detector():
    global _FACE
    if _FACE is None:
        import cv2
        _FACE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    return _FACE


def extract_face_frames(path, margin=0.3):
    """Like extract_raw_frames but crops the largest detected face per frame."""
    import cv2
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total <= 0:
        raise RuntimeError(f"Could not read frames from {path}")
    idxs = np.linspace(0, total - 1, SEQ_LEN).astype(int)
    det = _detector()
    frames, found = [], 0
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok:
            frames.append(np.zeros((SIZE, SIZE, 3), np.uint8)); continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = det.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
        if len(faces):
            found += 1
            x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
            mx, my = int(w * margin), int(h * margin)
            x0, y0 = max(0, x - mx), max(0, y - my)
            x1, y1 = min(rgb.shape[1], x + w + mx), min(rgb.shape[0], y + h + my)
            crop = rgb[y0:y1, x0:x1]
        else:
            crop = rgb  # fall back to full frame
        frames.append(cv2.resize(crop, (SIZE, SIZE)))
    cap.release()
    print(f"    (faces detected in {found}/{SEQ_LEN} frames)")
    return np.asarray(frames, np.uint8)


def preprocessings(raw_u8):
    """Yield (name, batched_input) for each candidate preprocessing."""
    from keras.applications.resnet50 import preprocess_input
    raw = raw_u8.astype(np.float32)
    yield "div255 (0-1)",        (raw / 255.0)[np.newaxis, ...]
    yield "raw (0-255)",         raw[np.newaxis, ...]
    yield "resnet preprocess",   preprocess_input(raw.copy())[np.newaxis, ...]


def interpret(score):
    # Extension convention: high score -> deepfake, low -> authentic
    if score > 0.85:   verdict = "DEEPFAKE (high confidence)"
    elif score > 0.5:  verdict = "leaning DEEPFAKE"
    elif score > 0.15: verdict = "leaning AUTHENTIC"
    else:              verdict = "AUTHENTIC (high confidence)"
    return verdict


videos = sys.argv[1:]
if not videos:
    print("\nNo video given - running a random dummy input to confirm inference works...")
    x = np.random.rand(1, SEQ_LEN, SIZE, SIZE, 3).astype(np.float32)
    score = float(model.predict(x, verbose=0).ravel()[0])
    print(f"Dummy score = {score:.5f}  ({interpret(score)})")
    print("(random input — verdict is meaningless, this only proves the model runs)")
else:
    for v in videos:
        print(f"\n=== {v} ===")
        try:
            print("  -- FULL FRAME --")
            raw = extract_raw_frames(v)
            for name, x in preprocessings(raw):
                score = float(model.predict(x, verbose=0).ravel()[0])
                print(f"  [full/{name:18}] score = {score:.5f}  ->  {interpret(score)}")

            print("  -- FACE CROP --")
            face = extract_face_frames(v)
            for name, x in preprocessings(face):
                score = float(model.predict(x, verbose=0).ravel()[0])
                print(f"  [face/{name:18}] score = {score:.5f}  ->  {interpret(score)}")
        except Exception as e:
            print("ERROR:", e)
