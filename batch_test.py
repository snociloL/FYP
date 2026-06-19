"""
Batch validation: score many deepfake clips (+ any real clips) and report the
score distributions. If the trained model works, fakes should cluster apart
from reals. Uses the most plausible pipeline: face-crop + ResNet preprocess.

Usage:
    python batch_test.py            # 25 random fakes + all reals
    python batch_test.py 50         # 50 random fakes + all reals
"""
import os, sys, glob, random
import numpy as np

FAKE_DIR = r"D:\FaceForensics\FaceForensics++_C23\Deepfakes"
REAL_DIR = r"D:\FaceForensics\FaceForensics++_C23\Real"
MODEL_PATH = r"D:\Codes\Code files\FYP\best_deepfake_model.h5"
SEQ_LEN, SIZE = 30, 224
N_FAKE = int(sys.argv[1]) if len(sys.argv) > 1 else 25

import keras
from keras.applications.resnet50 import preprocess_input
import cv2

print("Loading model...")
model = keras.models.load_model(MODEL_PATH, compile=False)
print("[OK] model loaded\n")

_face = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def face_seq(path):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total <= 0:
        cap.release(); return None
    idxs = np.linspace(0, total - 1, SEQ_LEN).astype(int)
    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok:
            frames.append(np.zeros((SIZE, SIZE, 3), np.uint8)); continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = _face.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
        if len(faces):
            x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
            mx, my = int(w * 0.3), int(h * 0.3)
            crop = rgb[max(0, y-my):y+h+my, max(0, x-mx):x+w+mx]
        else:
            crop = rgb
        frames.append(cv2.resize(crop, (SIZE, SIZE)))
    cap.release()
    arr = np.asarray(frames, np.float32)
    return preprocess_input(arr)[np.newaxis, ...]


def score_dir(label, files):
    scores = []
    for f in files:
        x = face_seq(f)
        if x is None:
            print(f"  [skip] {os.path.basename(f)}"); continue
        s = float(model.predict(x, verbose=0).ravel()[0])
        scores.append(s)
        print(f"  {label:5} {s:.4f}  {os.path.basename(f)}")
    return np.array(scores)


fake_files = glob.glob(os.path.join(FAKE_DIR, "*.mp4"))
real_files = glob.glob(os.path.join(REAL_DIR, "*.mp4"))
random.seed(42)
random.shuffle(fake_files)
fake_files = fake_files[:N_FAKE]

print(f"Scoring {len(fake_files)} FAKE and {len(real_files)} REAL clips "
      f"(face-crop + resnet preprocess)\n")

print("--- FAKE clips (should score toward one extreme) ---")
fake = score_dir("FAKE", fake_files)
print("\n--- REAL clips (should score toward the other extreme) ---")
real = score_dir("REAL", real_files)


def stats(name, a):
    if len(a) == 0:
        print(f"{name}: no data"); return
    print(f"{name}: n={len(a)}  mean={a.mean():.4f}  std={a.std():.4f}  "
          f"min={a.min():.4f}  max={a.max():.4f}")

print("\n========== SUMMARY ==========")
stats("FAKE", fake)
stats("REAL", real)
if len(fake) and len(real):
    gap = abs(fake.mean() - real.mean())
    print(f"\nMean separation between classes: {gap:.4f}")
    overlap = (fake.min() <= real.max() and real.min() <= fake.max())
    if gap < 0.05 or overlap:
        print(">>> VERDICT: classes do NOT separate — the model does not discriminate.")
    else:
        print(">>> VERDICT: classes separate — the model discriminates.")
else:
    print("\nNeed at least one clip per class for a separation verdict.")
    if len(fake):
        # All fakes scoring on the 'authentic' side proves it fails its own majority class
        wrong = (fake > 0.5).mean() if False else (fake < 0.5).mean()
        print(f"Fakes scoring <0.5 ('authentic' side): {100*wrong:.0f}%  "
              f"(if the model worked, fakes should mostly land on ONE consistent side)")
