"""
Extend the FAKE side of D:\train with extra video-sequence diversity from the
FaceForensics++ "Deepfakes" raw videos (1000 mp4s, different manipulation
method than the existing DFDC-derived D:\train\fake crops).

This does NOT add real videos and does NOT fix the real-data scarcity
bottleneck (still only ~58 real videos) — it only broadens fake-side coverage
so the LSTM sees more than one forgery method. Balanced sampling in
train_deepfake.py already oversamples the minority (real) class, so growing
the fake pool just means more steps/epoch and more reused, re-augmented real
windows — that's fine.

Writes directly into D:\train\fake\ as ffdf_{video}_{frame}_0.jpg, matching the
{video}_{frame}_{face}.png|jpg pattern prepare_dataset.py already scans for.
No changes needed to prepare_dataset.py or train_deepfake.py.

    python extract_ff_deepfakes.py
"""
import os, glob
import cv2
import numpy as np

SRC = r"D:\FaceForensics\FaceForensics++_C23\Deepfakes"
DST = r"D:\train\fake"
FRAMES_PER_VIDEO = 90        # evenly spaced across the full clip (~15-20s @ 25-30fps)
JPEG_QUALITY = 90
MIN_FACE_SEQ = 30            # prepare_dataset.py needs >=30 successful frames/video

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def crop_face(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
    mx, my = int(w * 0.3), int(h * 0.3)          # 30% margin, matches backend.py
    crop = bgr[max(0, y - my):y + h + my, max(0, x - mx):x + w + mx]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (224, 224))


videos = sorted(glob.glob(os.path.join(SRC, "*.mp4")))
print(f"{len(videos)} videos found in {SRC}")
os.makedirs(DST, exist_ok=True)

too_few = []
total_saved = 0
for vi, path in enumerate(videos):
    stem = "ffdf_" + os.path.splitext(os.path.basename(path))[0]   # e.g. ffdf_000_003
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total <= 0:
        cap.release()
        continue
    idxs = np.linspace(0, total - 1, FRAMES_PER_VIDEO).astype(int)
    saved = 0
    for fi, idx in enumerate(idxs):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        crop = crop_face(frame)
        if crop is None:
            continue
        out_path = os.path.join(DST, f"{stem}_{fi}_0.jpg")
        cv2.imwrite(out_path, crop, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        saved += 1
    cap.release()
    total_saved += saved
    if saved < MIN_FACE_SEQ:
        too_few.append((stem, saved))
    if vi % 50 == 0:
        print(f"  [{vi}/{len(videos)}] {stem}: {saved}/{FRAMES_PER_VIDEO} faces saved")

print(f"\nDone. {total_saved} new face crops written to {DST}")
print(f"Videos with < {MIN_FACE_SEQ} successful frames (won't form a usable "
      f"sequence): {len(too_few)} / {len(videos)}")
if too_few[:10]:
    print("  e.g.:", too_few[:10])
print("\nNext steps:")
print("  1. python prepare_dataset.py   (will pick up the new ffdf_* files automatically)")
print("  2. Re-zip D:\\train and re-upload to Colab, then re-run Stage B (train_deepfake.py / Cell 2)")
print("     Stage A / the backbone weights file do NOT need to be redone.")
