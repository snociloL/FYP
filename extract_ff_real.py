"""
Extract face sequences from the FaceForensics++ "original" (REAL) videos into
D:\train\real — the fix for real-data scarcity. Mirrors extract_ff_deepfakes.py.

Gives ~1000 real video sequences (vs the previous ~58), balancing the dataset
against the ~1251 fake sequences while keeping the ResNet+LSTM architecture
unchanged. FF++ originals are the SAME identities the FF++ fakes were made from,
so the model learns manipulation, not identity.

Writes ffreal_{video}_{frame}_0.jpg into D:\train\real, matching the
{video}_{frame}_{face} pattern prepare_dataset.py scans for.

    python extract_ff_real.py
"""
import os, glob
import cv2
import numpy as np

SRC = r"D:\FaceForensics\FaceForensics++_C23\FaceForensics++_C23\original"
DST = r"D:\train\real"
FRAMES_PER_VIDEO = 90
JPEG_QUALITY = 90
MIN_FACE_SEQ = 30

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def crop_face(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
    mx, my = int(w * 0.3), int(h * 0.3)
    crop = bgr[max(0, y - my):y + h + my, max(0, x - mx):x + w + mx]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (224, 224))


videos = sorted(glob.glob(os.path.join(SRC, "*.mp4")))
print(f"{len(videos)} REAL videos found in {SRC}")
os.makedirs(DST, exist_ok=True)

too_few, total_saved = [], 0
for vi, path in enumerate(videos):
    stem = "ffreal_" + os.path.splitext(os.path.basename(path))[0]    # e.g. ffreal_000
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total <= 0:
        cap.release(); continue
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
        cv2.imwrite(os.path.join(DST, f"{stem}_{fi}_0.jpg"), crop,
                    [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        saved += 1
    cap.release()
    total_saved += saved
    if saved < MIN_FACE_SEQ:
        too_few.append((stem, saved))
    if vi % 50 == 0:
        print(f"  [{vi}/{len(videos)}] {stem}: {saved}/{FRAMES_PER_VIDEO} faces saved")

print(f"\nDone. {total_saved} new REAL face crops written to {DST}")
print(f"Videos with < {MIN_FACE_SEQ} usable frames: {len(too_few)} / {len(videos)}")
print("\nNext:")
print("  1. python prepare_dataset.py   (real count should jump to ~1000 sequences)")
print("  2. python convert_to_jpeg.py   (re-make D:\\train_jpg with the new reals)")
print("  3. Re-zip train_jpg, upload, re-run Stage B")
