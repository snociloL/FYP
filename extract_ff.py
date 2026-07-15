"""
Build the FaceForensics++-ONLY dataset for Model C in D:\train_ff.

REAL  : FF++ original videos (ffreal_*)            -> D:\train_ff\real
FAKE  : 4 manipulation methods, all generated from the same originals:
        Deepfakes (ffdf_), Face2Face (fff2f_),
        FaceShifter (ffshift_), FaceSwap (ffswap_) -> D:\train_ff\fake

DFDC and the Pexels "Real" folder are deliberately EXCLUDED — single-source FF++.

To save time it MOVES the already-extracted Deepfakes (ffdf_) and originals
(ffreal_) out of D:\train, then extracts only the 3 remaining methods.
Same naming scheme so prepare_dataset.py + identity-aware split work unchanged.

    python extract_ff.py
"""
import os, glob
import cv2
import numpy as np

FF = r"D:\FaceForensics\FaceForensics++_C23"
ORIG = os.path.join(FF, "FaceForensics++_C23", "original")   # FF++ real videos
DST_FAKE = r"D:\train_ff\fake"
DST_REAL = r"D:\train_ff\real"
FRAMES_PER_VIDEO, SIZE, JPEG_Q = 90, 224, 90

os.makedirs(DST_FAKE, exist_ok=True)
os.makedirs(DST_REAL, exist_ok=True)
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
    return cv2.resize(crop, (SIZE, SIZE)) if crop.size else None


def existing_stems(dst):
    """Video stems already extracted into dst (for resume)."""
    s = set()
    if os.path.isdir(dst):
        for e in os.scandir(dst):
            if e.name.endswith(".jpg"):
                s.add(e.name[:-4].rsplit("_", 2)[0])   # strip _{frame}_{face}.jpg
    return s


def extract(folder, prefix, dst, done=None):
    done = done or set()
    src = folder if os.path.isabs(folder) else os.path.join(FF, folder)
    vids = sorted(glob.glob(os.path.join(src, "*.mp4")))
    print(f"\n{folder}: {len(vids)} videos -> {prefix}", flush=True)
    saved_total = skipped = 0
    for vi, path in enumerate(vids):
        stem = prefix + os.path.splitext(os.path.basename(path))[0]
        if stem in done:                 # already extracted in a previous run
            skipped += 1; continue
        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        if total <= 0:
            cap.release(); continue
        for fi, idx in enumerate(np.linspace(0, total - 1, FRAMES_PER_VIDEO).astype(int)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok:
                continue
            crop = crop_face(frame)
            if crop is None:
                continue
            cv2.imwrite(os.path.join(dst, f"{stem}_{fi}_0.jpg"), crop,
                        [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])
            saved_total += 1
        cap.release()
        if vi % 100 == 0:
            print(f"  {folder} {vi}/{len(vids)}", flush=True)
    print(f"  {folder}: {saved_total} crops saved, {skipped} videos skipped (already done)", flush=True)


# Resume-safe: videos already extracted in a previous run are skipped.
done_real = existing_stems(DST_REAL)
done_fake = existing_stems(DST_FAKE)
print(f"Resume: {len(done_real)} real + {len(done_fake)} fake videos already done", flush=True)

extract(ORIG,          "ffreal_",  DST_REAL, done_real)   # REAL: FF++ originals
extract("Deepfakes",   "ffdf_",    DST_FAKE, done_fake)   # FAKE: 4 manipulation methods
extract("Face2Face",   "fff2f_",   DST_FAKE, done_fake)
extract("FaceShifter", "ffshift_", DST_FAKE, done_fake)
extract("FaceSwap",    "ffswap_",  DST_FAKE, done_fake)

print("\nDone. D:\\train_ff is now FaceForensics++-only.")
print("Next: python prepare_dataset.py   (TRAIN_DIR already points to D:\\train_ff)")
