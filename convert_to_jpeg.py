"""
Re-encode the PNG face crops in D:\train as JPEG to shrink the dataset ~3x
before uploading to Colab. Writes a parallel D:\train_jpg\{fake,real}\ folder
(originals untouched). Zip D:\train_jpg and upload that.

    python convert_to_jpeg.py
"""
import os, glob
import cv2

SRC = r"D:\train"
DST = r"D:\train_jpg"
QUALITY = 92                       # high base quality; training adds more compression

for cls in ("fake", "real"):
    os.makedirs(os.path.join(DST, cls), exist_ok=True)
    # handle both the PNG (DFDC) crops and the already-JPEG (FF++) crops
    files = glob.glob(os.path.join(SRC, cls, "*.png")) + \
            glob.glob(os.path.join(SRC, cls, "*.jpg"))
    print(f"{cls}: {len(files)} images")
    for i, p in enumerate(files):
        img = cv2.imread(p)
        if img is None:
            continue
        name = os.path.splitext(os.path.basename(p))[0] + ".jpg"
        cv2.imwrite(os.path.join(DST, cls, name), img,
                    [cv2.IMWRITE_JPEG_QUALITY, QUALITY])
        if i % 10000 == 0:
            print(f"  {cls} {i}/{len(files)}")

print(f"\nDone -> {DST}")
print("Now zip D:\\train_jpg into train.zip and upload that.")
