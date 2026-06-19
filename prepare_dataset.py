"""
Build per-video face-crop SEQUENCES from D:\train\{fake,real} and split into
train/val WITHOUT video leakage. Writes sequences_manifest.json.

Filenames are DFDC crops: {videoID}_{frame}_{face}.png
We group by (videoID, faceIndex) = one face track through time = one sequence.
"""
import os, re, json, random
from collections import defaultdict

TRAIN_DIR = r"D:\train"
OUT = r"D:\Codes\Code files\FYP\sequences_manifest.json"
SEQ_LEN = 30
VAL_FRAC = 0.2
random.seed(42)

pat = re.compile(r"^(.*)_(\d+)_(\d+)\.png$", re.IGNORECASE)

def scan(folder, label):
    # groups[(video, face)] = list of (frame_no, path)
    groups = defaultdict(list)
    d = os.path.join(TRAIN_DIR, folder)
    for fn in os.listdir(d):
        m = pat.match(fn)
        if not m:
            continue
        vid, frame, face = m.group(1), int(m.group(2)), int(m.group(3))
        groups[(vid, face)].append((frame, os.path.join(d, fn)))
    seqs = []
    for (vid, face), frames in groups.items():
        if len(frames) < SEQ_LEN:
            continue                      # need at least 30 frames for a sequence
        frames.sort()
        seqs.append({"video": vid, "face": face, "label": label,
                     "frames": [p for _, p in frames]})
    return seqs

print("Scanning fake/ …")
fake = scan("fake", 1)
print("Scanning real/ …")
real = scan("real", 0)
print(f"Usable sequences (>= {SEQ_LEN} frames):  fake={len(fake)}  real={len(real)}")

# Split by VIDEO so frames/faces from one video never straddle train & val
def split_by_video(seqs):
    by_vid = defaultdict(list)
    for s in seqs:
        by_vid[s["video"]].append(s)
    vids = list(by_vid)
    random.shuffle(vids)
    n_val = max(1, int(len(vids) * VAL_FRAC))
    val_vids = set(vids[:n_val])
    train, val = [], []
    for v, ss in by_vid.items():
        (val if v in val_vids else train).extend(ss)
    return train, val

f_tr, f_va = split_by_video(fake)
r_tr, r_va = split_by_video(real)
train = f_tr + r_tr
val   = f_va + r_va
random.shuffle(train); random.shuffle(val)

manifest = {
    "seq_len": SEQ_LEN,
    "train": train,
    "val": val,
    "counts": {
        "train_fake": len(f_tr), "train_real": len(r_tr),
        "val_fake": len(f_va),   "val_real": len(r_va),
    },
}
with open(OUT, "w") as f:
    json.dump(manifest, f)

print("\n=== SPLIT (sequences) ===")
print(f"TRAIN: fake={len(f_tr)}  real={len(r_tr)}  total={len(train)}")
print(f"VAL  : fake={len(f_va)}  real={len(r_va)}  total={len(val)}")
print(f"\nClass imbalance (train): {len(f_tr)/max(1,len(r_tr)):.1f}x more fake than real")
print(f"Saved manifest -> {OUT}")
