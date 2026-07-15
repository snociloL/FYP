"""
Build per-video face-crop SEQUENCES from D:\train\{fake,real} and split into
train/val WITHOUT video leakage. Writes sequences_manifest.json.

Filenames are DFDC crops: {videoID}_{frame}_{face}.png
We group by (videoID, faceIndex) = one face track through time = one sequence.
"""
import os, re, json, random
from collections import defaultdict

TRAIN_DIR = r"D:\train_ff"   # Model C: FaceForensics++ ONLY (original reals + 4 manip fakes)
OUT = r"D:\Codes\Code files\FYP\sequences_manifest.json"
SEQ_LEN = 30
VAL_FRAC = 0.2
random.seed(42)

pat = re.compile(r"^(.*)_(\d+)_(\d+)\.(?:png|jpe?g)$", re.IGNORECASE)

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

# Identity-aware split: a FF++ real video (ffreal_NNN) and the fakes made from
# it (ffdf_NNN_*) share identity NNN, so they must land on the SAME side or the
# model could memorise the person/scene. We therefore split by IDENTITY across
# BOTH classes jointly. DFDC video IDs never share an identity across classes,
# so they fall back to a per-video split.
FF_PREFIXES = ("ffreal_", "ffdf_", "fff2f_", "ffshift_", "ffswap_")  # real + 4 manip methods
def identity_of(vid):
    if vid.startswith(FF_PREFIXES):
        return "ff_" + vid.split("_")[1]      # FF++ target id (e.g. ff_000) — shared by
        # the original video and every manipulation made from it
    return vid                                # (DFDC fallback; unused in FF++-only Model C)

all_seqs = fake + real
for s in all_seqs:
    s["identity"] = identity_of(s["video"])

by_id = defaultdict(list)
for s in all_seqs:
    by_id[s["identity"]].append(s)
ids = list(by_id)
random.shuffle(ids)
val_ids = set(ids[:max(1, int(len(ids) * VAL_FRAC))])

train, val = [], []
for i, ss in by_id.items():
    (val if i in val_ids else train).extend(ss)

f_tr = [s for s in train if s["label"] == 1]
r_tr = [s for s in train if s["label"] == 0]
f_va = [s for s in val   if s["label"] == 1]
r_va = [s for s in val   if s["label"] == 0]
random.shuffle(train); random.shuffle(val)
print(f"Identities: {len(ids)} total, {len(val_ids)} held out for validation")

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
