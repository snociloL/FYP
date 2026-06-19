"""
Train ResNet50 + LSTM deepfake detector on the face-crop sequences.

Strategy (fast on CPU): ResNet50 is FROZEN, so its 2048-d features are static.
We extract them ONCE (cached to features.npz), then train only the LSTM head
on cached features. Finally we rebuild the full ResNet50+LSTM model and save it
so the extension/backend can feed raw 30-frame clips.

PREPROCESSING: ResNet50 features use keras resnet50.preprocess_input. The saved
full model therefore expects inputs already passed through preprocess_input.
Inference (test_model.py / backend) MUST apply the same preprocess_input.

Run:
    python prepare_dataset.py        # once, builds sequences_manifest.json
    python train_deepfake.py
"""
import os, json, hashlib
import numpy as np

MANIFEST = r"D:\Codes\Code files\FYP\sequences_manifest.json"
FEAT_CACHE = r"D:\Codes\Code files\FYP\features.npz"
OUT_MODEL = r"D:\Codes\Code files\FYP\deepfake_resnet_lstm_v2.h5"
SEQ_LEN = 30
SIZE = 224
SAMPLES_PER_SEQ = 3      # random 30-frame windows per train sequence each epoch
EPOCHS = 40
BATCH = 8

import keras
from keras import layers
from keras.applications.resnet50 import ResNet50, preprocess_input
import cv2

man = json.load(open(MANIFEST))
train_seqs, val_seqs = man["train"], man["val"]
print("Sequences:", man["counts"])

# ---- 1. Collect every unique frame path referenced by the manifest ----
all_paths = set()
for s in train_seqs + val_seqs:
    all_paths.update(s["frames"])
all_paths = sorted(all_paths)
path_to_idx = {p: i for i, p in enumerate(all_paths)}
print(f"Unique frames to featurize: {len(all_paths)}")

# ---- 2. Extract (or load cached) ResNet50 features ----
def load_img(p):
    img = cv2.imread(p)                       # BGR
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (SIZE, SIZE))
    return img.astype(np.float32)

if os.path.exists(FEAT_CACHE):
    print("Loading cached features…")
    z = np.load(FEAT_CACHE, allow_pickle=True)
    feats = z["feats"]
    cached_paths = list(z["paths"])
    if cached_paths != all_paths:
        print("Cache mismatch — re-extracting.")
        os.remove(FEAT_CACHE)

if not os.path.exists(FEAT_CACHE):
    print("Extracting ResNet50 features (one-time; slow on CPU, fast on GPU)…")
    base = ResNet50(include_top=False, weights="imagenet",
                    pooling="avg", input_shape=(SIZE, SIZE, 3))
    base.trainable = False
    feats = np.zeros((len(all_paths), 2048), np.float32)
    B = 64
    for i in range(0, len(all_paths), B):
        batch_paths = all_paths[i:i + B]
        x = np.stack([load_img(p) for p in batch_paths])
        x = preprocess_input(x)
        feats[i:i + B] = base.predict(x, verbose=0)
        if (i // B) % 20 == 0:
            print(f"  {i}/{len(all_paths)}")
    np.savez(FEAT_CACHE, feats=feats, paths=np.array(all_paths))
    print(f"Saved feature cache -> {FEAT_CACHE}")

# ---- 3. Data generator over cached features ----
class SeqGen(keras.utils.PyDataset):
    def __init__(self, seqs, training, **kwargs):
        super().__init__(**kwargs)
        self.training = training
        # expand: list each seq SAMPLES_PER_SEQ times when training (random windows)
        self.items = []
        reps = SAMPLES_PER_SEQ if training else 1
        for s in seqs:
            for _ in range(reps):
                self.items.append(s)
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.items) / BATCH))

    def on_epoch_end(self):
        self.order = np.random.permutation(len(self.items)) if self.training \
                     else np.arange(len(self.items))

    def _window(self, frames):
        n = len(frames)
        if self.training:
            # random contiguous-ish window: evenly spaced start jitter
            if n > SEQ_LEN:
                start = np.random.randint(0, n - SEQ_LEN + 1)
                idx = np.linspace(start, min(n - 1, start + SEQ_LEN * 2), SEQ_LEN)
                idx = np.clip(idx.astype(int), 0, n - 1)
            else:
                idx = np.clip(np.arange(SEQ_LEN), 0, n - 1)
        else:
            idx = np.linspace(0, n - 1, SEQ_LEN).astype(int)
        return [frames[i] for i in idx]

    def __getitem__(self, b):
        ids = self.order[b * BATCH:(b + 1) * BATCH]
        X = np.zeros((len(ids), SEQ_LEN, 2048), np.float32)
        y = np.zeros((len(ids), 1), np.float32)
        for j, k in enumerate(ids):
            s = self.items[k]
            win = self._window(s["frames"])
            X[j] = feats[[path_to_idx[p] for p in win]]
            y[j] = s["label"]
        return X, y

# ---- 4. LSTM head model ----
head = keras.Sequential([
    layers.Input((SEQ_LEN, 2048)),
    layers.LSTM(128, name="lstm_head"),
    layers.Dense(512, activation="relu", name="dense_a"),
    layers.Dropout(0.5),
    layers.Dense(1, activation="sigmoid", name="dense_out"),
])
head.compile(optimizer=keras.optimizers.Adam(1e-4),
             loss="binary_crossentropy",
             metrics=["accuracy", keras.metrics.AUC(name="auc")])
head.summary()

# ---- 5. class weights for the 3.4x imbalance ----
n_fake = man["counts"]["train_fake"]
n_real = man["counts"]["train_real"]
total = n_fake + n_real
class_weight = {0: total / (2 * n_real), 1: total / (2 * n_fake)}
print("class_weight:", class_weight)

ckpt = keras.callbacks.ModelCheckpoint(
    "head_best.weights.h5", monitor="val_auc", mode="max",
    save_best_only=True, save_weights_only=True, verbose=1)
es = keras.callbacks.EarlyStopping(monitor="val_auc", mode="max",
                                   patience=8, restore_best_weights=True)
rlr = keras.callbacks.ReduceLROnPlateau(monitor="val_auc", mode="max",
                                        factor=0.5, patience=4, verbose=1)

head.fit(SeqGen(train_seqs, True),
         validation_data=SeqGen(val_seqs, False),
         epochs=EPOCHS, class_weight=class_weight,
         callbacks=[ckpt, es, rlr])

# ---- 6. Separation check on validation ----
print("\n=== VALIDATION SEPARATION CHECK ===")
vg = SeqGen(val_seqs, False)
ys, ps = [], []
for b in range(len(vg)):
    X, y = vg[b]
    ps.extend(head.predict(X, verbose=0).ravel())
    ys.extend(y.ravel())
ys, ps = np.array(ys), np.array(ps)
fake_scores, real_scores = ps[ys == 1], ps[ys == 0]
print(f"FAKE val mean={fake_scores.mean():.3f}  REAL val mean={real_scores.mean():.3f}")
print(f"Separation={abs(fake_scores.mean()-real_scores.mean()):.3f}  "
      f"(want a clear gap; >0.2 is good)")
acc = ((ps > 0.5).astype(int) == ys).mean()
print(f"Val accuracy @0.5: {acc:.3f}")

# ---- 7. Rebuild the FULL ResNet50+LSTM model and transfer head weights ----
print("\nRebuilding full ResNet50+LSTM model…")
frame_in = layers.Input((SEQ_LEN, SIZE, SIZE, 3))
base = ResNet50(include_top=False, weights="imagenet",
                pooling="avg", input_shape=(SIZE, SIZE, 3))
base.trainable = False
x = layers.TimeDistributed(base)(frame_in)
x = layers.LSTM(128, name="lstm_head")(x)
x = layers.Dense(512, activation="relu", name="dense_a")(x)
x = layers.Dropout(0.5)(x)
out = layers.Dense(1, activation="sigmoid", name="dense_out")(x)
full = keras.Model(frame_in, out)
for name in ["lstm_head", "dense_a", "dense_out"]:
    full.get_layer(name).set_weights(head.get_layer(name).get_weights())
full.save(OUT_MODEL)
print(f"[OK] Saved full model -> {OUT_MODEL}")
print("Inference must apply resnet50.preprocess_input to frames (as in test_model.py).")
