"""
Train ResNet50 + LSTM deepfake detector — v2 (balanced + augmented + fine-tune).

Upgrades over the cached-feature version, all aimed at fixing the model's bias
toward "fake" and its low confidence on real videos:

  1. BALANCED SAMPLING — each epoch draws an equal number of real and fake
     sequences (the minority real class is oversampled with fresh random
     windows + augmentation), so the model stops defaulting to "fake".
  2. IMAGE AUGMENTATION — per-sequence flip / brightness / contrast / JPEG
     re-compression. JPEG artifacts matter: deepfakes carry tell-tale
     compression traces, so this teaches robustness instead of memorising.
  3. OPTIONAL FINE-TUNING — after the LSTM head learns, unfreeze ResNet's top
     block (conv5) at a low LR so the CNN adapts to face-manipulation cues.

Because of (2) and (3) the frozen-feature cache can't be used — frames go
through ResNet end-to-end every epoch. THIS NEEDS A GPU (use Colab T4).

    python prepare_dataset.py     # once, builds sequences_manifest.json
    python train_deepfake.py
"""
import os, json, random
import numpy as np
import cv2
import keras
from keras import layers
from keras.applications.resnet50 import ResNet50, preprocess_input

# ── Config ──────────────────────────────────────────────────────────────────
MANIFEST   = r"D:\Codes\Code files\FYP\sequences_manifest.json"
OUT_MODEL  = r"D:\Codes\Code files\FYP\deepfake_resnet_lstm_v2.h5"
SEQ_LEN, SIZE = 30, 224
BATCH = 4                       # 4 seqs × 30 frames = 120 imgs/step (fits a T4)
EPOCHS_HEAD     = 25            # phase 1: frozen ResNet, train LSTM head
FINE_TUNE       = True          # phase 2: unfreeze conv5 and fine-tune
EPOCHS_FINETUNE = 12
random.seed(42); np.random.seed(42)

man = json.load(open(MANIFEST))
train = man["train"]; val = man["val"]
train_fake = [s for s in train if s["label"] == 1]
train_real = [s for s in train if s["label"] == 0]
print(f"Train: {len(train_fake)} fake / {len(train_real)} real   Val: {len(val)}")


# ── Data generator: balanced, with per-sequence augmentation ────────────────
class SeqGen(keras.utils.PyDataset):
    def __init__(self, training, fake=None, real=None, items=None, **kwargs):
        super().__init__(**kwargs)
        self.training = training
        self.fake, self.real, self.items = fake, real, items
        self.on_epoch_end()

    def on_epoch_end(self):
        if self.training:
            n = max(len(self.fake), len(self.real))     # oversample minority
            items = [(random.choice(self.fake), 1) for _ in range(n)] + \
                    [(random.choice(self.real), 0) for _ in range(n)]
            random.shuffle(items)
            self._items = items
        else:
            self._items = self.items

    def __len__(self):
        return int(np.ceil(len(self._items) / BATCH))

    def _window(self, frames):
        n = len(frames)
        if self.training and n > SEQ_LEN:
            start = random.randint(0, n - SEQ_LEN)
            idx = np.clip(np.linspace(start, min(n - 1, start + SEQ_LEN * 2),
                                      SEQ_LEN).astype(int), 0, n - 1)
        else:
            idx = np.linspace(0, n - 1, SEQ_LEN).astype(int)
        return [frames[i] for i in idx]

    def _aug_params(self):
        return dict(flip=random.random() < 0.5,
                    bright=random.uniform(0.85, 1.15),
                    contrast=random.uniform(0.85, 1.15),
                    jpeg_q=random.choice([None, 90, 70, 55, 40]))

    def _load(self, path, p):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (SIZE, SIZE))
        if p:                                            # same aug across the seq
            if p["flip"]:
                img = img[:, ::-1]
            img = np.clip((img.astype(np.float32) - 128) * p["contrast"] + 128, 0, 255)
            img = np.clip(img * p["bright"], 0, 255).astype(np.uint8)
            if p["jpeg_q"]:
                enc = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                                   [cv2.IMWRITE_JPEG_QUALITY, p["jpeg_q"]])[1]
                img = cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        return img.astype(np.float32)

    def __getitem__(self, b):
        batch = self._items[b * BATCH:(b + 1) * BATCH]
        X = np.zeros((len(batch), SEQ_LEN, SIZE, SIZE, 3), np.float32)
        y = np.zeros((len(batch), 1), np.float32)
        for j, (seq, label) in enumerate(batch):
            p = self._aug_params() if self.training else None
            frames = self._window(seq["frames"])
            X[j] = preprocess_input(np.stack([self._load(f, p) for f in frames]))
            y[j] = label
        return X, y


train_gen = SeqGen(True, fake=train_fake, real=train_real)
val_gen   = SeqGen(False, items=[(s, s["label"]) for s in val])

# ── Model ───────────────────────────────────────────────────────────────────
base = ResNet50(include_top=False, weights="imagenet", pooling="avg",
                input_shape=(SIZE, SIZE, 3))
base.trainable = False
inp = layers.Input((SEQ_LEN, SIZE, SIZE, 3))
x = layers.TimeDistributed(base)(inp)
x = layers.LSTM(128, name="lstm_head")(x)
x = layers.Dense(512, activation="relu", name="dense_a")(x)
x = layers.Dropout(0.5)(x)
out = layers.Dense(1, activation="sigmoid", name="dense_out")(x)
model = keras.Model(inp, out)

model.compile(optimizer=keras.optimizers.Adam(1e-4),
              loss="binary_crossentropy",
              metrics=["accuracy", keras.metrics.AUC(name="auc")])

ckpt = keras.callbacks.ModelCheckpoint(OUT_MODEL, monitor="val_auc", mode="max",
                                       save_best_only=True, verbose=1)
es  = keras.callbacks.EarlyStopping(monitor="val_auc", mode="max",
                                    patience=7, restore_best_weights=True)
rlr = keras.callbacks.ReduceLROnPlateau(monitor="val_auc", mode="max",
                                        factor=0.5, patience=3, verbose=1)

print("\n=== Phase 1: train LSTM head (ResNet frozen) ===")
model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS_HEAD,
          callbacks=[ckpt, es, rlr])

# ── Phase 2: fine-tune ResNet's top block (conv5) at a low LR ────────────────
if FINE_TUNE:
    print("\n=== Phase 2: fine-tune ResNet conv5 block ===")
    base.trainable = True
    for l in base.layers:
        # only conv5_* trainable; keep BatchNorm frozen to preserve running stats
        l.trainable = l.name.startswith("conv5") and not isinstance(
            l, keras.layers.BatchNormalization)
    model.compile(optimizer=keras.optimizers.Adam(1e-5),
                  loss="binary_crossentropy",
                  metrics=["accuracy", keras.metrics.AUC(name="auc")])
    es2 = keras.callbacks.EarlyStopping(monitor="val_auc", mode="max",
                                        patience=6, restore_best_weights=True)
    model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS_FINETUNE,
              callbacks=[ckpt, es2, rlr])

# ── Validation separation check ─────────────────────────────────────────────
print("\n=== VALIDATION SEPARATION CHECK ===")
ys, ps = [], []
for b in range(len(val_gen)):
    X, y = val_gen[b]
    ps.extend(model.predict(X, verbose=0).ravel())
    ys.extend(y.ravel())
ys, ps = np.array(ys), np.array(ps)
fake_s, real_s = ps[ys == 1], ps[ys == 0]
print(f"FAKE mean={fake_s.mean():.3f}  REAL mean={real_s.mean():.3f}")
print(f"Separation={abs(fake_s.mean()-real_s.mean()):.3f}  (want > 0.3)")
print(f"Val accuracy @ midpoint: "
      f"{(((ps > (fake_s.mean()+real_s.mean())/2).astype(int)) == ys).mean():.3f}")

# ── Auto-calibrate decision thresholds from the validation distributions ─────
F, R = float(fake_s.mean()), float(real_s.mean())
fs, rs = float(fake_s.std()), float(real_s.std())
B = (F + R) / 2
auth_edge, fake_edge = R + rs, F - fs          # 1-std edge of each class
if fake_edge > auth_edge:                       # clean gap → real uncertain band
    deepfake_t, authentic_t = fake_edge, auth_edge
else:                                            # distributions overlap → hard boundary
    deepfake_t = authentic_t = B
calib = {
    "boundary": B,
    "deepfake_t": deepfake_t,
    "authentic_t": authentic_t,
    "half_width": max(0.1, (F - R) / 2),        # maps each class mean to ~100% confidence
    "stats": {"fake_mean": F, "real_mean": R, "fake_std": fs, "real_std": rs},
}
calib_path = os.path.join(os.path.dirname(OUT_MODEL) or ".", "calibration.json")
json.dump(calib, open(calib_path, "w"), indent=2)
print("\nAuto-calibration:", json.dumps(calib, indent=2))

model.save(OUT_MODEL)
print(f"\n[OK] Saved -> {OUT_MODEL}")
print(f"[OK] Saved -> {calib_path}")
print("Upload BOTH the .h5 and calibration.json to the Space. Backend already does "
      "face-crop + preprocess_input to match training.")
