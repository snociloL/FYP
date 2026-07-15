# Deepfake Video Validator

A browser extension that detects AI-manipulated (deepfake) videos in near real time, backed by a
**ResNet-50 + LSTM** spatiotemporal deep-learning model served from a remote Python backend on
Hugging Face Spaces.

> Final Year Project (CSP600/CSP650) — Bachelor of Computer Science, Universiti Teknologi MARA (UiTM).

---

## How it works

```
Browser extension                      Remote backend (Hugging Face Space)
─────────────────                      ────────────────────────────────────
1. Capture video      ──30 JPEG──►     3. Haar-cascade face gate
   (file upload or      frames            (< 8/30 faces → NO_FACE, no guess)
   active tab)                         4. Face crop 224×224 + ResNet-50
2. Sample 30 evenly                       preprocessing
   spaced frames                       5. ResNet-50 per-frame features
                                          (2,048-dim, FF++ fine-tuned)
                      ◄──verdict──     6. LSTM over the 30-frame sequence
                                       7. Calibrated tri-state decision
```

**Verdict thresholds** (loaded from `calibration.json`):

| Score | Verdict | Extension state |
|---|---|---|
| ≥ 0.80 | DEEPFAKE | 🔴 Red |
| ≤ 0.45 | AUTHENTIC | 🟢 Green |
| between | UNCERTAIN | 🟡 Amber |
| < 8/30 faces | NO_FACE | 🟡 Unprocessable |

**Validation results** (identity-disjoint 957-sequence FaceForensics++ split, boundary 0.614):
accuracy **88.5%**, precision **92.6%**, recall **92.9%**, F1 **92.7%**, class separation **0.638**.

---

## Repository layout

| Path | Purpose |
|---|---|
| `deepfake-space/` | Dockerized Flask inference backend deployed to Hugging Face Spaces |
| `backend.py` | Local mirror of the deployed backend (same decision logic) |
| `deepfake-validator-ext/` | Chrome extension (MV3): tab capture, file upload, tri-state UI |
| `train_deepfake.py` | Two-stage training (Stage A ResNet-50 backbone, Stage B LSTM head) + auto-calibration |
| `prepare_dataset.py` | Identity-aware train/val split over FF++ face-crop sequences |
| `extract_ff*.py` | FaceForensics++ (c23) frame extraction (reals + 4 manipulation methods) |
| `convert_to_jpeg.py` | Dataset re-encoding utility |
| `benchmark_latency.py` | N-run latency benchmark against the deployed Space (thesis Table 4.6.3) |
| `batch_test.py`, `test_model.py` | Sanity checks: batch separation test, single-video smoke test |
| `score.png`, `training_curves.png` | Chapter 4 result figures (score distribution, training curves) |

Model weights (`*.h5`), feature caches (`*.npz`) and dataset manifests are **git-ignored** —
they are reproducible via the training scripts, and the deployed weights live in the
[Hugging Face Space](https://huggingface.co/spaces/snocilol/deepfake-validator).

---

## Quick start

**Backend (local):**
```bash
pip install flask flask-cors tensorflow keras opencv-python numpy
python backend.py            # expects deepfake_resnet_lstm_v2.h5 + calibration.json
```

**Extension:** load `deepfake-validator-ext/` as an unpacked extension in
`chrome://extensions` (Developer mode). It points at the deployed Space by default.

**Training (reproduce Model C-v2):**
```bash
python extract_ff_real.py && python extract_ff_deepfakes.py   # FF++ c23 frames
python prepare_dataset.py                                     # identity-safe split -> sequences_manifest.json
python train_deepfake.py                                      # Stage A + B -> .h5 + calibration.json
```
Upload the resulting `.h5` and `calibration.json` to the Space.

> ⚠️ **Evaluation note:** always evaluate against the *training-time* `val_ids.json` identity split.
> Regenerating the manifest produces a different split (≈124/200 identity overlap) and silently
> leaks training identities into validation, inflating metrics.

---

## Development timeline

| Period (2026) | Phase | What happened |
|---|---|---|
| **early Jun** | Project scaffold | Initial extension + in-browser TF.js inference attempt; tensor pre-processing pipeline; workarounds for site security policies (CSP) to capture frames |
| **Jun 7** | First model | First ResNet-50 + LSTM trained; discovered `best_deepfake_model.h5` was a **collapsed constant predictor** (~0.40 for everything) — training data had ~1,000 fakes but almost no reals |
| **mid Jun** | Architecture pivot | TF.js could not load `TimeDistributed(ResNet-50)` (`loadLayersModel` hangs) → inference moved to a **Python/Flask backend**; extension refactored to POST 30-frame batches |
| **Jun 19** | Hybrid retrain | Two-stage fix: Stage A frame-level ResNet-50 backbone on a balanced image set, Stage B LSTM head on video sequences — first model that separated real vs fake (sep ≈ 0.48) |
| **Jun 23–29** | **Model C-v2 (final)** | Both stages retrained purely on **FaceForensics++ (c23)** with an identity-disjoint split shared between stages (`val_ids.json`). Validation: 88.5% acc / 92.7% F1 / separation 0.638; thresholds auto-calibrated (boundary 0.614) |
| **early Jul** | Deployment | Backend containerized and deployed to **Hugging Face Spaces** (gunicorn + tensorflow-cpu 2.18 / keras 3.14); `NO_FACE` gate and per-request timing added; verdict band widened manually (0.80 / 0.45) after the auto-calibrated band collapsed due to real-side variance |
| **Jul 10–13** | Evaluation & thesis | Live latency benchmark (81.8 ms/frame server-side, ≈6.6 s warm end-to-end); full validation re-run reproduced the reported confusion matrix exactly (144/56/54/703); Chapter 4/5 figures generated |
| **Jul 15** | Consolidation | All deployment, training, extraction and benchmarking code committed and PR opened ([#2](https://github.com/snociloL/FYP/pull/2)) |

---

## Modification log

### v2.1 — 2026-07-15 (`d6bc7ec`, [PR #2](https://github.com/snociloL/FYP/pull/2))
- **Added** `deepfake-space/`: Dockerfile, Flask app, pinned requirements for the HF Space deployment
- **Changed** `backend.py` / `app.py`: thresholds loaded from `calibration.json` (deepfake ≥ 0.80, authentic ≤ 0.45, else UNCERTAIN); added `NO_FACE` gate (min 8/30 face frames) and `timing_ms` breakdown in responses
- **Changed** `train_deepfake.py`: FF++-only two-stage training, JPEG re-compression augmentation, validation separation check, automatic threshold calibration
- **Changed** `prepare_dataset.py`: split by *identity* rather than by video (an FF++ real and every fake derived from it stay on the same side)
- **Added** `extract_ff.py` / `extract_ff_real.py` / `extract_ff_deepfakes.py`, `convert_to_jpeg.py`: FF++ (c23) dataset tooling
- **Added** `benchmark_latency.py`: repeated live requests against the Space with server-side timing split
- **Changed** extension: 30-frame batch POST to remote backend, tri-state verdict rendering (Red/Green/Amber + NO_FACE), restyled on-page indicator, MV3 manifest updates
- **Added** `score.png`, `training_curves.png`: Chapter 4 figures

### v2.0 — 2026-06-19 (`98f6d05`)
- **Changed**: retrained ResNet-50 + LSTM as a two-stage hybrid (frame-level backbone + LSTM head) — fixed the collapsed constant-output model
- **Changed**: inference served via Python backend instead of in-browser TF.js (TF.js cannot load `TimeDistributed(ResNet50)`)
- **Added**: YouTube/active-tab capture path in the extension

### v1.x — 2026-06-07 (`59baac3`…`af14898`)
- **Added**: initial extension scaffold and first ResNet-50 + LSTM model
- **Added**: tensor pre-processing pipeline ("Processed Tensors")
- **Changed**: frame-capture workarounds for site security policies (CSP)
- **Fixed**: extension resource path issues

---

## Known limitations

- Trained only on the four FaceForensics++ manipulation families — detection of newer generation
  techniques (e.g. diffusion-based synthesis) is not guaranteed.
- Frontal-face Haar cascade processes the largest face only; profile views, occlusions and
  multi-subject videos may not be analysed correctly.
- Remote inference adds network latency and a cold-start delay (~54 s) on the first request after idle.
