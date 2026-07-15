---
title: Deepfake Video Validator
emoji: 🕵️
colorFrom: indigo
colorTo: red
sdk: docker
app_port: 7860
pinned: false
---

# Deepfake Video Validator — Backend

Flask API serving a ResNet-50 + LSTM deepfake detector for the browser extension.

## Endpoints
- `GET /health` → `{ "status": "ok" }`
- `POST /predict` → body `{ "frames": [<30 base64 JPEG frames>] }` → `{ "score", "label", "confidence" }`

The model file `deepfake_resnet_lstm_v2.h5` must be uploaded to this Space repo
(tracked via Git LFS — the HF web uploader does this automatically).
