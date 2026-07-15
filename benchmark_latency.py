"""
Inference-latency benchmark for Chapter 4.6.3.
Extracts 30 face frames from a sample video, POSTs them to the deployed backend
N times, and reports the server-side face-detection vs inference breakdown plus
the end-to-end round-trip time.

    pip install opencv-python requests numpy
    python benchmark_latency.py path/to/sample_with_a_face.mp4
    python benchmark_latency.py sample.mp4 --url https://snocilol-deepfake-validator.hf.space --n 10
"""
import sys, time, base64, argparse
import cv2
import numpy as np
import requests

ap = argparse.ArgumentParser()
ap.add_argument("video")
ap.add_argument("--url", default="https://snocilol-deepfake-validator.hf.space")
ap.add_argument("--n", type=int, default=10)
args = ap.parse_args()

SEQ_LEN = 30

# --- extract 30 evenly-spaced frames as JPEG base64 (same as the extension) ---
cap = cv2.VideoCapture(args.video)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
if total <= 0:
    sys.exit("Could not read video / no frames.")
idxs = np.linspace(0, total - 1, SEQ_LEN).astype(int)
frames = []
for i in idxs:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
    ok, fr = cap.read()
    if not ok:
        continue
    h, w = fr.shape[:2]
    scale = min(1.0, 720 / max(h, w))
    fr = cv2.resize(fr, (int(w * scale), int(h * scale)))
    ok, enc = cv2.imencode(".jpg", fr, [cv2.IMWRITE_JPEG_QUALITY, 80])
    frames.append("data:image/jpeg;base64," + base64.b64encode(enc).decode())
cap.release()
if len(frames) != SEQ_LEN:
    sys.exit(f"Got {len(frames)} frames, need {SEQ_LEN}.")

print(f"Benchmarking {args.url}/predict over {args.n} runs …\n")

face, infer, server_total, roundtrip = [], [], [], []
for k in range(args.n):
    t0 = time.perf_counter()
    r = requests.post(f"{args.url}/predict", json={"frames": frames}, timeout=120)
    rt = (time.perf_counter() - t0) * 1000
    r.raise_for_status()
    d = r.json()
    if "timing_ms" not in d:
        print("Run", k, "->", d.get("label"), "(no server timing; redeploy the updated app.py)")
        roundtrip.append(rt); continue
    t = d["timing_ms"]
    face.append(t["face_detect"]); infer.append(t["inference"]); server_total.append(t["total"])
    roundtrip.append(rt)
    print(f"run {k+1:2}: {d['label']:9} face={t['face_detect']:6.1f}ms  "
          f"infer={t['inference']:7.1f}ms  server={t['total']:7.1f}ms  roundtrip={rt:7.1f}ms")

def stat(name, a, unit="ms"):
    if a: print(f"  {name:22}: {np.mean(a):8.1f} {unit}  (±{np.std(a):.1f})")

print("\n=== AVERAGES (paste into Table 4.6.3) ===")
stat("Face detection", face)
stat("Model inference", infer)
stat("Server total (per clip)", server_total)
stat("End-to-end round-trip", roundtrip)
if server_total:
    print(f"\n  Per-clip total ~ {np.mean(server_total)/1000:.2f} s server-side, "
          f"{np.mean(roundtrip)/1000:.2f} s including network.")
