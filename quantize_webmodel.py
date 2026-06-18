"""
Quantize an existing TF.js layers model (float32) to uint16 in-place,
without tensorflowjs/Keras. Reads the fp32 backup, writes quantized output
that tf.loadLayersModel can consume directly.

uint16 affine quantization (matches TF.js read_weights):
    scale = (max - min) / 65535
    stored = round((value - min) / scale)            # uint16, little-endian
    value  = stored * scale + min                     # dequant on load
"""
import json, os, struct
import numpy as np

SRC = r"D:\Codes\Code files\FYP\deepfake-validator-ext\lib\web_model_fp32"
DST = r"D:\Codes\Code files\FYP\deepfake-validator-ext\lib\web_model"

with open(os.path.join(SRC, "model.json"), "r", encoding="utf-8") as f:
    model = json.load(f)

manifest = model["weightsManifest"]
assert len(manifest) == 1, "expected a single weight group"
group = manifest[0]
specs = group["weights"]

# Concatenate all shard bytes in path order
blob = bytearray()
for p in group["paths"]:
    with open(os.path.join(SRC, p), "rb") as fh:
        blob += fh.read()
blob = bytes(blob)
print(f"Read {len(blob)/1e6:.1f} MB of fp32 weights across {len(group['paths'])} shards")

DTYPE_BYTES = {"float32": 4, "int32": 4, "uint8": 1, "bool": 1}

out = bytearray()
offset = 0
n_quant = 0
for spec in specs:
    shape = spec.get("shape", [])
    count = 1
    for d in shape:
        count *= d
    dtype = spec.get("dtype", "float32")
    nbytes = count * DTYPE_BYTES.get(dtype, 4)
    raw = blob[offset:offset + nbytes]
    offset += nbytes

    if dtype != "float32":
        # leave non-float weights untouched
        out += raw
        continue

    arr = np.frombuffer(raw, dtype="<f4")
    vmin = float(arr.min()) if arr.size else 0.0
    vmax = float(arr.max()) if arr.size else 0.0
    if vmax > vmin:
        scale = (vmax - vmin) / 65535.0
        q = np.round((arr - vmin) / scale).clip(0, 65535).astype("<u2")
    else:
        scale = 1.0
        q = np.zeros(arr.shape, dtype="<u2")
    out += q.tobytes()
    spec["quantization"] = {"dtype": "uint16", "scale": scale, "min": vmin}
    n_quant += 1

assert offset == len(blob), f"consumed {offset} of {len(blob)} bytes — spec/blob mismatch"
print(f"Quantized {n_quant}/{len(specs)} float32 tensors -> {len(out)/1e6:.1f} MB uint16")

# Write a single shard and rewrite the manifest paths
os.makedirs(DST, exist_ok=True)
# Remove old shard files in DST
for fn in os.listdir(DST):
    if fn.endswith(".bin"):
        os.remove(os.path.join(DST, fn))

shard_name = "group1-shard1of1.bin"
with open(os.path.join(DST, shard_name), "wb") as fh:
    fh.write(out)
group["paths"] = [shard_name]

with open(os.path.join(DST, "model.json"), "w", encoding="utf-8") as f:
    json.dump(model, f)

print(f"Wrote quantized model.json + {shard_name} to web_model/")
