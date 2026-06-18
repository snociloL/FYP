"""
Reconverts best_deepfake_model.h5 to TF.js GraphModel format.
GraphModel avoids the Keras 3.x / TF.js class-name format mismatch.
Run: python convert_model.py
"""
import os
import subprocess
import sys

MODEL_H5   = "best_deepfake_model.h5"
SAVED_DIR  = "saved_model_tmp"
OUT_DIR    = r"deepfake-validator-ext\lib\web_model"

def main():
    import tensorflow as tf
    print(f"TensorFlow: {tf.__version__}")

    # Load from .h5
    print(f"Loading {MODEL_H5} ...")
    model = tf.keras.models.load_model(MODEL_H5)
    model.summary()

    # Export as SavedModel (TF2 format — no Keras version issues)
    print(f"Exporting SavedModel to {SAVED_DIR} ...")
    tf.saved_model.save(model, SAVED_DIR)

    # Convert to TF.js GraphModel via CLI
    print(f"Converting to TF.js GraphModel at {OUT_DIR} ...")
    cmd = [
        sys.executable, "-m", "tensorflowjs.converters.converter",
        "--input_format", "tf_saved_model",
        "--output_format", "tfjs_graph_model",
        "--signature_name", "serving_default",
        "--saved_model_tags", "serve",
        SAVED_DIR,
        OUT_DIR
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print("ERROR:", result.stderr)
        sys.exit(1)

    print("Done! Reload the extension.")

if __name__ == "__main__":
    main()
