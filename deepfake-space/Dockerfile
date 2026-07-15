FROM python:3.11-slim

# opencv-python-headless needs libglib at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Keep Keras/HF caches inside a writable dir (HF Spaces runs as a non-root user)
ENV HOME=/app \
    KERAS_HOME=/app/.keras \
    HF_HOME=/app/.cache

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

# Single worker: the model loads once into memory; long timeout for cold inference
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "1", "--timeout", "120", "app:app"]
