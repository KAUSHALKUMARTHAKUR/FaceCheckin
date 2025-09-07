FROM python:3.9-slim

# Install system dependencies + execstack tool
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgstreamer1.0-0 \
    libgstreamer-plugins-base1.0-0 \
    libgtk-3-0 \
    libcairo-gobject2 \
    cmake \
    build-essential \
    libopenblas-dev \
    liblapack-dev \
    execstack \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# Use stable ONNX Runtime version
RUN pip install --no-cache-dir -r requirements.txt

# Fix ONNX Runtime executable stack issue
RUN find /usr/local/lib/python*/site-packages/onnxruntime -name "*.so" -exec execstack -c {} \; 2>/dev/null || true

COPY . .

RUN mkdir -p models/anti-spoofing

EXPOSE 5000

# Reduce workers to 1 for stability
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--timeout", "120", "app:app"]
