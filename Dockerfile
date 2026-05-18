FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# ── System dependencies ────────────────────────────────────────────────────────
# Every library here fixes a known failure mode for one of the pip packages.
RUN apt-get update && apt-get install -y --no-install-recommends \
    \
    # C/C++ build tools — chromadb compiles hnswlib from source
    build-essential \
    cmake \
    pkg-config \
    python3-dev \
    \
    # Audio — sounddevice, pyaudio, piper-tts, faster-whisper
    portaudio19-dev \
    libasound2-dev \
    libsndfile1 \
    libsndfile1-dev \
    libpulse0 \
    pulseaudio-utils \
    \
    # Media decoding — faster-whisper uses ffmpeg under the hood
    ffmpeg \
    \
    # OpenMP runtime — onnxruntime (used by piper-tts) requires libgomp
    libgomp1 \
    \
    # OpenCV headless — libglib2.0 is still needed even without a display
    libglib2.0-0 \
    \
    # TLS / download helpers
    ca-certificates \
    curl \
    \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python dependencies ────────────────────────────────────────────────────────
# Upgrade build toolchain first so wheels compile correctly
RUN pip install --upgrade pip setuptools wheel

# Install numpy/scipy before anything else — many packages pin against them
RUN pip install "numpy>=1.24,<2" scipy

# chromadb: builds hnswlib — needs build-essential above to be present
RUN pip install chromadb

# Audio/voice stack — most likely to have system-lib issues
RUN pip install sounddevice pyaudio

# Remaining requirements (everything else)
COPY requirements.txt requirements.xtts.txt ./
RUN pip install -r requirements.txt

# Uncomment the next line to enable XTTS voice cloning (adds ~2GB for PyTorch)
# RUN pip install -r requirements.xtts.txt

# ── Application code ───────────────────────────────────────────────────────────
COPY jarvis/ ./jarvis/
COPY ui/ ./ui/
COPY run.py .

EXPOSE 8000

# Default: web UI + API (no audio hardware required)
CMD ["python", "run.py", "--mode", "api"]
