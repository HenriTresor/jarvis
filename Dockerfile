FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1 \
    ffmpeg \
    portaudio19-dev \
    libasound2-dev \
    pulseaudio-utils \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies before copying code (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY jarvis/ ./jarvis/
COPY ui/ ./ui/
COPY run.py .
COPY .env.example .

EXPOSE 8000

# Default: web UI + API only (no audio hardware required)
CMD ["python", "run.py", "--mode", "api"]
