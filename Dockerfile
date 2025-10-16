# --- base image ---
FROM python:3.11-slim-bullseye

# --- system setup ---
WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libhdf5-dev && \
    rm -rf /var/lib/apt/lists/*

# --- copy and install dependencies ---
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

# --- copy code ---
COPY main.py .

# --- default mount path for data ---
VOLUME ["/app/data"]

# --- entry ---
CMD ["python", "main.py"]
