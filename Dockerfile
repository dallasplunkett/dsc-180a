FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/workspace

WORKDIR /workspace

COPY requirements.txt .
RUN apt-get update && \
    apt-get install -y --no-install-recommends libhdf5-dev ca-certificates && \
    pip install --no-cache-dir -r requirements.txt && \
    rm -rf /var/lib/apt/lists/*

COPY . .
RUN mkdir -p checkpoints wandb && chmod -R a+rwX /workspace

CMD ["python", "main.py"]
