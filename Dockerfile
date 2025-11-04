FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /workspace

RUN apt-get update && \
    apt-get install -y --no-install-recommends libhdf5-dev ca-certificates && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY . /workspace

RUN mkdir -p /workspace/checkpoints && chmod -R 777 /workspace/checkpoints

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/workspace

CMD ["/bin/bash"]