FROM nvcr.io/nvidia/pytorch:24.07-py3
#pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime

RUN apt-get update && apt-get -y install libopenmpi-dev nano htop ffmpeg build-essential gcc g++ make cmake

RUN mkdir -p /root/apps/rag_app
COPY . /root/apps/rag_app

WORKDIR /root/apps/rag_app

RUN pip install numpy typing_extensions 
RUN pip install -r requirements.txt

# SGLANg
RUN pip install --upgrade pip
RUN pip install uv
RUN pip install "sglang[all]>=0.5.3rc0"

# Faster Whisper
RUN pip install faster-whisper

# Livekit Cli (for cloud)
RUN curl -sSL https://get.livekit.io/cli | bash
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8989", "--limit-concurrency", "100", "--limit-max-requests", "10000", "--workers", "1", "--loop", "uvloop", "--http", "httptools", "--access-log", "--log-level", "warning"]
