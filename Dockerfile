# Two-stage build. PyTorch's CUDA wheels bring their own CUDA/cuDNN libraries, so the
# minimal "base" CUDA image is enough at runtime (the host GPU driver is mounted in by
# the NVIDIA container runtime). Build tools stay in the first stage only.

# ---------- Stage 1: install and compile Python packages ----------
FROM nvidia/cuda:12.8.1-base-ubuntu22.04 AS build

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev build-essential \
    && rm -rf /var/lib/apt/lists/*

# Pin PyTorch so requirements.txt cannot replace it with another build.
RUN printf "torch==2.7.1\ntorchvision==0.22.1\n" > /tmp/constraints.txt \
    && python3 -m pip install --upgrade pip \
    && python3 -m pip install torch==2.7.1 torchvision==0.22.1 \
        --index-url https://download.pytorch.org/whl/cu128

COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install -r /tmp/requirements.txt -c /tmp/constraints.txt

# GroundingDINO's C++ extension needs the compiler from this stage.
COPY weights/GroundingDINO /tmp/GroundingDINO
RUN cd /tmp/GroundingDINO && python3 -m pip install --no-build-isolation . -c /tmp/constraints.txt

# ---------- Stage 2: runtime image ----------
FROM nvidia/cuda:12.8.1-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# curl + zstd: health checks and the Ollama installer used by the Vast.ai on-start
# script (the installer needs zstd to unpack its archive).
# libgl1/libglib2.0-0/libsm6/libxext6/libxrender1: OpenCV runtime libraries.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip curl ca-certificates zstd \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Installed packages and their console scripts (uvicorn, etc.) from the build stage.
COPY --from=build /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=build /usr/local/bin /usr/local/bin

WORKDIR /app
COPY . .

# Fail the build here if any runtime import is broken.
RUN python3 -c "import cv2, torch, torchvision, groundingdino, segment_anything, open_clip, transformers, fastapi, uvicorn, ollama; \
from services.description.video_item_describer import describe_detections; print('runtime imports OK')"

EXPOSE 8000
CMD ["python3", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
