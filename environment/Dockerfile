# Docker image for the Glacier Modelling course (UNIL), used as a RenkuLab session.
# JupyterLab + IGM (main branch, pinned commit in requirements.txt) + the scientific Python stack.
# The same image runs on CPU and on NVIDIA GPU nodes (TensorFlow ships its CUDA 12 libraries).
#
# Build & test locally:
#   docker build -t glamod environment/
#   docker run -p 8888:8888 glamod          # then open the printed http://127.0.0.1:8888/lab?token=...
#
# The image is built and pushed automatically to ghcr.io/jouvetg/glamod by
# .github/workflows/build-image.yml, and referenced by the Renku session launcher.

FROM quay.io/jupyter/scipy-notebook:python-3.11.10

USER root
RUN apt-get update && apt-get install -y --no-install-recommends tree ncview curl \
 && rm -rf /var/lib/apt/lists/*
USER ${NB_UID}

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Quieter TensorFlow logs
ENV TF_CPP_MIN_LOG_LEVEL=3
# GPU sessions: allocate GPU memory on demand instead of grabbing the whole GPU
# (needed if several sessions share one GPU). No effect on CPU-only sessions.
ENV TF_FORCE_GPU_ALLOW_GROWTH=true

WORKDIR /home/${NB_USER}
