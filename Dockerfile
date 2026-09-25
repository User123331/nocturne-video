# syntax=docker/dockerfile:1.7
ARG TARGETPLATFORM=linux/amd64
ARG WORKER_COMFYUI_IMAGE=runpod/worker-comfyui:5.10.0-base-cuda12.8.1
FROM --platform=${TARGETPLATFORM} ${WORKER_COMFYUI_IMAGE}

USER root

# The base image pins its own ComfyUI checkout; MiniMax H3 support (native
# nodes, "minimax" CLIPLoader type, nested AV latent decode, core RIFE
# interpolation) must exist, so advance ComfyUI to a tag verified to ship
# nodes_minimax_h3.py and nodes_frame_interpolation.py.
ARG COMFYUI_TAG=v0.37.2
# torch is pinned FIRST to the +cu128 wheels, exactly as this base image does:
# ComfyUI's requirements.txt declares a bare `torch`, and default PyPI now
# serves CUDA 13 builds, which fail CUDA init on hosts whose driver is 570/575.
# Installing requirements.txt without this pin silently replaces the working
# cu128 torch and every worker then dies in the upstream GPU pre-flight check.
RUN cd /comfyui \
    && git fetch --tags --force \
    && git checkout "${COMFYUI_TAG}" \
    && uv pip install --python /opt/venv/bin/python \
         torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
         --index-url https://download.pytorch.org/whl/cu128 \
    && uv pip install --python /opt/venv/bin/python -r requirements.txt \
    && uv pip install --python /opt/venv/bin/python "transformers>=4.50.3,<5" "huggingface-hub<1.0" \
    && /opt/venv/bin/python -c "import torch; assert torch.version.cuda.startswith('12.'), torch.version.cuda; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_ROOT=/opt/nocturne-video \
    ASSET_MANIFEST=/opt/nocturne-video/config/assets.manifest.json \
    VOLUME_ROOT=/runpod-volume \
    ASSET_SYNC_MODE=off \
    ALLOW_MODEL_DOWNLOADS=0 \
    ALLOW_RAW_WORKFLOW=0

# Custom node packs, pinned. Mirrors the DaSiWa C-MMH3 requirements minus
# rgthree (pure UI labels, never appear in API-format graphs):
#   kjnodes          — MiniMaxChunkFeedForward (VRAM chunking)
#   gguf             — GGUF quant loaders (future quant swaps)
#   dasiwa-nodes     — Advanced LoRA stack (str/vs/as), MiniMaxH3Cache,
#                      watermark, RTX refiner, torch resize
#   mmh3-upscale     — MMH3UltimateUpscale latent 2x pipeline
RUN comfy-node-install \
    comfyui-kjnodes \
    comfyui-gguf

# The two packs the ComfyUI registry does not carry: clone at pinned commits.
# Pinned by FULL commit SHA. GitHub's upload-pack only resolves a complete
# 40-char object id for a shallow fetch; the abbreviated 9-char form fails with
# "fatal: couldn't find remote ref", which is what broke the previous build.
RUN git init -q /comfyui/custom_nodes/ComfyUI-DaSiWa-Nodes \
    && cd /comfyui/custom_nodes/ComfyUI-DaSiWa-Nodes \
    && git remote add origin https://github.com/darksidewalker/ComfyUI-DaSiWa-Nodes.git \
    && git fetch --depth 1 origin a9ea632f8d862394e7ef21cb990ac9cb4bc334bf \
    && git checkout -q FETCH_HEAD
RUN git init -q /comfyui/custom_nodes/Comfyui-MMH3-UltimateUpscale \
    && cd /comfyui/custom_nodes/Comfyui-MMH3-UltimateUpscale \
    && git remote add origin https://github.com/bbaudio-2025/Comfyui-MMH3-UltimateUpscale.git \
    && git fetch --depth 1 origin fe6658f6d144066f14150d3526247b417683ff2b \
    && git checkout -q FETCH_HEAD

# comfy-node-install resolves requirements in its isolated build environment,
# while the worker runs on /opt/venv — install the runtime imports there too
# (KJNodes: color-matcher/matplotlib; GGUF: gguf; DaSiWa pack: stdlib + av).
RUN uv pip install --python /opt/venv/bin/python \
    "color-matcher==0.6.0" \
    "matplotlib==3.11.2" \
    "gguf==0.10.0"

ARG VERIFY_TIMEOUT=300
COPY scripts/verify-comfy-nodes.py /usr/local/bin/verify-comfy-nodes.py
RUN chmod 0555 /usr/local/bin/verify-comfy-nodes.py \
    && timeout ${VERIFY_TIMEOUT} /opt/venv/bin/python /usr/local/bin/verify-comfy-nodes.py

# Preserve the upstream handler and entrypoint so our thin wrapper can
# delegate ComfyUI execution and Runpod output handling to the base image.
RUN mv /handler.py /handler_upstream.py \
    && mv /start.sh /start-upstream.sh

COPY config/ ${APP_ROOT}/config/
COPY worker/ ${APP_ROOT}/worker/
COPY docker/start.sh /start.sh
COPY worker/handler.py /handler.py

RUN chmod 0555 /start.sh /handler.py ${APP_ROOT}/worker/*.py \
    && chmod 0444 ${APP_ROOT}/config/*.json

WORKDIR /
CMD ["/start.sh"]
