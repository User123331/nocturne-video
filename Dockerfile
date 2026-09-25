# syntax=docker/dockerfile:1.7
ARG TARGETPLATFORM=linux/amd64
ARG WORKER_COMFYUI_IMAGE=runpod/worker-comfyui:5.10.0-base-cuda12.8.1
FROM --platform=${TARGETPLATFORM} ${WORKER_COMFYUI_IMAGE}

USER root

# The base image pins its own ComfyUI checkout; MiniMax H3 support (native
# nodes, "minimax" CLIPLoader type, nested AV latent decode) must exist, so
# advance ComfyUI to a tag that is verified to ship nodes_minimax_h3.py.
ARG COMFYUI_TAG=v0.37.2
RUN cd /comfyui \
    && git fetch --tags --force \
    && git checkout "${COMFYUI_TAG}" \
    && /opt/venv/bin/python -m pip install --no-cache-dir -r requirements.txt

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_ROOT=/opt/nocturne-video \
    ASSET_MANIFEST=/opt/nocturne-video/config/assets.manifest.json \
    VOLUME_ROOT=/runpod-volume \
    ASSET_SYNC_MODE=off \
    ALLOW_MODEL_DOWNLOADS=0 \
    ALLOW_RAW_WORKFLOW=0

# No custom node packs: the production graphs use ComfyUI core nodes only
# (MiniMax H3 native nodes + standard loaders/samplers/SaveVideo).

COPY scripts/verify-comfy-nodes.py /usr/local/bin/verify-comfy-nodes.py
RUN chmod 0555 /usr/local/bin/verify-comfy-nodes.py \
    && timeout 300 /opt/venv/bin/python /usr/local/bin/verify-comfy-nodes.py

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
