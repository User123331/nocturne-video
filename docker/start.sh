#!/usr/bin/env bash
set -Eeuo pipefail

mode="${ASSET_SYNC_MODE:-off}"

# Expose verified volume models to ComfyUI (fast, idempotent), then optionally
# sync missing weights from Civitai/HuggingFace when downloads are enabled.
/opt/venv/bin/python -u /opt/nocturne-video/worker/asset_manager.py expose

case "$mode" in
  off)
    echo 'nocturne-video: runtime asset sync is disabled'
    ;;
  metadata|weights|all)
    required="${REQUIRED_ASSETS:-dasiwa-hybrid-v2-int8,qwen3vl-32b-nvfp4-awq,video-vae-fp16,audio-vae-fp32}"
    echo "nocturne-video: running asset sync mode=$mode"
    python -u /opt/nocturne-video/worker/asset_manager.py sync --mode "$mode" --required "$required"
    ;;
  *)
    echo 'nocturne-video: ASSET_SYNC_MODE must be off, metadata, weights, or all' >&2
    exit 2
    ;;
esac

exec /start-upstream.sh
