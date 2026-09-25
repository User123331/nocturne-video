"""Build gate: boot ComfyUI on CPU inside the image and verify every node
class the production H3 graphs rely on, plus the CLIPLoader "minimax" type.

Exits non-zero with "ComfyUI node verification failed: ..." if anything is
missing. Modeled on the sibling Deck's gate (a generic import check can pass
while an individual node failed to register).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request

REQUIRED_NODES = [
    "UNETLoader",
    "CLIPLoader",
    "VAELoader",
    "EmptyMiniMaxH3LatentAV",
    "MiniMaxH3ImageToVideo",
    "MiniMaxH3AddGuide",
    "MiniMaxH3ReferenceToVideo",
    "MiniMaxH3SigmaShift",
    "RandomNoise",
    "KSamplerSelect",
    "BasicScheduler",
    "BasicGuider",
    "SamplerCustomAdvanced",
    "VAEDecode",
    "VAEDecodeAudio",
    "CreateVideo",
    "SaveVideo",
    "LoraLoaderModelOnly",
    "UpscaleModelLoader",
    "ImageUpscaleWithModel",
    # core RIFE frame interpolation
    "FrameInterpolate",
    "FrameInterpolationModelLoader",
    # DaSiWa-Nodes pack
    "DaSiWa_LTX2LoraLoader",
    "MiniMaxH3Cache",
    "DaSiWa_Watermark",
    # KJNodes
    "MiniMaxChunkFeedForward",
    # MMH3-UltimateUpscale pack
    "MMH3UltimateUpscale",
]

COMFYUI_DIR = os.getenv("COMFYUI_DIR", "/comfyui")
PYTHON = os.getenv("COMFYUI_PYTHON", "/opt/venv/bin/python")
PORT = "8188"
TIMEOUT = 240


def wait_for_object_info(deadline: float) -> dict:
    url = f"http://127.0.0.1:{PORT}/object_info"
    last_err = "no attempt"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                return json.load(resp)
        except Exception as exc:  # noqa: BLE001
            last_err = str(exc)
            time.sleep(2)
    raise RuntimeError(f"ComfyUI /object_info never became ready: {last_err}")


def main() -> int:
    proc = subprocess.Popen(
        [PYTHON, "main.py", "--cpu", "--listen", "127.0.0.1", "--port", PORT,
         "--disable-auto-launch", "--quick-test-for-ci" if os.getenv("QUICK_TEST") else "--dont-print-server"],
        cwd=COMFYUI_DIR,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
    try:
        info = wait_for_object_info(time.time() + TIMEOUT)
    except Exception as exc:  # noqa: BLE001
        print(f"ComfyUI node verification failed: {exc}", file=sys.stderr)
        proc.terminate()
        return 1
    finally:
        proc.terminate()

    missing = [name for name in REQUIRED_NODES if name not in info]
    problems = []
    if missing:
        problems.append(f"missing node classes: {missing}")
    clip = info.get("CLIPLoader", {})
    try:
        types = clip["input"]["required"]["type"][0]
        if "minimax" not in types:
            problems.append(f"CLIPLoader type options lack 'minimax': {types}")
    except (KeyError, TypeError, IndexError):
        problems.append("CLIPLoader schema unreadable")

    if problems:
        print("ComfyUI node verification failed: " + "; ".join(problems), file=sys.stderr)
        return 1
    print(f"ComfyUI node verification passed: {len(REQUIRED_NODES)} nodes + minimax CLIP type")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
