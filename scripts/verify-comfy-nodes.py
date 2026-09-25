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
    "DaSiWa_TorchResize",
    "DaSiWa_RTX_UpscalerRefiner",
    # KJNodes
    "MiniMaxChunkFeedForward",
    # MMH3-UltimateUpscale pack
    "MMH3UltimateUpscale",
    "MMH3LatentUpscaleWithModelParams",
    "MMH3TemporalSplitParams",
    "MMH3SpatialSplitParams",
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




def build_all_graphs() -> dict:
    """Build one graph per production mode via the real factory."""
    sys.path.insert(0, os.getenv("APP_ROOT", "/opt/nocturne-video") + "/worker")
    import workflow_factory as wf

    paths = {a["slug"]: a_manager_comfy_relative(a)
             for a in load_manifest_assets()}
    specs = {
        "t2va": {"task": "t2va",
                 "prompt": {"integrated_multimodal_description": "gate check"}},
        "turbo": {"task": "t2va", "quality": "turbo",
                  "prompt": {"integrated_multimodal_description": "gate check"}},
        "i2va": {"task": "i2va", "first_frame": "gate.png",
                 "prompt": {"integrated_multimodal_description": "gate check"}},
        "flf2va": {"task": "flf2va", "first_frame": "gate.png", "last_frame": "gate.png",
                   "prompt": {"integrated_multimodal_description": "gate check"}},
        "ref2va": {"task": "ref2va", "ref_images": ["gate.png"],
                   "prompt": {"subject_definitions": "s", "summary": "s",
                              "retention_analysis": "r", "detailed_description": "d",
                              "overall_soundscape": "o"}},
        "interpolated": {"task": "t2va", "frame_interpolation": True,
                         "prompt": {"integrated_multimodal_description": "gate check"}},
        "upscale_model": {"task": "t2va",
                          "upscale": {"mode": "model", "model": "2x-animesharpv4-rcan"},
                          "prompt": {"integrated_multimodal_description": "gate check"}},
        "upscale_simple": {"task": "t2va",
                           "upscale": {"mode": "simple", "multiplier": 2},
                           "prompt": {"integrated_multimodal_description": "gate check"}},
        "upscale_rtx": {"task": "t2va", "upscale": {"mode": "rtx", "scale": 2},
                        "prompt": {"integrated_multimodal_description": "gate check"}},
        "upscale_h3_latent": {"task": "t2va", "upscale": {"mode": "h3_latent"},
                              "prompt": {"integrated_multimodal_description": "gate check"}},
        "cache_watermark": {"task": "t2va", "cache": {"reuse_threshold": 0.05},
                            "watermark": {"image": "gate.png"},
                            "chunk_ffn": True,
                            "prompt": {"integrated_multimodal_description": "gate check"}},
    }
    graphs = {}
    for name, spec in specs.items():
        graph, _ = wf.build_graph(spec, paths=paths, gen_id="gate", upload_dir="gate")
        graphs[f"{name}"] = graph
    return graphs


def load_manifest_assets():
    import json as _json
    manifest_path = os.getenv("ASSET_MANIFEST",
                              os.getenv("APP_ROOT", "/opt/nocturne-video") + "/config/assets.manifest.json")
    return _json.loads(open(manifest_path).read())["assets"]


def a_manager_comfy_relative(asset):
    rel = asset.get("relative_dir") or ""
    return f"{rel}/{asset['filename']}" if rel else asset["filename"]


def validate_graph_inputs(graphs: dict, object_info: dict) -> list:
    """Every input name a graph sets must exist in the node's schema; string
    values must be members of their combo list where one exists."""
    problems = []
    for gname, graph in graphs.items():
        for nid, node in graph.items():
            ct = node["class_type"]
            schema = object_info.get(ct)
            if schema is None:
                problems.append(f"{gname}/{nid}: unknown class {ct}")
                continue
            inp = schema.get("input", {})
            valid = set((inp.get("required") or {}).keys()) | set((inp.get("optional") or {}).keys())
            for iname, ivalue in node["inputs"].items():
                if iname not in valid:
                    problems.append(
                        f"{gname}/{nid} ({ct}): input {iname!r} not in node schema "
                        f"(valid: {sorted(valid)})")
                    continue
                spec_in = (inp.get("required") or {}).get(iname) or (inp.get("optional") or {}).get(iname)
                # Combo validation applies only to non-empty option lists. The
                # file-picker combos (unet_name, vae_name, watermark_path,
                # model_name, image, ...) list files from disk, and during the
                # build the volume is not mounted and the folders are empty, so
                # an empty list means "resolved at runtime", not "invalid".
                # Some packs answer with a placeholder like
                # "(no .safetensors upscale models found in: ...)" instead.
                if (isinstance(spec_in, (list, tuple)) and spec_in
                        and isinstance(spec_in[0], (list, tuple)) and len(spec_in[0]) > 0
                        and not any(str(opt).startswith("(") for opt in spec_in[0])
                        and isinstance(ivalue, str) and ivalue not in spec_in[0]):
                    problems.append(
                        f"{gname}/{nid} ({ct}): {iname}={ivalue!r} not in combo "
                        f"{list(spec_in[0])[:12]}")
    return problems


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

    # Validate every input name the production graphs set, against the real
    # schema of the ComfyUI this image ships. This is what catches renames
    # like BasicGuider's positive->conditioning before they burn GPU jobs.
    try:
        graphs = build_all_graphs()
        problems += validate_graph_inputs(graphs, info)
    except Exception as exc:  # noqa: BLE001
        problems.append(f"graph validation itself failed: {type(exc).__name__}: {exc}")

    if problems:
        print("ComfyUI node verification failed: " + "; ".join(problems), file=sys.stderr)
        return 1
    print(f"ComfyUI node verification passed: {len(REQUIRED_NODES)} nodes + minimax CLIP type "
          f"+ {sum(len(g) for g in graphs.values())} graph nodes' inputs across "
          f"{len(graphs)} production graphs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
