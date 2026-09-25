"""Build native ComfyUI API-format graphs for MiniMax H3 video+audio jobs.

The graphs replicate DaSiWa's C-MMH3 settings (samplers, sigma shifts, prompt
section format) using ComfyUI's native MiniMax H3 nodes only — no custom node
packs — so the worker image stays minimal and the build gate can verify every
node class against a live CPU-booted ComfyUI.

Reference material:
- comfy_extras/nodes_minimax_h3.py (ComfyUI core, the node schemas)
- DaSiWa C-MMH3 v2.3 workflow (settings note: sampler/shift/step tables and
  the Director prompt section format)
"""

from __future__ import annotations

import math
import random
from typing import Any

FPS = 24
CANVAS_MULTIPLE = 32
MAX_PIXELS = 768 * 1344
DEFAULT_WIDTH = 1024
DEFAULT_HEIGHT = 768
MAX_WIDTH = 2560
MAX_HEIGHT = 1440
MIN_FRAMES = 5
MAX_FRAMES = 362  # ~15s, the trained range per the native node tooltip

TASKS = ("t2va", "i2va", "flf2va", "ref2va")

PRESETS = {
    "quality": {
        "sampler_name": "res_multistep",
        "scheduler": "simple",
        "steps": 25,
        "shift_video": 11.0,
        "shift_audio": 4.0,
    },
    "turbo": {
        "sampler_name": "euler",
        "scheduler": "simple",
        "steps": 8,
        "shift_video": 7.0,
        "shift_audio": 4.5,
    },
}


class SpecError(ValueError):
    """Raised for user-input problems the local app should surface verbatim."""


def align_frame_count(n: int) -> int:
    """Snap a frame count up to the model's 17k+5 grid."""
    while n % 17 != 5:
        n += 1
    return n


def frames_for_duration(duration_seconds: float) -> int:
    if not 1.0 <= duration_seconds <= 15.0:
        raise SpecError("duration_seconds must be between 1 and 15")
    return align_frame_count(max(MIN_FRAMES, round(duration_seconds * FPS)))


def clamp_canvas(width: int, height: int) -> tuple[int, int]:
    def _snap(v: int) -> int:
        return max(CANVAS_MULTIPLE, round(v / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)

    width, height = _snap(int(width)), _snap(int(height))
    if width * height > MAX_PIXELS:
        # The trained canvas tops out around 768x1344; allow explicit larger
        # canvases (the node accepts them) but cap hard at ~2K area.
        cap = 2560 * 1440
        if width * height > cap:
            s = math.sqrt(cap / (width * height))
            width, height = int(width * s), int(height * s)
    return min(width, MAX_WIDTH), min(height, MAX_HEIGHT)


def _section(key: str, value: str) -> str:
    return f"{key}: {value.strip()}"


def assemble_prompt(task: str, prompt: dict[str, Any], duration_seconds: float) -> str:
    """Replicate DaSiWa Director's prompt sections and alignment lines."""
    if task == "ref2va":
        sections = [
            ("subject_definitions", "subject_definitions"),
            ("summary", "summary"),
            ("retention_analysis", "retention_analysis"),
            ("detailed_description", "detailed_description"),
            ("overall_soundscape", "overall_soundscape"),
            ("non_diegetic_music", "non_diegetic_music"),
        ]
        parts = []
        for field, key in sections:
            value = (prompt.get(field) or "").strip()
            if field == "non_diegetic_music" and not value:
                value = "N/A"
            parts.append(_section(key, value))
        return "\n\n".join(parts)

    imd = (prompt.get("integrated_multimodal_description") or "").strip()
    if not imd:
        raise SpecError("integrated_multimodal_description is required")
    soundscape = (prompt.get("overall_soundscape") or "").strip()
    music = (prompt.get("non_diegetic_music") or "").strip() or "N/A"
    body = "\n\n".join([
        _section("integrated_multimodal_description", imd),
        _section("overall_soundscape", soundscape),
        _section("non_diegetic_music", music),
    ])
    if task == "t2va":
        return body
    if task == "i2va":
        alignment = (
            "For the target video, at 0.00 seconds into the target video, "
            "Picture 1 (from Shot 1) is fully referenced."
        )
        return f"{alignment}\n{body}"
    if task == "flf2va":
        alignment = (
            "How the reference pictures align with the target video — "
            "Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; "
            f"Picture 2 (from Shot 1) aligns with the {duration_seconds:.2f}-second mark of the target video."
        )
        return f"{alignment}\n{body}"
    raise SpecError(f"unknown task {task!r}")


def _n(node_id: str, class_type: str, inputs: dict[str, Any]) -> dict[str, Any]:
    return {"class_type": class_type, "inputs": inputs}


def build_graph(
    spec: dict[str, Any],
    *,
    paths: dict[str, str],
    gen_id: str,
    upload_dir: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (api_format_graph, meta) for one H3 generation.

    paths: manifest slug -> absolute volume path, provided by the handler after
    asset verification. upload_dir: directory (inside ComfyUI's input folder)
    where the handler has already written this job's uploaded media.
    """
    task = spec.get("task")
    if task not in TASKS:
        raise SpecError(f"task must be one of {TASKS}")
    quality = spec.get("quality", "quality")
    if quality not in PRESETS:
        raise SpecError("quality must be 'quality' or 'turbo'")
    preset = PRESETS[quality]

    checkpoint_slug = spec.get("checkpoint", "dasiwa-hybrid-v2-int8")
    if checkpoint_slug not in paths:
        raise SpecError(f"unknown checkpoint slug {checkpoint_slug!r}")

    duration = float(spec.get("duration_seconds", 5))
    frames = frames_for_duration(duration)
    width, height = clamp_canvas(spec.get("width", DEFAULT_WIDTH), spec.get("height", DEFAULT_HEIGHT))

    prompt = spec.get("prompt") or {}
    final_prompt = assemble_prompt(task, prompt, frames / FPS)

    steps = int(spec.get("steps", preset["steps"]))
    if not 1 <= steps <= 60:
        raise SpecError("steps must be 1..60")
    shift_video = float(spec.get("shift_video", preset["shift_video"]))
    shift_audio = float(spec.get("shift_audio", preset["shift_audio"]))
    sampler_name = spec.get("sampler_name", preset["sampler_name"])
    scheduler = spec.get("scheduler", preset["scheduler"])

    seed = spec.get("seed", -1)
    if seed in (None, "", -1, "-1"):
        seed = random.randint(0, 2**53 - 1)
    seed = int(seed)
    if not 0 <= seed <= 2**53 - 1:
        raise SpecError("seed must be 0..2^53-1")

    te_slug = spec.get("text_encoder", "qwen3vl-32b-nvfp4-awq")
    if te_slug not in paths:
        raise SpecError(f"unknown text_encoder slug {te_slug!r}")

    workflow: dict[str, Any] = {}

    workflow["1"] = _n("1", "UNETLoader", {
        "unet_name": paths[checkpoint_slug], "weight_dtype": "default",
    })

    model_source = "1"
    for idx, lora in enumerate(spec.get("loras") or [], start=1):
        name = (lora.get("name") or "").strip()
        if not name:
            raise SpecError("lora entries need a name")
        strength = float(lora.get("strength", 1.0))
        if not -4.0 <= strength <= 4.0:
            raise SpecError("lora strength must be within -4..4")
        nid = f"10{idx}"
        workflow[nid] = _n(nid, "LoraLoaderModelOnly", {
            "lora_name": name, "strength_model": strength, "model": [model_source, 0],
        })
        model_source = nid

    workflow["2"] = _n("2", "CLIPLoader", {
        "clip_name": paths[te_slug], "type": "minimax", "device": "default",
    })
    workflow["3"] = _n("3", "VAELoader", {"vae_name": paths["video-vae-fp16"]})
    workflow["4"] = _n("4", "VAELoader", {"vae_name": paths["audio-vae-fp32"]})

    if task == "ref2va":
        cond_inputs: dict[str, Any] = {
            "clip": ["2", 0],
            "vae": ["3", 0],
            "audio_vae": ["4", 0],
            "prompt": final_prompt,
            "width": width,
            "height": height,
            "length": frames,
            "ref_image_size": spec.get("ref_image_size", "match"),
        }
        ref_images = spec.get("ref_images") or []
        if not 1 <= len(ref_images) <= 9:
            raise SpecError("ref2va needs 1..9 reference images")
        grow: dict[str, Any] = {}
        for i, ref in enumerate(ref_images, start=1):
            nid = f"20{i}"
            workflow[nid] = _n(nid, "LoadImage", {"image": f"{upload_dir}/{ref}"})
            grow[f"ref_image_{i}"] = [nid, 0]
        cond_inputs["ref_images"] = grow
        workflow["5"] = _n("5", "MiniMaxH3ReferenceToVideo", cond_inputs)
    else:
        cond_inputs = {
            "clip": ["2", 0],
            "vae": ["3", 0],
            "prompt": final_prompt,
            "width": width,
            "height": height,
            "length": frames,
        }
        if task in ("i2va", "flf2va"):
            first = spec.get("first_frame")
            if not first:
                raise SpecError(f"{task} requires a first frame image")
            workflow["21"] = _n("21", "LoadImage", {"image": f"{upload_dir}/{first}"})
            cond_inputs["first_frame"] = ["21", 0]
            if task == "flf2va":
                last = spec.get("last_frame")
                if not last:
                    raise SpecError("flf2va requires a last frame image")
                workflow["22"] = _n("22", "LoadImage", {"image": f"{upload_dir}/{last}"})
                cond_inputs["last_frame"] = ["22", 0]
        workflow["5"] = _n("5", "MiniMaxH3ImageToVideo", cond_inputs)

    workflow["6"] = _n("6", "MiniMaxH3SigmaShift", {
        "model": [model_source, 0],
        "shift_video": shift_video,
        "shift_audio": shift_audio,
    })
    workflow["7"] = _n("7", "RandomNoise", {"noise_seed": seed})
    workflow["8"] = _n("8", "KSamplerSelect", {"sampler_name": sampler_name})
    workflow["9"] = _n("9", "BasicScheduler", {
        "model": ["6", 0], "scheduler": scheduler, "steps": steps, "denoise": 1.0,
    })
    workflow["10"] = _n("10", "BasicGuider", {"model": ["6", 0], "positive": ["5", 0]})
    workflow["11"] = _n("11", "SamplerCustomAdvanced", {
        "noise": ["7", 0], "guider": ["10", 0], "sampler": ["8", 0],
        "sigmas": ["9", 0], "latent": ["5", 1],
    })
    workflow["12"] = _n("12", "VAEDecode", {"samples": ["11", 0], "vae": ["3", 0]})
    workflow["13"] = _n("13", "VAEDecodeAudio", {"samples": ["11", 0], "vae": ["4", 0]})

    images_source = ["12", 0]
    upscale_slug = spec.get("upscale_model")
    if upscale_slug:
        if upscale_slug not in paths:
            raise SpecError(f"unknown upscale_model slug {upscale_slug!r}")
        workflow["30"] = _n("30", "UpscaleModelLoader", {"model_name": paths[upscale_slug]})
        workflow["31"] = _n("31", "ImageUpscaleWithModel", {
            "upscale_model": ["30", 0], "image": images_source,
        })
        images_source = ["31", 0]

    workflow["14"] = _n("14", "CreateVideo", {
        "images": images_source, "fps": FPS, "audio": ["13", 0],
    })
    workflow["15"] = _n("15", "SaveVideo", {
        "video": ["14", 0],
        "filename_prefix": f"nocturne/{gen_id}",
        "format": "mp4",
        "codec": {"codec": "h264"},
    })

    meta = {
        "gen_id": gen_id,
        "task": task,
        "quality": quality,
        "checkpoint": checkpoint_slug,
        "width": width,
        "height": height,
        "frames": frames,
        "duration_seconds": round(frames / FPS, 2),
        "fps": FPS,
        "steps": steps,
        "sampler_name": sampler_name,
        "scheduler": scheduler,
        "shift_video": shift_video,
        "shift_audio": shift_audio,
        "seed": seed,
        "prompt_final": final_prompt,
        "upscale_model": upscale_slug,
        "loras": spec.get("loras") or [],
    }
    return workflow, meta
