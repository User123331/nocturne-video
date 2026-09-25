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

import json
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
MIN_FPS = 4
MAX_FPS = 48

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


def frames_for_duration(duration_seconds: float, fps: int = FPS) -> int:
    if not 1.0 <= duration_seconds <= 15.0:
        raise SpecError("duration_seconds must be between 1 and 15")
    if not MIN_FPS <= int(fps) <= MAX_FPS:
        raise SpecError(f"fps must be {MIN_FPS}..{MAX_FPS}")
    return align_frame_count(max(MIN_FRAMES, round(duration_seconds * int(fps))))


def clamp_canvas(width: int, height: int) -> tuple[int, int]:
    def _snap(v: float) -> int:
        return max(CANVAS_MULTIPLE, round(v / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)

    width, height = _snap(int(width)), _snap(int(height))
    ratio = width / height
    # Over the cap, pin the constrained axis and derive the other from the
    # requested ratio, so the frame keeps its shape instead of distorting.
    if width > MAX_WIDTH:
        width, height = MAX_WIDTH, _snap(MAX_WIDTH / ratio)
    if height > MAX_HEIGHT:
        height, width = MAX_HEIGHT, _snap(MAX_HEIGHT * ratio)
    cap_area = MAX_WIDTH * MAX_HEIGHT
    if width * height > cap_area:
        if ratio >= 1:
            width = _snap(math.sqrt(cap_area * ratio))
            height = _snap(width / ratio)
        else:
            height = _snap(math.sqrt(cap_area / ratio))
            width = _snap(height * ratio)
    return width, height


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


# DaSiWa C-MMH3 output-pipeline defaults (Settings note + workflow widgets).
RTX_QUALITY_LEVELS = ("Low", "Medium", "High", "Ultra")
SIMPLE_INTERPOLATIONS = ("Nearest", "Bilinear", "Bicubic", "Area", "Lanczos")
H3_LATENT_UPSCALER_DEFAULT = "latent-upscaler-3d"


def _normalize_upscale(spec: dict[str, Any], paths: dict[str, str],
                       width: int, height: int) -> dict[str, Any] | None:
    """Validate the four DaSiWa output-pipeline upscale modes into one dict."""
    raw = spec.get("upscale")
    if raw is None and spec.get("upscale_model"):
        raw = {"mode": "model", "model": spec.get("upscale_model")}
    if not raw:
        return None
    if not isinstance(raw, dict):
        raise SpecError("upscale must be an object")
    mode = raw.get("mode")
    if mode == "model":
        slug = raw.get("model", "")
        if slug not in paths:
            raise SpecError(f"unknown upscale model slug {slug!r}")
        return {"mode": "model", "model": slug}
    if mode == "simple":
        out = {
            "mode": "simple",
            "multiplier": float(raw.get("multiplier", 2.0)),
            "interpolation": raw.get("interpolation", "Lanczos"),
            "divisible_by": int(raw.get("divisible_by", 16)),
        }
        if not 0.01 <= out["multiplier"] <= 16.0:
            raise SpecError("simple upscale multiplier must be 0.01..16")
        if out["interpolation"] not in SIMPLE_INTERPOLATIONS:
            raise SpecError(f"simple interpolation must be one of {SIMPLE_INTERPOLATIONS}")
        if not 1 <= out["divisible_by"] <= 512:
            raise SpecError("simple divisible_by must be 1..512")
        return out
    if mode == "rtx":
        out = {
            "mode": "rtx",
            "scale": float(raw.get("scale", 2.0)),
            "upscale_quality": raw.get("upscale_quality", "Ultra"),
            "denoise": bool(raw.get("denoise", False)),
            "denoise_quality": raw.get("denoise_quality", "Ultra"),
            "deblur": bool(raw.get("deblur", False)),
            "deblur_quality": raw.get("deblur_quality", "Ultra"),
            "divisible_by": str(raw.get("divisible_by", "8")),
        }
        if not 1.0 <= out["scale"] <= 4.0:
            raise SpecError("rtx scale must be 1.0..4.0")
        for key in ("upscale_quality", "denoise_quality", "deblur_quality"):
            if out[key] not in RTX_QUALITY_LEVELS:
                raise SpecError(f"rtx {key} must be one of {RTX_QUALITY_LEVELS}")
        return out
    if mode == "h3_latent":
        # The 3D latent upscaler re-samples the AV latent: target pixel size
        # snaps to the VAE's 16x grid. Default target is a straight 2x.
        target_w = int(raw.get("target_width", width * 2))
        target_h = int(raw.get("target_height", height * 2))
        precision = raw.get("precision", "fp16")
        if precision not in ("fp16", "fp32", "bf16"):
            raise SpecError("h3_latent precision must be fp16, fp32, or bf16")
        return {
            "mode": "h3_latent",
            "model": raw.get("model", H3_LATENT_UPSCALER_DEFAULT),
            "target_width": (target_w // 16) * 16,
            "target_height": (target_h // 16) * 16,
            "precision": precision,
        }
    raise SpecError("upscale mode must be one of: off, model, simple, rtx, h3_latent")


def _check_cache(spec: dict[str, Any]) -> dict[str, Any] | None:
    cache = spec.get("cache")
    if not cache:
        return None
    if not isinstance(cache, dict):
        raise SpecError("cache must be an object")
    out = {
        "reuse_threshold": float(cache.get("reuse_threshold", 0.05)),
        "start_percent": float(cache.get("start_percent", 0.15)),
        "end_percent": float(cache.get("end_percent", 0.90)),
        "max_steps": int(cache.get("max_steps", 2)),
    }
    if not 0.0 <= out["reuse_threshold"] <= 1.0:
        raise SpecError("cache reuse_threshold must be 0..1")
    if not 0.0 <= out["start_percent"] <= out["end_percent"] <= 1.0:
        raise SpecError("cache start_percent must not exceed end_percent")
    if not 1 <= out["max_steps"] <= 10:
        raise SpecError("cache max_steps must be 1..10")
    return out


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
    # "custom" means the caller sent explicit sampling values; the "quality"
    # preset still supplies whatever was left unset.
    preset = PRESETS.get(quality if quality in PRESETS else "quality")
    if quality not in PRESETS and quality != "custom":
        raise SpecError("quality must be 'quality', 'turbo', or 'custom'")

    checkpoint_slug = spec.get("checkpoint", "dasiwa-hybrid-v2-int8")
    if checkpoint_slug not in paths:
        raise SpecError(f"unknown checkpoint slug {checkpoint_slug!r}")

    duration = float(spec.get("duration_seconds", 5))
    fps = int(spec.get("fps", FPS))
    frames = frames_for_duration(duration, fps)
    width, height = clamp_canvas(spec.get("width", DEFAULT_WIDTH), spec.get("height", DEFAULT_HEIGHT))

    prompt = spec.get("prompt") or {}
    final_prompt = assemble_prompt(task, prompt, frames / fps)

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

    upscale = _normalize_upscale(spec, paths, width, height)
    cache = _check_cache(spec)
    watermark = spec.get("watermark") or None
    if watermark:
        if not isinstance(watermark, dict) or not (watermark.get("image") or "").strip():
            raise SpecError("watermark needs an uploaded image")
        if watermark.get("position", "bottom-right") not in (
                "bottom-right", "bottom-left", "top-right", "top-left", "center"):
            raise SpecError("watermark position is invalid")
        if not 0.01 <= float(watermark.get("scale", 0.12)) <= 1.0:
            raise SpecError("watermark scale must be 0.01..1.0")
        if not 0.0 <= float(watermark.get("transparency", 0.35)) <= 1.0:
            raise SpecError("watermark opacity must be 0..1")

    workflow: dict[str, Any] = {}

    workflow["1"] = _n("1", "UNETLoader", {
        "unet_name": paths[checkpoint_slug], "weight_dtype": "default",
    })

    workflow["2"] = _n("2", "CLIPLoader", {
        "clip_name": paths[te_slug], "type": "minimax", "device": "default",
    })
    workflow["3"] = _n("3", "VAELoader", {"vae_name": paths["video-vae-fp16"]})
    workflow["4"] = _n("4", "VAELoader", {"vae_name": paths["audio-vae-fp32"]})

    # LoRA stack via DaSiWa's Advanced LoRA Loader (the workflow's
    # DaSiWa_LTX2LoraLoader): one node, JSON stack of {on, lora, str, vs, as}.
    # model_type=Basic applies every tensor universally — vs/as only separate
    # on LTX-2.3, so for H3 the master strength is what matters.
    model_source = ["1", 0]
    clip_source = ["2", 0]
    loras = [l for l in (spec.get("loras") or [])
             if (l.get("name") or "").strip() not in ("", "None")]
    if loras:
        if len(loras) > 10:
            raise SpecError("at most 10 LoRAs per job")
        stack = []
        for lora in loras:
            name = (lora.get("name") or "").strip()
            strength = float(lora.get("strength", 1.0))
            if not -4.0 <= strength <= 4.0:
                raise SpecError("lora strength must be within -4..4")
            stack.append({"on": True, "lora": name, "str": strength, "vs": 1, "as": 1})
        workflow["16"] = _n("16", "DaSiWa_LTX2LoraLoader", {
            "model": model_source,
            "clip": clip_source,
            "stack_data": json.dumps(stack),
            "model_type": "Basic",
            "use_cache": False,
        })
        model_source = ["16", 0]
        clip_source = ["16", 1]

    if task == "ref2va":
        cond_inputs: dict[str, Any] = {
            "clip": clip_source,
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
            "clip": clip_source,
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
        "model": model_source,
        "shift_video": shift_video,
        "shift_audio": shift_audio,
    })

    # Model chain mirrors DaSiWa's Settings subgraph order: LoRA -> sigma
    # shift -> block cache -> chunked feed-forward.
    sampled_model = ["6", 0]
    if cache:
        workflow["18"] = _n("18", "MiniMaxH3Cache", {
            "model": sampled_model,
            "reuse_threshold": cache["reuse_threshold"],
            "start_percent": cache["start_percent"],
            "end_percent": cache["end_percent"],
            "max_steps": cache["max_steps"],
            "device": "auto",
            "verbose": False,
        })
        sampled_model = ["18", 0]
    if spec.get("chunk_ffn"):
        chunks = int(spec.get("chunk_count", 4))
        if not 1 <= chunks <= 64:
            raise SpecError("chunk_count must be 1..64")
        workflow["17"] = _n("17", "MiniMaxChunkFeedForward", {
            "model": sampled_model, "chunks": chunks, "seq_threshold": 4096,
        })
        sampled_model = ["17", 0]

    workflow["7"] = _n("7", "RandomNoise", {"noise_seed": seed})
    workflow["8"] = _n("8", "KSamplerSelect", {"sampler_name": sampler_name})
    workflow["9"] = _n("9", "BasicScheduler", {
        "model": sampled_model, "scheduler": scheduler, "steps": steps, "denoise": 1.0,
    })
    workflow["10"] = _n("10", "BasicGuider", {"model": sampled_model, "conditioning": ["5", 0]})
    workflow["11"] = _n("11", "SamplerCustomAdvanced", {
        "noise": ["7", 0], "guider": ["10", 0], "sampler": ["8", 0],
        "sigmas": ["9", 0], "latent_image": ["5", 1],
    })

    # H3-latent upscale re-samples the denoised AV latent through the 3D latent
    # upscaler (MMH3UltimateUpscale), so both decode branches read its output.
    latent_source = ["11", 0]
    if upscale and upscale["mode"] == "h3_latent":
        workflow["40"] = _n("40", "MMH3LatentUpscaleWithModelParams", {
            "model_name": paths[upscale["model"]],
            "width": upscale["target_width"],
            "height": upscale["target_height"],
            "device": "cuda",
            "precision": upscale["precision"],
            "offload_model": True,
        })
        workflow["41"] = _n("41", "MMH3TemporalSplitParams", {
            "chunk_length": 85, "temporal_overlap": 17, "anchor_strength": 0.999,
        })
        # DaSiWa's spatial defaults: equal 2x3 tile grid over the upscaled
        # frame, 128px overlaps, 64px fades, later-side overlap wins.
        workflow["42"] = _n("42", "MMH3SpatialSplitParams", {
            "upscale_width": upscale["target_width"],
            "upscale_height": upscale["target_height"],
            "tile_size_mode": "rows_cols",
            "tile_width": 512, "tile_height": 512,
            "grid_rows": 2, "grid_cols": 3,
            "spatial_w_overlap": 128, "spatial_h_overlap": 128,
            "fade_width": 64, "fade_height": 64,
            "min_tile_size": 256,
            "overlap_mode": "later", "overlap_blend": "linear",
            "masked_area_noise": 0.05, "brightness_match": True,
            "dynamic_fade": "widening", "dynamic_fade_min": 32,
        })
        workflow["43"] = _n("43", "MMH3UltimateUpscale", {
            "model": sampled_model, "conditioning": ["5", 0], "latent": ["11", 0],
            "noise": ["7", 0], "sampler": ["8", 0], "sigmas": ["9", 0],
            "cfg": 1.0,
            "latent_upscale_param": ["40", 0],
            "temporal_split_param": ["41", 0],
            "spatial_split_param": ["42", 0],
        })
        latent_source = ["43", 0]

    workflow["12"] = _n("12", "VAEDecode", {"samples": latent_source, "vae": ["3", 0]})
    workflow["13"] = _n("13", "VAEDecodeAudio", {"samples": latent_source, "vae": ["4", 0]})

    images_source = ["12", 0]
    fps_out = fps
    if spec.get("frame_interpolation"):
        mult = int(spec.get("interpolation_multiplier", 2))
        if not 2 <= mult <= 16:
            raise SpecError("interpolation_multiplier must be 2..16")
        workflow["32"] = _n("32", "FrameInterpolationModelLoader", {
            "model_name": spec.get("interpolation_model", "rife_v4.26.safetensors"),
        })
        workflow["33"] = _n("33", "FrameInterpolate", {
            "interp_model": ["32", 0], "images": images_source, "multiplier": mult,
        })
        images_source = ["33", 0]
        fps_out = fps * mult

    if upscale and upscale["mode"] == "model":
        workflow["30"] = _n("30", "UpscaleModelLoader", {
            "model_name": paths[upscale["model"]],
        })
        workflow["31"] = _n("31", "ImageUpscaleWithModel", {
            "upscale_model": ["30", 0], "image": images_source,
        })
        images_source = ["31", 0]
    elif upscale and upscale["mode"] == "simple":
        workflow["34"] = _n("34", "DaSiWa_TorchResize", {
            "image": images_source,
            "size_mode": "Multiplier",
            "aspect_mode": "Fit",
            "target_width": 1920, "target_height": 1080,
            "scale_multiplier": upscale["multiplier"],
            "interpolation": upscale["interpolation"],
            "gamma_correct": True,
            "divisible_by": upscale["divisible_by"],
            "pad_color": "0, 0, 0",
            "crop_position": "center",
            "batch_size": 0,
            "max_batch_megapixels": 16.0,
            "cache_size": 64,
        })
        images_source = ["34", 0]
    elif upscale and upscale["mode"] == "rtx":
        workflow["35"] = _n("35", "DaSiWa_RTX_UpscalerRefiner", {
            "images": images_source,
            "denoise": upscale["denoise"],
            "denoise_quality": upscale["denoise_quality"],
            "deblur": upscale["deblur"],
            "deblur_quality": upscale["deblur_quality"],
            "upscale": "VSR" if upscale["scale"] > 1.0 else "Off",
            "upscale_quality": upscale["upscale_quality"],
            "resize_type": "Scale",
            "scale": upscale["scale"],
            "megapixels": 2.0,
            "width": 1920, "height": 1080,
            "divisible_by": upscale["divisible_by"],
            "ratio_preset": "16:9",
            "resize_method": "Center Crop (Fill)",
            "device_id": 0,
            "empty_cache": False,
            "use_mmap": False,
            "auto_unload_models": True,
        })
        images_source = ["35", 0]

    if watermark:
        # watermark_path is a filename combo relative to ComfyUI's input dir;
        # the handler stages the uploaded PNG under the job's upload folder.
        workflow["37"] = _n("37", "DaSiWa_Watermark", {
            "images": images_source,
            "watermark_path": f"{upload_dir}/{watermark['image']}",
            "position": watermark.get("position", "bottom-right"),
            "scale": float(watermark.get("scale", 0.12)),
            "resampling": "bicubic",
            "transparency": float(watermark.get("transparency", 0.35)),
            "rotation": 0,
            "padding_x": int(watermark.get("padding_x", 20)),
            "padding_y": int(watermark.get("padding_y", 20)),
            "optical_padding": False,
            "optical_strength": 0.4,
            "random_switches": 3,
            "fade": False,
            "fade_margin": 0.1,
            "randomize_position": False,
            "random_seed": 0,
        })
        images_source = ["37", 0]

    workflow["14"] = _n("14", "CreateVideo", {
        "images": images_source, "fps": fps_out, "audio": ["13", 0],
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
        "generation_fps": fps,
        "duration_seconds": round(frames / fps, 2),
        "fps": fps_out,
        "frame_interpolation": bool(spec.get("frame_interpolation")),
        "interpolation_multiplier": int(spec.get("interpolation_multiplier", 2))
        if spec.get("frame_interpolation") else None,
        "chunk_ffn": bool(spec.get("chunk_ffn")),
        "steps": steps,
        "sampler_name": sampler_name,
        "scheduler": scheduler,
        "shift_video": shift_video,
        "shift_audio": shift_audio,
        "seed": seed,
        "prompt_final": final_prompt,
        "upscale": upscale,
        "cache": cache,
        "watermark": bool(watermark),
        "loras": spec.get("loras") or [],
    }
    return workflow, meta
