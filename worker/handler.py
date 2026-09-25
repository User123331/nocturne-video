"""Runpod serverless handler for Nocturne Video (MiniMax H3 video+audio).

Thin wrapper over the maintained runpod/worker-comfyui handler: this module
validates the high-level job spec, stages uploaded media into ComfyUI's input
folder, builds the native H3 API graph (workflow_factory), delegates execution
to the upstream handler, then mirrors the finished MP4 to the network volume
and to the Runpod S3 bucket so the local Nocturne Video app can sync it.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import os
import shutil
import sys
import time
import uuid
from pathlib import Path

APP_ROOT = Path(os.getenv("APP_ROOT", "/opt/nocturne-video"))
# Resolve sibling modules from this file's own directory first: the worker dir
# is authoritative for asset_manager/workflow_factory, and this keeps imports
# working even if APP_ROOT or the working directory is unexpected.
_HERE = Path(__file__).resolve().parent
for _candidate in (_HERE, APP_ROOT / "worker"):
    if (_candidate / "asset_manager.py").is_file():
        sys.path.insert(0, str(_candidate))
        break
sys.path.insert(0, "/")

import handler_upstream  # type: ignore  # provided by the base image
import runpod  # type: ignore

import asset_manager
import workflow_factory

VOLUME_ROOT = Path(os.getenv("VOLUME_ROOT", "/runpod-volume"))
COMFYUI_INPUT = Path(os.getenv("COMFYUI_INPUT_DIR", "/comfyui/input"))
COMFYUI_OUTPUT = Path(os.getenv("COMFYUI_OUTPUT_DIR", "/comfyui/output"))
MAX_UPLOAD_BYTES = 32 * 1024 * 1024
ALLOWED_UPLOAD_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".wav", ".mp3", ".flac", ".m4a", ".ogg"}

DEFAULT_REQUIRED = [
    "qwen3vl-32b-nvfp4-awq",
    "video-vae-fp16",
    "audio-vae-fp32",
]


# ---------------------------------------------------------------------------
# Runpod S3 upload support (adapted from the sibling Deck's worker patch).
# ---------------------------------------------------------------------------

def _configure_rp_upload() -> None:
    try:
        import runpod.serverless.utils.rp_upload as rp_upload
        from boto3 import session as boto_session
        from botocore.config import Config as boto_config
    except Exception:  # noqa: BLE001 — S3 upload stays optional at import time
        return

    def get_boto_client(bucket_creds=None):
        creds = bucket_creds or {}
        endpoint_url = creds.get("endpointUrl") or os.environ.get("BUCKET_ENDPOINT_URL")
        access_key_id = creds.get("accessId") or os.environ.get("BUCKET_ACCESS_KEY_ID")
        secret = creds.get("accessSecret") or os.environ.get("BUCKET_SECRET_ACCESS_KEY")
        if not (endpoint_url and access_key_id and secret):
            raise ValueError("missing bucket credentials")
        region = (
            creds.get("region")
            or os.environ.get("BUCKET_REGION")
            or os.environ.get("AWS_REGION")
        )
        session = boto_session.Session()
        client = session.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret,
            region_name=region,
            config=boto_config(signature_version="s3v4", retries={"max_attempts": 3, "mode": "standard"}),
        )
        return client

    rp_upload.get_boto_client = get_boto_client


_configure_rp_upload()


def _bucket_env() -> dict[str, str] | None:
    needed = ("BUCKET_ENDPOINT_URL", "BUCKET_ACCESS_KEY_ID", "BUCKET_SECRET_ACCESS_KEY", "BUCKET_BUCKET_NAME")
    values = {k: os.environ.get(k, "") for k in needed}
    if all(values.values()):
        return values
    return None


def _s3_upload(path: Path, key: str) -> dict:
    """Upload one output file to the Runpod S3 bucket; returns an S3 descriptor."""
    import boto3
    from botocore.config import Config as boto_config

    env = _bucket_env()
    assert env is not None
    client = boto3.client(
        "s3",
        endpoint_url=env["BUCKET_ENDPOINT_URL"],
        aws_access_key_id=env["BUCKET_ACCESS_KEY_ID"],
        aws_secret_access_key=env["BUCKET_SECRET_ACCESS_KEY"],
        region_name=os.environ.get("BUCKET_REGION", "US-NE-1"),
        config=boto_config(signature_version="s3v4", retries={"max_attempts": 3, "mode": "standard"}),
    )
    from boto3.s3.transfer import TransferConfig
    transfer = TransferConfig(
        multipart_threshold=64 * 1024 * 1024,
        multipart_chunksize=64 * 1024 * 1024,
        max_concurrency=8,
        use_threads=True,
    )
    client.upload_file(str(path), env["BUCKET_BUCKET_NAME"], key, Config=transfer)
    return {"bucket": env["BUCKET_BUCKET_NAME"], "key": key}


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------

class JobError(ValueError):
    """User-facing errors returned as a normal failed job response."""


def _write_upload(data_b64: str, dest_dir: Path, stem: str, kind_hint: str) -> str:
    try:
        raw = base64.b64decode(data_b64, validate=False)
    except (binascii.Error, ValueError) as exc:
        raise JobError(f"{kind_hint}: invalid base64 payload") from exc
    if not raw:
        raise JobError(f"{kind_hint}: empty upload")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise JobError(f"{kind_hint}: upload exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MiB limit")
    suffix = Path(kind_hint).suffix.lower() or ".png"
    if suffix not in ALLOWED_UPLOAD_SUFFIXES:
        raise JobError(f"{kind_hint}: unsupported file type")
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{stem}{suffix}"
    dest.write_bytes(raw)
    return dest.name


def _stage_uploads(spec: dict, upload_dir: Path) -> dict:
    staged: dict[str, str] = {}
    for field in ("first_frame", "last_frame"):
        b64 = spec.get(f"{field}_b64")
        if b64:
            staged[field] = _write_upload(b64, upload_dir, field, f"{field}.png")
    ref_images = spec.get("ref_images_b64") or []
    staged_refs = []
    for i, b64 in enumerate(ref_images, start=1):
        staged_refs.append(_write_upload(b64, upload_dir, f"ref_image_{i}", f"ref_image_{i}.png"))
    if staged_refs:
        staged["ref_images"] = staged_refs
    ref_audios = spec.get("ref_audios_b64") or []
    staged_audios = []
    for i, b64 in enumerate(ref_audios, start=1):
        staged_audios.append(_write_upload(b64, upload_dir, f"ref_audio_{i}", f"ref_audio_{i}.mp3"))
    if staged_audios:
        staged["ref_audios"] = staged_audios
    watermark = spec.get("watermark") or {}
    if isinstance(watermark, dict) and watermark.get("image_b64"):
        staged["watermark_image"] = _write_upload(
            watermark["image_b64"], upload_dir, "watermark", "watermark.png")
    return staged


def _apply_staged_to_spec(spec: dict, staged: dict, upload_rel: str) -> None:
    """Rewrite spec so the factory references staged filenames under upload_rel."""
    spec = dict(spec)
    if staged.get("first_frame"):
        spec["first_frame"] = staged["first_frame"]
    if staged.get("last_frame"):
        spec["last_frame"] = staged["last_frame"]
    if staged.get("ref_images"):
        spec["ref_images"] = staged["ref_images"]
    if staged.get("ref_audios"):
        spec["ref_audios"] = staged["ref_audios"]
    if staged.get("watermark_image"):
        spec["watermark"] = {**(spec.get("watermark") or {}),
                             "image": staged["watermark_image"]}
    return spec


def _paths_for_graph(paths_abs: dict[str, str]) -> dict[str, str]:
    """Map asset slugs to the loader names ComfyUI expects.

    Each loader node (UNETLoader, VAELoader, CLIPLoader, ...) already resolves
    within its own models/<kind>/ directory, so the value must be the path
    *inside* that directory: "MiniMaxH3/foo.safetensors", not
    "diffusion_models/MiniMaxH3/foo.safetensors". Prefixing the kind makes the
    value fail the node's own filename list and the workflow is rejected with
    "value_not_in_list" before anything runs.
    """
    manifest = {a["slug"]: a for a in asset_manager.load_manifest()}
    out = {}
    for slug, _abs in paths_abs.items():
        asset = manifest.get(slug)
        if asset is None:
            continue
        out[slug] = asset_manager.comfy_relative_name(asset)
    return out


def _find_outputs(gen_id: str) -> list[Path]:
    out_dir = COMFYUI_OUTPUT / "nocturne"
    if not out_dir.is_dir():
        return []
    return sorted(out_dir.glob(f"{gen_id}*.mp4"))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024 * 8), b""):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

def handler(job: dict) -> dict:
    job_input = job.get("input") or {}
    gen_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    allow_downloads = os.getenv("ALLOW_MODEL_DOWNLOADS", "0") == "1"
    upload_rel = f"nocturne/{gen_id}"
    upload_dir = COMFYUI_INPUT / upload_rel

    try:
        if job_input.get("task") == "healthcheck":
            # Verify the whole manifest so one healthcheck proves the volume is
            # complete, instead of discovering gaps one job at a time.
            all_slugs = [a["slug"] for a in asset_manager.load_manifest()]
            started = time.time()
            try:
                verified = asset_manager.ensure_assets(all_slugs, allow_downloads=allow_downloads)
                failures = []
            except Exception as exc:  # noqa: BLE001 — report per-asset detail
                verified, failures = {}, [str(exc)]
            asset_manager.expose()
            return {
                "status": "ok" if not failures else "error",
                "mode": "healthcheck",
                "assets_verified": sorted(verified.keys()),
                "assets_failed": failures,
                "seconds": round(time.time() - started, 1),
                "refresh_workers": True,
            }

        spec = dict(job_input)
        staged = _stage_uploads(spec, upload_dir)
        spec = _apply_staged_to_spec(spec, staged, upload_rel)

        checkpoint = spec.get("checkpoint", "dasiwa-hybrid-v2-int8")
        required = DEFAULT_REQUIRED + [checkpoint]
        upscale = spec.get("upscale") or {}
        if not upscale and spec.get("upscale_model"):
            upscale = {"mode": "model", "model": spec.get("upscale_model")}
        mode = upscale.get("mode") if isinstance(upscale, dict) else None
        if mode == "rtx":
            # DaSiWa's RTX node imports the NVIDIA VFX bindings lazily; refuse
            # here (before GPU time is spent) rather than inside the graph run.
            try:
                import nvvfx  # noqa: F401
            except ImportError:
                raise JobError(
                    "RTX upscaling needs the NVIDIA Video Effects SDK (nvvfx), "
                    "which this worker image does not include. Use the Model, "
                    "Simple, or H3 Latent upscale mode instead.")
        if mode in ("model", "h3_latent"):
            slug = upscale.get("model") or (
                workflow_factory.H3_LATENT_UPSCALER_DEFAULT if mode == "h3_latent" else "")
            if slug:
                required.append(slug)
        paths_abs = asset_manager.ensure_assets(required, allow_downloads=allow_downloads)

        graph, meta = workflow_factory.build_graph(
            spec, paths=_paths_for_graph(paths_abs), gen_id=gen_id, upload_dir=upload_rel)

        upstream_job = {
            "id": job.get("id", gen_id),
            "input": {"workflow": graph},
        }
        upstream_result = handler_upstream.handler(upstream_job) or {}

        videos = []
        for mp4 in _find_outputs(gen_id):
            digest = _sha256(mp4)
            volume_dir = VOLUME_ROOT / "outputs" / gen_id
            volume_dir.mkdir(parents=True, exist_ok=True)
            volume_copy = volume_dir / mp4.name
            if not volume_copy.exists():
                shutil.copy2(mp4, volume_copy)
            entry = {
                "filename": mp4.name,
                "bytes": mp4.stat().st_size,
                "sha256": digest,
                "volume_path": str(volume_copy),
            }
            if _bucket_env():
                day = time.strftime("%Y/%m/%d")
                key = f"outputs/{day}/{gen_id}/{mp4.name}"
                try:
                    entry.update(_s3_upload(mp4, key))
                    entry["s3_uri"] = f"s3://{entry['bucket']}/{entry['key']}"
                except Exception as exc:  # noqa: BLE001 — S3 down shouldn't kill the job
                    entry["s3_error"] = str(exc)
            videos.append(entry)

        provenance = {
            "gen_id": gen_id,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "meta": meta,
            "videos": videos,
            "worker_image": os.environ.get("NOCTURNE_IMAGE_TAG", "dev"),
            "upstream_status": "completed" if videos else "no_output",
        }
        prov_dir = VOLUME_ROOT / "provenance" / "video"
        prov_dir.mkdir(parents=True, exist_ok=True)
        (prov_dir / f"{gen_id}.json").write_text(json.dumps(provenance, indent=2))

        if not videos:
            return {
                "status": "error",
                "error": "worker produced no video output",
                "gen_id": gen_id,
                "upstream_result": upstream_result,
            }

        shutil.rmtree(upload_dir, ignore_errors=True)
        for mp4 in _find_outputs(gen_id):
            mp4.unlink(missing_ok=True)

        return {
            "status": "ok",
            "gen_id": gen_id,
            "meta": meta,
            "videos": videos,
            "provenance": provenance,
        }
    except workflow_factory.SpecError as exc:
        return {"status": "error", "error": str(exc), "retryable": False}
    except JobError as exc:
        return {"status": "error", "error": str(exc), "retryable": False}
    except Exception as exc:  # noqa: BLE001 — surfaced to the caller either way
        return {
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "gen_id": gen_id,
            "retryable": True,
        }
    finally:
        shutil.rmtree(upload_dir, ignore_errors=True)


if __name__ == "__main__":
    print("nocturne-video: starting Runpod handler")
    runpod.serverless.start({"handler": handler})
