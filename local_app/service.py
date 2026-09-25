"""Job pipeline: submit, poll, sync finished videos into the local gallery."""

from __future__ import annotations

import json
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from . import config, database, runpod_client, s3_sync

# Director prompt sections per mode, mirroring the DaSiWa C-MMH3 Director.
PROMPT_SECTIONS = {
    "t2va": ["integrated_multimodal_description", "overall_soundscape", "non_diegetic_music"],
    "i2va": ["integrated_multimodal_description", "overall_soundscape", "non_diegetic_music"],
    "flf2va": ["integrated_multimodal_description", "overall_soundscape", "non_diegetic_music"],
    "ref2va": ["subject_definitions", "summary", "retention_analysis",
               "detailed_description", "overall_soundscape", "non_diegetic_music"],
}

TASKS = ("t2va", "i2va", "flf2va", "ref2va")


class ServiceError(ValueError):
    pass


class Service:
    def __init__(self, db: database.Database):
        self.db = db
        self.s3 = s3_sync.RunpodS3.from_config()
        self._poller_stop = threading.Event()
        self._poller = threading.Thread(target=self._poll_loop, daemon=True, name="poller")
        self._poller.start()

    def shutdown(self) -> None:
        self._poller_stop.set()

    # -- submission -----------------------------------------------------------

    def submit(self, payload: dict) -> dict:
        task = payload.get("task")
        if task not in TASKS:
            raise ServiceError(f"task must be one of {TASKS}")
        quality = payload.get("quality", "quality")
        if quality not in ("quality", "turbo"):
            raise ServiceError("quality must be 'quality' or 'turbo'")

        prompt = payload.get("prompt") or {}
        if not isinstance(prompt, dict):
            raise ServiceError("prompt must be an object")
        prompt_text = " | ".join(
            str(v).strip() for v in prompt.values() if str(v).strip())[:400]

        files = payload.get("files") or {}
        spec: dict = {
            "task": task,
            "quality": quality,
            "prompt": {k: str(v) for k, v in prompt.items()},
            "duration_seconds": float(payload.get("duration_seconds", 5)),
        }
        for key in ("width", "height", "seed", "steps", "shift_video", "shift_audio",
                    "sampler_name", "scheduler", "ref_image_size", "upscale_model",
                    "frame_interpolation", "interpolation_multiplier", "chunk_ffn",
                    "chunk_count", "checkpoint", "fps"):
            if payload.get(key) not in (None, ""):
                spec[key] = payload[key]
        # Structured blocks pass through for the worker to validate; the worker
        # factory owns the exact value ranges, so the local app only checks type.
        for key in ("upscale", "cache", "watermark"):
            value = payload.get(key)
            if value:
                if not isinstance(value, dict):
                    raise ServiceError(f"{key} must be an object")
                spec[key] = value
        if payload.get("loras"):
            spec["loras"] = [
                {"name": str(l.get("name", "")).strip(), "strength": float(l.get("strength", 1.0))}
                for l in payload["loras"] if str(l.get("name", "")).strip()
            ][:10]
        if files.get("first_frame"):
            spec["first_frame_b64"] = files["first_frame"]
        if files.get("last_frame"):
            spec["last_frame_b64"] = files["last_frame"]
        if files.get("ref_images"):
            spec["ref_images_b64"] = files["ref_images"][:9]
        if files.get("ref_audios"):
            spec["ref_audios_b64"] = files["ref_audios"][:3]

        if task in ("i2va", "flf2va") and not spec.get("first_frame_b64"):
            raise ServiceError(f"{task} needs a first frame image")
        if task == "flf2va" and not spec.get("last_frame_b64"):
            raise ServiceError("flf2va needs a last frame image")
        if task == "ref2va" and not spec.get("ref_images_b64"):
            raise ServiceError("ref2va needs at least one reference image")

        job_id = f"nv-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        result = runpod_client.run(spec)
        stored_spec = dict(spec)
        if isinstance(stored_spec.get("watermark"), dict):
            # The uploaded watermark bytes are large and never reused; keep the
            # parameter block only.
            stored_spec["watermark"] = {k: v for k, v in spec["watermark"].items()
                                        if k != "image_b64"}
            stored_spec["watermark"]["image"] = "<uploaded>"
        self.db.insert_job(job_id, result["id"], task, quality, stored_spec, prompt_text)
        return {"job_id": job_id, "runpod_job_id": result["id"]}

    def healthcheck(self) -> dict:
        result = runpod_client.run({"task": "healthcheck"})
        job_id = f"nv-hc-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        self.db.insert_job(job_id, result["id"], "healthcheck", "", {"task": "healthcheck"}, "")
        return {"job_id": job_id, "runpod_job_id": result["id"]}

    def cancel(self, job_id: str) -> dict:
        job = self.db.get_job(job_id)
        if not job:
            raise ServiceError(f"unknown job {job_id}")
        if job.get("runpod_job_id") and job["status"] in ("QUEUED", "RUNNING"):
            try:
                runpod_client.cancel(job["runpod_job_id"])
            except runpod_client.RunpodClientError as exc:
                raise ServiceError(str(exc)) from exc
        self.db.update_job(job_id, status="CANCELLED", error="cancelled by user")
        return {"job_id": job_id, "status": "CANCELLED"}

    # -- polling ---------------------------------------------------------------

    def _poll_loop(self) -> None:
        while not self._poller_stop.wait(4.0):
            try:
                for job in self.db.active_jobs():
                    if job["status"] == "QUEUED" and not job.get("runpod_job_id"):
                        continue
                    self._poll_one(job)
            except Exception as exc:  # noqa: BLE001 — poller must survive errors
                print(f"service: poll error: {exc}", flush=True)

    def _poll_one(self, job: dict) -> None:
        try:
            state = runpod_client.status(job["runpod_job_id"])
        except runpod_client.RunpodClientError as exc:
            self.db.update_job(job["id"], status="FAILED", error=str(exc))
            return
        status = state.get("status")
        if status in ("IN_QUEUE", "IN_PROGRESS", None):
            if status and status != job["status"]:
                self.db.update_job(job["id"], status="RUNNING")
            return
        if status == "COMPLETED":
            self._complete(job, state.get("output") or {})
        elif status == "FAILED":
            self.db.update_job(job["id"], status="FAILED",
                               error=str(state.get("error") or "worker failed"))
        elif status == "CANCELLED":
            self.db.update_job(job["id"], status="CANCELLED", error="cancelled on Runpod")

    def _complete(self, job: dict, output: dict) -> None:
        if output.get("status") == "error":
            self.db.update_job(job["id"], status="FAILED",
                               error=str(output.get("error") or "worker error"))
            return
        # A healthcheck verifies the volume and produces no media by design:
        # report what it verified instead of demanding a video from it.
        if job.get("task") == "healthcheck" or output.get("mode") == "healthcheck":
            verified = output.get("assets_verified") or []
            failed = output.get("assets_failed") or []
            summary = f"{len(verified)} assets verified"
            if failed:
                summary += f"; {len(failed)} failed"
            self.db.update_job(job["id"], status="COMPLETED", error=summary)
            return
        videos = output.get("videos") or []
        if not videos:
            self.db.update_job(job["id"], status="FAILED",
                               error=f"no video in output: {json.dumps(output)[:300]}")
            return
        video = videos[0]
        meta = output.get("meta") or {}
        gen_id = output.get("gen_id") or job["id"]
        self.db.update_job(job["id"], status="COMPLETED", gen_id=gen_id)
        self.db.insert_generation(
            gen_id=gen_id,
            job_id=job["id"],
            task=meta.get("task", job["task"]),
            quality=meta.get("quality", job.get("quality", "")),
            checkpoint=meta.get("checkpoint", ""),
            prompt_text=job["prompt_text"],
            meta_json=json.dumps(meta),
            video_path=None,
            s3_uri=video.get("s3_uri"),
            sha256=video.get("sha256"),
            bytes=video.get("bytes"),
            width=meta.get("width"),
            height=meta.get("height"),
            frames=meta.get("frames"),
            duration=meta.get("duration_seconds"),
            seed=str(meta.get("seed", "")),
        )
        if self.s3 and video.get("key"):
            self.sync_generation(gen_id)

    # -- sync -------------------------------------------------------------------

    def sync_generation(self, gen_id: str) -> dict:
        gen = self.db.get_generation(gen_id)
        if not gen:
            raise ServiceError(f"unknown generation {gen_id}")
        if not self.s3:
            raise ServiceError("S3 credentials not configured (Keychain or NOCTURNE_S3_*)")
        if not gen.get("s3_uri"):
            raise ServiceError("generation has no S3 object yet")
        key = gen["s3_uri"].split("/", 3)[3] if gen["s3_uri"].startswith("s3://") else None
        if not key:
            raise ServiceError(f"unparseable s3_uri {gen['s3_uri']}")
        day = gen["created_at"][:10]
        dest = Path(config.OUTPUT_DIR) / day / f"{gen_id}.mp4"
        if dest.exists() and dest.stat().st_size == (gen.get("bytes") or -1):
            self.db.update_generation(gen_id, video_path=str(dest))
            return {"gen_id": gen_id, "video_path": str(dest), "already": True}
        self.s3.download(key, dest)
        self.db.update_generation(gen_id, video_path=str(dest))
        return {"gen_id": gen_id, "video_path": str(dest)}

    def sync_missing(self) -> dict:
        results = {"synced": [], "failed": []}
        for gen in self.db.generations_missing_files():
            try:
                self.sync_generation(gen["gen_id"])
                results["synced"].append(gen["gen_id"])
            except Exception as exc:  # noqa: BLE001
                results["failed"].append({"gen_id": gen["gen_id"], "error": str(exc)})
        return results

    # -- status -------------------------------------------------------------------

    def s3_status(self) -> dict:
        if not self.s3:
            return {"configured": False}
        try:
            objects = self.s3.list_outputs("outputs/", max_keys=1000)
            return {"configured": True, "objects": len(objects),
                    "bytes": sum(o["bytes"] for o in objects)}
        except Exception as exc:  # noqa: BLE001
            return {"configured": True, "error": str(exc)}

    # -- library tags / reuse ---------------------------------------------------

    def set_tags(self, gen_id: str, tags: list) -> dict:
        if not isinstance(tags, list) or len(tags) > 32:
            raise ServiceError("tags must be a list of at most 32 strings")
        clean = [str(t).strip().lstrip("#") for t in tags]
        if any(len(t) > 48 for t in clean):
            raise ServiceError("tags must be at most 48 characters")
        self.db.set_tags(gen_id, clean)
        return {"gen_id": gen_id, "tags": sorted({t for t in clean if t})}

    def set_favorite(self, gen_id: str, favorited: bool) -> dict:
        self.db.set_favorite(gen_id, favorited)
        return {"gen_id": gen_id, "favorited": bool(favorited)}

    def reuse(self, gen_id: str) -> dict:
        """The original submission spec for one generation, so the composer can
        be refilled exactly: prompt sections, files used, and every parameter."""
        gen = self.db.get_generation(gen_id)
        if not gen:
            raise ServiceError(f"unknown generation {gen_id}")
        job = self.db.get_job(gen["job_id"]) if gen.get("job_id") else None
        spec = json.loads(job["spec_json"]) if job and job.get("spec_json") else {}
        spec.pop("first_frame_b64", None)
        spec.pop("last_frame_b64", None)
        spec.pop("ref_images_b64", None)
        spec.pop("ref_audios_b64", None)
        spec.pop("watermark", None)  # watermark bytes are never stored
        return {
            "gen_id": gen_id,
            "task": gen.get("task"),
            "prompt": spec.get("prompt") or {},
            "spec": spec,
            "meta": json.loads(gen.get("meta_json") or "{}"),
        }

    def bootstrap(self) -> dict:
        endpoint = config.endpoint_id()
        endpoint_health = None
        if endpoint:
            try:
                endpoint_health = runpod_client.health(endpoint)
            except runpod_client.RunpodClientError as exc:
                endpoint_health = {"error": str(exc)}
        return {
            "app": "Nocturne Video",
            "version": "0.1.0",
            "endpoint_id": endpoint,
            "endpoint_health": endpoint_health,
            "s3": self.s3_status(),
            "stats": self.db.stats(),
            "output_dir": str(config.OUTPUT_DIR),
        }
