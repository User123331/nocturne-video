"""Manifest-verified asset staging for the Nocturne Video worker.

Assets live on the Runpod network volume under $VOLUME_ROOT/models/<kind>/.
This module verifies pinned byte sizes (and sha256 where published) exactly
once per file — a marker file records the last successful verification so cold
starts don't re-hash tens of gigabytes — and downloads anything missing when
ALLOW_MODEL_DOWNLOADS=1. It also exposes the verified files to ComfyUI via
per-file symlinks, mirroring the sibling Serverless Runpod Deck's pattern.

CLI:
    python asset_manager.py expose
    python asset_manager.py sync --mode weights --required slug-a,slug-b
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

APP_ROOT = Path(os.getenv("APP_ROOT", "/opt/serverless-image"))
VOLUME_ROOT = Path(os.getenv("VOLUME_ROOT", "/runpod-volume"))
COMFYUI_MODELS = Path(os.getenv("COMFYUI_MODELS_DIR", "/comfyui/models"))
MANIFEST_PATH = Path(os.getenv("ASSET_MANIFEST", APP_ROOT / "config" / "assets.manifest.json"))
MARKER_DIR_NAME = ".nocturne-verified"

EXPOSED_KINDS = ("diffusion_models", "text_encoders", "vae", "vae_approx",
                 "frame_interpolation", "loras")


def load_manifest(path: Path | None = None) -> list[dict]:
    data = json.loads((path or MANIFEST_PATH).read_text())
    if data.get("schema_version") != 1:
        raise ValueError("unsupported manifest schema_version")
    return data["assets"]


def by_slug(manifest: list[dict]) -> dict[str, dict]:
    return {a["slug"]: a for a in manifest}


def asset_volume_path(asset: dict) -> Path:
    rel = asset.get("relative_dir") or ""
    return VOLUME_ROOT / "models" / asset["kind"] / rel / asset["filename"]


def comfy_relative_name(asset: dict) -> str:
    """Path relative to /comfyui/models/<kind>/ as ComfyUI loader nodes expect it."""
    rel = asset.get("relative_dir") or ""
    return f"{rel}/{asset['filename']}" if rel else asset["filename"]


def _marker(asset: dict) -> Path:
    return VOLUME_ROOT / "models" / MARKER_DIR_NAME / f"{asset['slug']}.json"


def _sha256(path: Path, chunk: int = 1024 * 1024 * 8) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest().upper()


def _verify(asset: dict, path: Path, deep: bool) -> None:
    size = path.stat().st_size
    if size != asset["bytes"]:
        raise ValueError(
            f"{asset['slug']}: size mismatch on disk ({size}) vs manifest ({asset['bytes']})")
    want = asset.get("sha256")
    if deep and want:
        got = _sha256(path)
        if got != want:
            raise ValueError(f"{asset['slug']}: sha256 mismatch ({got} != {want})")


def _download(asset: dict, dest: Path) -> None:
    if asset["model_id"] == "civitai":
        url = f"https://civitai.com/api/download/models/{asset['version_id']}?fileId={asset['file_id']}"
        token = os.getenv("CIVITAI_API_TOKEN", "")
        if not token:
            raise RuntimeError(f"{asset['slug']}: CIVITAI_API_TOKEN is not set")
    elif asset["model_id"] == "huggingface":
        url = f"https://huggingface.co/{asset['repo']}/resolve/main/{asset['repo_path']}"
    else:
        raise RuntimeError(f"{asset['slug']}: unknown model_id {asset['model_id']}")

    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={
        "Authorization": f"Bearer {os.getenv('CIVITAI_API_TOKEN', '')}" if asset["model_id"] == "civitai" else "",
        "User-Agent": "nocturne-video-worker/0.1",
    })
    total = 0
    with urllib.request.urlopen(req, timeout=120) as resp, part.open("wb") as out:
        while True:
            block = resp.read(1024 * 1024 * 8)
            if not block:
                break
            out.write(block)
            total += len(block)
            if total % (1024 * 1024 * 512) < len(block):
                print(f"asset_manager: {asset['slug']} {total / 1e9:.1f} GB", flush=True)
    if total != asset["bytes"]:
        part.unlink(missing_ok=True)
        raise ValueError(
            f"{asset['slug']}: downloaded {total} bytes, manifest expects {asset['bytes']}")
    _verify(asset, part, deep=bool(asset.get("sha256")))
    part.replace(dest)
    _marker(asset).parent.mkdir(parents=True, exist_ok=True)
    _marker(asset).write_text(json.dumps({
        "bytes": asset["bytes"], "sha256": asset.get("sha256"), "source": asset["model_id"],
    }))


def ensure_assets(required: list[str], *, allow_downloads: bool = False,
                  manifest: list[dict] | None = None) -> dict[str, str]:
    """Verify (and if allowed, download) every required asset.

    Returns slug -> absolute volume path. Raises RuntimeError naming every
    missing asset when downloads are disabled.
    """
    index = by_slug(manifest or load_manifest())
    missing = [s for s in required if s not in index]
    if missing:
        raise RuntimeError(f"unknown asset slugs in manifest: {missing}")

    def check(slug: str) -> tuple[str, Path | None, str | None]:
        asset = index[slug]
        path = asset_volume_path(asset)
        marker = _marker(asset)
        if path.is_file():
            try:
                if marker.is_file():
                    recorded = json.loads(marker.read_text())
                    if recorded.get("bytes") == asset["bytes"]:
                        return slug, path, None
            except (OSError, ValueError):
                pass
            try:
                _verify(asset, path, deep=bool(asset.get("sha256")))
                _marker(asset).parent.mkdir(parents=True, exist_ok=True)
                marker.write_text(json.dumps({"bytes": asset["bytes"], "sha256": asset.get("sha256")}))
                return slug, path, None
            except ValueError as exc:
                return slug, None, str(exc)
        return slug, None, "missing from volume"

    results: dict[str, str] = {}
    errors: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for slug, path, err in pool.map(check, required):
            if err:
                errors[slug] = err
            else:
                results[slug] = str(path)

    if errors and not allow_downloads:
        raise RuntimeError(
            "missing/invalid assets (set ALLOW_MODEL_DOWNLOADS=1 to fetch): " +
            "; ".join(f"{s}: {e}" for s, e in errors.items()))
    if errors:
        def fetch(slug: str) -> str:
            asset = index[slug]
            path = asset_volume_path(asset)
            print(f"asset_manager: downloading {slug} ({asset['bytes'] / 1e9:.2f} GB)", flush=True)
            _download(asset, path)
            return str(path)

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {pool.submit(fetch, slug): slug for slug in errors}
            for fut in as_completed(futures):
                results[futures[fut]] = fut.result()
        errors.clear()

    return results


def expose() -> None:
    """Symlink verified volume files into ComfyUI's model folders (idempotent)."""
    manifest = load_manifest()
    for asset in manifest:
        kind = asset["kind"]
        if kind not in EXPOSED_KINDS:
            continue
        src = asset_volume_path(asset)
        if not src.is_file():
            continue
        rel = asset.get("relative_dir") or ""
        dest_dir = COMFYUI_MODELS / kind / rel
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / asset["filename"]
        if dest.exists() or dest.is_symlink():
            if dest.is_symlink() and Path(os.readlink(dest)) == src:
                continue
            if dest.is_symlink():
                dest.unlink()
            else:
                continue
        os.symlink(src, dest)
        print(f"asset_manager: exposed {kind}/{comfy_relative_name(asset)}", flush=True)


def main(argv: list[str]) -> int:
    if len(argv) >= 2 and argv[1] == "expose":
        expose()
        return 0
    if len(argv) >= 4 and argv[1] == "sync":
        mode = argv[argv.index("--mode") + 1] if "--mode" in argv else "weights"
        required = argv[argv.index("--required") + 1].split(",") if "--required" in argv else []
        expose()
        if mode == "metadata":
            return 0
        try:
            ensure_assets(required, allow_downloads=True)
        except Exception as exc:  # noqa: BLE001 — cold start must report clearly
            print(f"asset_manager: sync failed: {exc}", file=sys.stderr, flush=True)
            return 1
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
