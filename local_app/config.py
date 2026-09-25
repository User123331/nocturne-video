"""Local configuration for the Nocturne Video app.

No third-party dependencies: system Python 3 only. Secrets come from the
macOS Keychain first, then environment variables. Non-secret overrides live
in config.local.json next to this file.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "Output"
UPLOADS_DIR = PROJECT_ROOT / "uploads"
DB_PATH = OUTPUT_DIR / "gallery.sqlite3"
LOCAL_CONFIG_PATH = PROJECT_ROOT / "config.local.json"

HOST = "127.0.0.1"
PORT = int(os.environ.get("NOCTURNE_PORT", "8788"))

API_BASE = "https://api.runpod.ai/v2"

S3_ENDPOINT = "https://s3api-us-ne-1.runpod.io"
S3_BUCKET = "g2bg559zan"
S3_REGION = "US-NE-1"

KEYCHAIN_SERVICE_RUNPOD = "nocturne-video-runpod"
KEYCHAIN_SERVICE_S3 = "nocturne-video-s3"

MAX_BODY_BYTES = 64 * 1024 * 1024
USER_AGENT = "NocturneVideo/0.1"

_DNS: dict | None = None


def _local_config() -> dict:
    global _DNS
    if _DNS is None:
        try:
            _DNS = json.loads(LOCAL_CONFIG_PATH.read_text())
        except (OSError, ValueError):
            _DNS = {}
    return _DNS


def endpoint_id() -> str:
    return os.environ.get(
        "NOCTURNE_ENDPOINT_ID", _local_config().get("endpoint_id", "")
    ).strip()


def set_endpoint_id(value: str) -> None:
    cfg = _local_config()
    cfg["endpoint_id"] = value.strip()
    LOCAL_CONFIG_PATH.write_text(json.dumps(cfg, indent=2) + "\n")


def keychain_password(service: str, account: str) -> str:
    try:
        result = subprocess.run(
            ["security", "find-generic-password", "-s", service, "-a", account, "-w"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return ""


def runpod_api_key() -> str:
    return (
        os.environ.get("RUNPOD_API_KEY")
        or keychain_password(KEYCHAIN_SERVICE_RUNPOD, "api-key")
    ).strip()


def s3_credentials() -> dict:
    return {
        "access_key": (
            os.environ.get("NOCTURNE_S3_ACCESS_KEY")
            or keychain_password(KEYCHAIN_SERVICE_S3, "access-key-id")
        ).strip(),
        "secret_key": (
            os.environ.get("NOCTURNE_S3_SECRET")
            or keychain_password(KEYCHAIN_SERVICE_S3, "secret-access-key")
        ).strip(),
    }
