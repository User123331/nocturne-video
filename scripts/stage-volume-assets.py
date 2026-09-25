#!/usr/bin/env python3
"""Stage manifest assets onto the Runpod Network Volume over its S3 API.

The worker's own downloader (ASSET_SYNC_MODE=weights) needs ALLOW_MODEL_DOWNLOADS=1
on the template; this tool is the offline alternative. It downloads each asset
from its source (Hugging Face or Civitai) to a temporary file, verifies the
byte count and sha256, uploads it to models/<kind>/<relative_dir>/<filename> on
the volume bucket, and deletes the temporary file.

Credentials come from the same places the local app reads: NOCTURNE_S3_* env
vars, then the macOS Keychain entries in docs/LOCAL-APP.md.

Usage:
    python3 scripts/stage-volume-assets.py latent-upscaler-3d 2x-animesharpv4-rcan
    python3 scripts/stage-volume-assets.py --list
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import sys
import tempfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from local_app import config  # noqa: E402
from worker import asset_manager  # noqa: E402

CHUNK = 1024 * 1024


def s3_creds() -> tuple[str, str]:
    creds = config.s3_credentials()
    if not creds["access_key"] or not creds["secret_key"]:
        raise SystemExit("no S3 credentials: set NOCTURNE_S3_ACCESS_KEY/NOCTURNE_S3_SECRET "
                         "or store the Keychain entries from docs/LOCAL-APP.md")
    return creds["access_key"], creds["secret_key"]


def sigv4_headers(method: str, host: str, region: str, key_uri: str,
                  access_key: str, secret_key: str, payload_hash: str) -> dict[str, str]:
    now = datetime.now(timezone.utc)
    amz_date = now.strftime("%Y%m%dT%H%M%SZ")
    date_stamp = now.strftime("%Y%m%d")
    canonical = "\n".join([
        method, key_uri, "",
        f"host:{host}\nx-amz-content-sha256:{payload_hash}\nx-amz-date:{amz_date}\n",
        "host;x-amz-content-sha256;x-amz-date", payload_hash,
    ])
    scope = f"{date_stamp}/{region}/s3/aws4_request"
    string_to_sign = "\n".join([
        "AWS4-HMAC-SHA256", amz_date, scope,
        hashlib.sha256(canonical.encode()).hexdigest(),
    ])
    signing_key = hmac.new(f"AWS4{secret_key}".encode(), date_stamp.encode(), hashlib.sha256).digest()
    for part in (region, "s3", "aws4_request"):
        signing_key = hmac.new(signing_key, part.encode(), hashlib.sha256).digest()
    signature = hmac.new(signing_key, string_to_sign.encode(), hashlib.sha256).hexdigest()
    return {
        "Authorization": (f"AWS4-HMAC-SHA256 Credential={access_key}/{scope}, "
                          f"SignedHeaders=host;x-amz-content-sha256;x-amz-date, Signature={signature}"),
        "x-amz-content-sha256": payload_hash,
        "x-amz-date": amz_date,
    }


def s3_put(endpoint: str, region: str, bucket: str, key: str, path: Path) -> None:
    access_key, secret_key = s3_creds()
    host = endpoint.split("//", 1)[1]
    payload_hash = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(CHUNK), b""):
            payload_hash.update(block)
    headers = sigv4_headers("PUT", host, region, f"/{bucket}/{key}",
                            access_key, secret_key, payload_hash.hexdigest())
    headers["User-Agent"] = config.USER_AGENT  # gateway rejects Python-urllib
    request = urllib.request.Request(f"{endpoint}/{bucket}/{key}", data=path.open("rb"),
                                     headers=headers, method="PUT")
    with urllib.request.urlopen(request, timeout=600) as response:
        if response.status not in (200, 201):
            raise RuntimeError(f"PUT {key} -> HTTP {response.status}")


def s3_head(endpoint: str, region: str, bucket: str, key: str) -> int | None:
    access_key, secret_key = s3_creds()
    host = endpoint.split("//", 1)[1]
    headers = sigv4_headers("HEAD", host, region, f"/{bucket}/{key}",
                            access_key, secret_key, hashlib.sha256(b"").hexdigest())
    headers["User-Agent"] = config.USER_AGENT  # gateway rejects Python-urllib
    request = urllib.request.Request(f"{endpoint}/{bucket}/{key}", headers=headers, method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return int(response.headers.get("Content-Length") or -1)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise


def download(asset: dict, dest: Path, token: str | None) -> None:
    if asset["model_id"] == "huggingface":
        url = (f"https://huggingface.co/{asset['repo']}/resolve/main/{asset['repo_path']}")
        opener = urllib.request.build_opener()
        request = urllib.request.Request(url)
    elif asset["model_id"] == "civitai":
        if not token:
            raise SystemExit(f"{asset['slug']}: CivitAI asset needs CIVITAI_API_TOKEN")
        # Token travels as a query parameter: CivitAI redirects to a signed S3
        # URL and urllib drops Authorization headers across hosts (worker bug 3).
        url = (f"https://civitai.com/api/download/models/{asset['version_id']}"
               f"?token={urllib.parse.quote(token)}")
        request = urllib.request.Request(url)
        opener = urllib.request.build_opener(_NoRedirect())
    else:
        raise SystemExit(f"{asset['slug']}: unsupported model_id {asset['model_id']!r}")

    digest = hashlib.sha256()
    total = 0
    with opener.open(request, timeout=120) as response, dest.open("wb") as out:
        final_url = response.geturl()
        if asset["model_id"] == "civitai" and response.status in (301, 302, 307, 308):
            with opener.open(final_url, timeout=120) as redirect_response, dest.open("ab") as out2:
                for block in iter(lambda: redirect_response.read(CHUNK), b""):
                    digest.update(block)
                    total += len(block)
                    out2.write(block)
        else:
            response2 = response
            for block in iter(lambda: response2.read(CHUNK), b""):
                digest.update(block)
                total += len(block)
                out.write(block)
    if total != asset["bytes"]:
        raise SystemExit(f"{asset['slug']}: downloaded {total} bytes, manifest says {asset['bytes']}")
    expected = asset.get("sha256")
    if expected and digest.hexdigest().upper() != expected.upper():
        raise SystemExit(f"{asset['slug']}: sha256 mismatch: {digest.hexdigest()}")
    print(f"downloaded {asset['slug']}: {total:,} bytes, sha256 {digest.hexdigest()[:16]}…")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):  # noqa: ANN002, ANN003
        return None


def main(argv: list[str]) -> int:
    if "--list" in argv:
        for asset in asset_manager.load_manifest():
            print(f"{asset['slug']:28s} {asset['kind']:24s} {asset['bytes']:>14,}")
        return 0
    if not argv:
        print(__doc__)
        return 2
    manifest = {a["slug"]: a for a in asset_manager.load_manifest()}
    endpoint = config.S3_ENDPOINT
    region = config.S3_REGION
    bucket = os.environ.get("NOCTURNE_VOLUME_ID", "") or config.S3_BUCKET
    if not bucket:
        raise SystemExit("no bucket: set NOCTURNE_VOLUME_ID or NOCTURNE_S3_BUCKET")
    token = os.environ.get("CIVITAI_API_TOKEN", "")
    for slug in argv:
        asset = manifest.get(slug)
        if not asset:
            print(f"unknown slug {slug!r}; --list shows the manifest", file=sys.stderr)
            return 1
        key_uri = f"models/{asset['kind']}/{(asset.get('relative_dir') + '/') if asset.get('relative_dir') else ''}{asset['filename']}"
        existing = s3_head(endpoint, region, bucket, key_uri)
        if existing == asset["bytes"]:
            print(f"{slug}: already on the volume ({existing:,} bytes)")
            continue
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / asset["filename"]
            print(f"{slug}: downloading {asset['bytes']:,} bytes …")
            download(asset, dest, token or None)
            print(f"{slug}: uploading to {key_uri} …")
            s3_put(endpoint, region, bucket, key_uri, dest)
        verified = s3_head(endpoint, region, bucket, key_uri)
        if verified != asset["bytes"]:
            raise SystemExit(f"{slug}: volume size {verified} != manifest {asset['bytes']}")
        print(f"{slug}: staged and verified ({verified:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
