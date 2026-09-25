"""Minimal read-only SigV4 client for Runpod's S3-compatible storage API.

Adapted from the sibling Deck's scripts/s3-usage.py, which solved the
gateway's quirks: RFC 3986 encoding in the canonical query, a real
User-Agent, and region-scoped signing against the datacenter region.
"""

from __future__ import annotations

import hashlib
import hmac
import re
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from . import config

_RETRYABLE = {500, 502, 503, 504}


def _sign(key: bytes, message: str) -> bytes:
    return hmac.new(key, message.encode("utf-8"), hashlib.sha256).digest()


class RunpodS3:
    def __init__(self, endpoint: str, region: str, access_key: str, secret_key: str):
        parsed = urllib.parse.urlparse(endpoint)
        if parsed.scheme != "https" or not parsed.netloc:
            raise ValueError("S3 endpoint must be an HTTPS URL")
        self.endpoint = endpoint.rstrip("/")
        self.host = parsed.netloc
        self.region = region
        self.access_key = access_key
        self.secret_key = secret_key

    @classmethod
    def from_config(cls) -> "RunpodS3 | None":
        creds = config.s3_credentials()
        if not creds["access_key"] or not creds["secret_key"]:
            return None
        return cls(config.S3_ENDPOINT, config.S3_REGION, creds["access_key"], creds["secret_key"])

    def _request(self, method: str, uri: str, params: list[tuple[str, str]] | None = None,
                 headers_extra: dict | None = None, stream: bool = False):
        params = sorted(params or [])
        canonical_query = "&".join(
            f"{urllib.parse.quote(k, safe='-_.~')}={urllib.parse.quote(v, safe='-_.~')}"
            for k, v in params)
        now = datetime.now(timezone.utc)
        amz_date = now.strftime("%Y%m%dT%H%M%SZ")
        date_stamp = now.strftime("%Y%m%d")
        payload_hash = hashlib.sha256(b"").hexdigest()
        canonical_headers = (
            f"host:{self.host}\n"
            f"x-amz-content-sha256:{payload_hash}\n"
            f"x-amz-date:{amz_date}\n"
        )
        signed_headers = "host;x-amz-content-sha256;x-amz-date"
        canonical_request = "\n".join(
            [method, uri, canonical_query, canonical_headers, signed_headers, payload_hash])
        scope = f"{date_stamp}/{self.region}/s3/aws4_request"
        string_to_sign = "\n".join([
            "AWS4-HMAC-SHA256", amz_date, scope,
            hashlib.sha256(canonical_request.encode("utf-8")).hexdigest(),
        ])
        key = ("AWS4" + self.secret_key).encode("utf-8")
        for step in (date_stamp, self.region, "s3", "aws4_request"):
            key = _sign(key, step)
        signature = hmac.new(key, string_to_sign.encode("utf-8"), hashlib.sha256).hexdigest()

        url = f"{self.endpoint}{uri}"
        if canonical_query:
            url = f"{url}?{canonical_query}"
        headers = {
            "Authorization": (
                "AWS4-HMAC-SHA256 "
                f"Credential={self.access_key}/{scope}, "
                f"SignedHeaders={signed_headers}, Signature={signature}"
            ),
            "x-amz-content-sha256": payload_hash,
            "x-amz-date": amz_date,
            "User-Agent": config.USER_AGENT,
        }
        headers.update(headers_extra or {})
        request = urllib.request.Request(url, method=method, headers=headers)
        last_error: Exception | None = None
        for attempt in range(4):
            try:
                response = urllib.request.urlopen(request, timeout=300)
                return response if stream else response.read()
            except urllib.error.HTTPError as error:
                detail = error.read(512).decode("utf-8", errors="replace")
                if error.code in _RETRYABLE and attempt < 3:
                    last_error = error
                    continue
                raise RuntimeError(f"Runpod S3 returned HTTP {error.code}: {detail}") from error
            except urllib.error.URLError as error:
                last_error = error
                if attempt < 3:
                    continue
                raise RuntimeError(f"Could not reach Runpod S3: {error.reason}") from error
        raise RuntimeError(f"Could not reach Runpod S3: {last_error}")

    def list_outputs(self, prefix: str = "outputs/", max_keys: int = 1000) -> list[dict]:
        """List objects under a prefix: [{key, bytes, last_modified}]."""
        results: list[dict] = []
        token: str | None = None
        while True:
            params = [("list-type", "2"), ("prefix", prefix), ("max-keys", str(max_keys))]
            if token:
                params.append(("continuation-token", token))
            body = self._request("GET", f"/{config.S3_BUCKET}", params=params)
            text = body.decode("utf-8", errors="replace")
            keys = re.findall(r"<Key>([^<]+)</Key>", text)
            sizes = [int(s) for s in re.findall(r"<Size>(\d+)</Size>", text)]
            modified = re.findall(r"<LastModified>([^<]+)</LastModified>", text)
            for i, raw_key in enumerate(keys):
                results.append({
                    "key": urllib.parse.unquote(raw_key),
                    "bytes": sizes[i] if i < len(sizes) else 0,
                    "last_modified": modified[i] if i < len(modified) else "",
                })
            match = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", text)
            token = match.group(1) if match else None
            if not token or len(keys) == 0:
                break
        return results

    def download(self, key: str, dest_path, progress=None) -> str:
        """Stream an object to dest_path (ranged GETs tolerate big files)."""
        dest = dest_path if isinstance(dest_path, Path) else Path(dest_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_suffix(dest.suffix + ".part")
        response = self._request("GET", f"/{config.S3_BUCKET}/{urllib.parse.quote(key)}", stream=True)
        total = 0
        try:
            with part.open("wb") as out:
                while True:
                    block = response.read(1024 * 1024 * 4)
                    if not block:
                        break
                    out.write(block)
                    total += len(block)
                    if progress:
                        progress(total)
        except Exception:
            # Never leave a truncated .part behind: it would be mistaken for a
            # resumable download on the next attempt.
            response.close()
            part.unlink(missing_ok=True)
            raise
        finally:
            response.close()
        if total == 0:
            part.unlink(missing_ok=True)
            raise RuntimeError(f"empty download for {key}")
        part.replace(dest)
        return str(dest)
