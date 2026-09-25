"""Runpod serverless client: async run / poll / cancel."""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from . import config


class RunpodClientError(RuntimeError):
    pass


def _request(method: str, url: str, payload: dict | None = None, timeout: int = 60) -> dict:
    key = config.runpod_api_key()
    if not key:
        raise RunpodClientError("no Runpod API key (env RUNPOD_API_KEY or Keychain)")
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(
        url, data=data, method=method,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "User-Agent": config.USER_AGENT,
        })
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            body = resp.read()
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as error:
        detail = error.read(512).decode("utf-8", errors="replace")
        raise RunpodClientError(f"Runpod API HTTP {error.code}: {detail}") from error
    except urllib.error.URLError as error:
        raise RunpodClientError(f"Could not reach Runpod API: {error.reason}") from error


def _endpoint_url() -> str:
    endpoint = config.endpoint_id()
    if not endpoint:
        raise RunpodClientError("no endpoint configured (set NOCTURNE_ENDPOINT_ID or config.local.json)")
    return f"{config.API_BASE}/{endpoint}"


def run(payload: dict) -> dict:
    """POST the async /run; returns {"id": <runpod job id>}."""
    result = _request("POST", f"{_endpoint_url()}/run", {"input": payload})
    job_id = result.get("id")
    if not job_id:
        raise RunpodClientError(f"run() returned no id: {json.dumps(result)[:300]}")
    return result


def status(job_id: str) -> dict:
    return _request("GET", f"{_endpoint_url()}/status/{job_id}")


def cancel(job_id: str) -> dict:
    return _request("POST", f"{_endpoint_url()}/cancel/{job_id}")


def health(endpoint_id: str | None = None) -> dict:
    """Probe the endpoint with GET /health (no GPU cost)."""
    endpoint = endpoint_id or config.endpoint_id()
    if not endpoint:
        raise RunpodClientError("no endpoint configured")
    return _request("GET", f"{config.API_BASE}/{endpoint}/health")
