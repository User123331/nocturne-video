# Nocturne Video

Runpod serverless video+audio generation on **MiniMax H3** (DaSiWa Hybrid v2
checkpoint) with a local macOS dashboard for prompting, uploads, and a
searchable gallery. Sibling project to *Serverless Runpod Deck* (the image
generation system) — same conventions, new modality.

- **Endpoint region:** US-NE-1 (volume-locked)
- **Network volume:** `minimax-h3` (`g2bg559zan`, 80 GB) — all weights, provenance, and an output mirror live there
- **Output bucket:** same volume via Runpod's S3 API (`s3api-us-ne-1.runpod.io`)

## Architecture

```
web/ (dashboard, localhost:8788)
 └─ local_app/server.py      HTTP API + static hosting
     ├─ local_app/service.py job pipeline, prompt assembly, polling
     │    └─ local_app/runpod_client.py → POST /v2/<endpoint>/run
     │         └─ GPU worker (Dockerfile → runpod/worker-comfyui 5.10.0 + ComfyUI v0.37.2)
     │              worker/handler.py → workflow_factory (native H3 nodes)
     │                                 → handler_upstream (base image)
     │              outputs → /runpod-volume/outputs/<id>/ + S3 outputs/YYYY/MM/DD/<id>/
     ├─ local_app/database.py   SQLite gallery + search
     └─ local_app/s3_sync.py    download finished videos from the bucket
```

The worker uses **ComfyUI core nodes only** (`MiniMaxH3ImageToVideo`,
`MiniMaxH3ReferenceToVideo`, `MiniMaxH3SigmaShift`, `SamplerCustomAdvanced`,
`VAEDecode` + `VAEDecodeAudio`, `CreateVideo`, `SaveVideo`) — DaSiWa's C-MMH3
settings (res_multistep/simple 25 steps shift 11/4 quality; euler/simple 8
steps shift 7/4.5 turbo) and Director prompt sections are reproduced in
`workflow_factory.py`. Her Hybrid v2 checkpoint is the primary UNET; the turbo
LoRA is already merged into it.

## Model set (staged on the volume, ~55 GB)

| Slug | File | GB |
|---|---|---|
| dasiwa-hybrid-v2-int8 | DaSiWa Hybrid v2 (FL2VA+REF2VA+turbo) | 20.9 |
| dasiwa-hybrid-v2-int4 | same, int4 for 24–32 GB GPUs | 12.5 |
| qwen3vl-32b-nvfp4-awq | text encoder (CLIPLoader type `minimax`) | 15.7 |
| video-vae-fp16 / audio-vae-fp32 | decode | 5.2 / 0.6 |
| taeh3-preview, rife-v4.26, latent-upscaler-3d | reserved | ~0.7 |

## Deploy flow

1. Push to `main`, then publish a GitHub **release** (or run the
   `build-image` workflow manually) → GHCR image
   `ghcr.io/user123331/nocturne-video:<commit>`.
2. `runpodctl template create` with that image + env (see docs/DEPLOY.md).
3. Endpoint lives in **US-NE-1** (volume-locked). GPU pool: RTX PRO 6000
   Blackwell 48 GB first, then PRO 6000 MIG 48GB, then H100 SXM.
4. First warm-up: send `{"task": "healthcheck"}` with
   `ASSET_SYNC_MODE=weights`, `ALLOW_MODEL_DOWNLOADS=1`, `CIVITAI_API_TOKEN`
   set — the worker stages the volume (~55 GB, one time). Then flip
   `ASSET_SYNC_MODE=off`.
5. S3 outputs need an S3 API key (Console → Settings → S3 API Keys): env
   `BUCKET_ENDPOINT_URL=https://s3api-us-ne-1.runpod.io`,
   `BUCKET_BUCKET_NAME=g2bg559zan`, `BUCKET_REGION=US-NE-1`,
   `BUCKET_ACCESS_KEY_ID=user_…`, `BUCKET_SECRET_ACCESS_KEY=rps_…`.

## Local app

```
python3 local_app/server.py          # http://127.0.0.1:8788
python3 -m unittest discover tests   # unit tests
```

Secrets: `RUNPOD_API_KEY` env or Keychain entry `nocturne-video-runpod/api-key`;
S3 creds in Keychain `nocturne-video-s3/access-key-id|secret-access-key`.
