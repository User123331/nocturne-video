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

The worker builds native H3 graphs in `workflow_factory.py`: ComfyUI core
nodes for the model path (`MiniMaxH3ImageToVideo`, `MiniMaxH3ReferenceToVideo`,
`MiniMaxH3SigmaShift`, `SamplerCustomAdvanced`, `VAEDecode` +
`VAEDecodeAudio`, `CreateVideo`, `SaveVideo`) plus the node packs DaSiWa's
workflow requires for the output pipeline: the Advanced LoRA loader, block
cache, watermark, torch resize and RTX refiner from ComfyUI-DaSiWa-Nodes,
`MiniMaxChunkFeedForward` from KJNodes, and the `MMH3UltimateUpscale` latent
pipeline. DaSiWa's C-MMH3 settings (res_multistep/simple 25 steps shift 11/4
Final; euler/simple 8 steps shift 7/4.5 Draft) and Director prompt sections
are reproduced in `workflow_factory.py`. Her Hybrid v2 checkpoint is the
primary UNET; the turbo LoRA is already merged into it.

## Render settings exposed in the dashboard

| Group | Controls |
|---|---|
| Model | Checkpoint (Hybrid v2 Int8 / Int4), mode (T2VA, I2VA, FLF2VA, REF2VA) |
| Pace | Draft (8 steps) / Final (25 steps) / Custom; every sampling field is editable and shows the workflow's recommended ranges |
| Canvas | Aspect chips with Auto (follows the first attached image), resolution presets (SD 0.52 MP, HD 0.83 MP, HD+ 1.05 MP, 2K lite, FHD 2.10 MP), duration, frame rate 8 to 48 fps |
| Sampling | Sampler, scheduler, steps, shifts, seed |
| Acceleration | Chunk feed-forward (chunk count), block cache (reuse threshold, max steps, start/end percent) |
| Finishing | Frame interpolation (RIFE ×2/×3/×4), upscale mode, watermark (PNG, position, scale, opacity), LoRA stack |

The four upscale modes are DaSiWa's: **Model** (pixel upscaler, 2x AnimeSharpV4
RCAN), **Simple** (torch resize with Lanczos or bicubic), **RTX** (NVIDIA VSR,
requires the `nvidia-vfx` bindings; the worker refuses the job with a clear
message when they are absent), and **H3 Latent** (re-samples the AV latent
through the 3D latent upscaler; highest quality, slowest).

## Model set (staged on the volume, ~56 GB)

| Slug | File | GB |
|---|---|---|
| dasiwa-hybrid-v2-int8 | DaSiWa Hybrid v2 (FL2VA+REF2VA+turbo) | 20.9 |
| dasiwa-hybrid-v2-int4 | same, int4 for 24–32 GB GPUs | 12.5 |
| qwen3vl-32b-nvfp4-awq | text encoder (CLIPLoader type `minimax`) | 15.7 |
| video-vae-fp16 / audio-vae-fp32 | decode | 5.2 / 0.6 |
| rife-v4.26, 2x-animesharpv4-rcan | interpolation, Model upscale | ~0.05 |
| latent-upscaler-3d | H3 Latent upscale | 0.7 |
| taeh3-preview | reserved for previews | 0.01 |

Stage a manifest asset onto the volume from a machine without the pod:

```
NOCTURNE_VOLUME_ID=g2bg559zan python3 scripts/stage-volume-assets.py <slug> …
```

## Deploy flow

**Runpod's GitHub integration builds the image** (Serverless -> New Endpoint ->
GitHub -> this repo/branch). Runpod pulls the repo, builds the Dockerfile on
its own infrastructure, and stores the result in its registry as
`registry.runpod.net/<owner>-<repo>-<branch>-dockerfile:<commit>`. This is the
same path the sibling *Serverless Runpod Deck* uses; it needs no GitHub Actions
and no CI billing. A push only takes effect once a GitHub **release** is
published, which is what triggers the rebuild.

The workflow in `.github/workflows/` is a fallback only and does not run on
accounts where Actions is unavailable.

Then point the template at the built image and the endpoint at the template
(details, including the exact `runpodctl` calls, in `docs/DEPLOY.md`).
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

The library is a working dashboard: tag any generation (single, or multi-select
in bulk), star favorites, filter by tag, search prompts/tags/seeds/ids, and
sort. Selecting a generation opens an inspector with the full parameter record
(seed, sampler, steps, shifts, size, frame rate, upscale, cache, LoRAs), the
assembled prompt sections, raw metadata, and **Reuse settings**, which refills
the composer with the original values.

Secrets: `RUNPOD_API_KEY` env or Keychain entry `nocturne-video-runpod/api-key`;
S3 creds in Keychain `nocturne-video-s3/access-key-id|secret-access-key`.
