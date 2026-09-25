# Deploying the Nocturne Video worker image

The image builds via GitHub Actions into GHCR
(`ghcr.io/user123331/nocturne-video:<commit9>` / `:latest`) and Runpod pulls it
into the `nocturne-video-worker` template (`jey3jzaqc4`).

Status: the workflow file is valid, but every run — including a 9-line smoke
test — fails with `startup_failure`. The sibling Deck's fallback workflow hit
the same wall on 2026-09-18, so this is account-level, almost certainly the
GitHub Actions **spending limit / billing state** on the private repo. Pick one
of these paths:

## Option A — make the repo public (fastest, recommended)

Public repos get Actions minutes without billing, and packages published from
a public repo are public, so Runpod can pull without registry auth.

```
gh repo edit User123331/nocturne-video --visibility public --accept-visibility-change-consequences
gh workflow run build-image --repo User123331/nocturne-video --ref main
```

No secrets are in the repository (verified: manifests, code, and docs contain
no tokens; `.env` and `config.local.json` are gitignored).

## Option B — keep the repo private

1. github.com → Settings → Billing → Spending limits → raise the Actions
   spending limit above $0 (or confirm the payment method).
2. Re-run the build: `gh workflow run build-image --repo User123331/nocturne-video --ref main`.
3. After the first successful push, make the GHCR package public once
   (github.com → your profile → Packages → nocturne-video → Package settings →
   Change visibility → Public), or create a Runpod registry auth for GHCR with
   a `read:packages` token and attach it:
   `runpodctl registry add ... && runpodctl template update jey3jzaqc4 --registry-auth-id <id>`.

## Option C — build locally (no GitHub dependency)

```
docker buildx build --platform linux/amd64 -t <dockerhub-user>/nocturne-video:latest --push .
docker login first; QEMU cross-build of a CUDA image takes 1-3 h
runpodctl template update jey3jzaqc4 --image docker.io/<user>/nocturne-video:latest
```

## After the image exists

The endpoint `z54sh6h7vrla60` is already live (US-NE-1, volume `g2bg559zan`,
RTX PRO 6000 Blackwell, 0-1 workers, 3600 s job timeout). First run:

1. Open the dashboard, press **Health-check** (or POST `/api/healthcheck`).
   With `ASSET_SYNC_MODE=weights` + `ALLOW_MODEL_DOWNLOADS=1` the worker
   downloads ~55 GB of weights onto the volume (one time, ~10-30 min).
2. Watch it: `runpodctl serverless get z54sh6h7vrla60` / Runpod console logs.
3. When the healthcheck returns ok, set `ASSET_SYNC_MODE=off`
   (`runpodctl template update jey3jzaqc4 --env '{...full replace...}'`).

## S3 outputs (still pending)

Output upload needs the Runpod **S3 API secret** (`rps_…`, Console → Settings →
S3 API Keys — shown once). Until it is set on the template
(`BUCKET_SECRET_ACCESS_KEY`), finished videos still land on the network volume
mirror; S3 upload is skipped with a warning. The local app also needs the same
pair in Keychain (`nocturne-video-s3/access-key-id|secret-access-key`) to sync
videos into the gallery.
