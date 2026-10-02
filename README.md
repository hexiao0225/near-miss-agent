# Near-Miss Agent

**Every camera, every near miss between people and vehicles, ranked by risk.**

Cities, fleets and warehouses record thousands of hours of footage, and nobody watches it
until after someone gets hurt. Near misses are the early warning, but they're buried in that
archive. Near-Miss Agent searches every camera at once (SF street cams, Toronto dashcam,
I-24 highway, neighborhood cams, warehouse aisles), has an LLM judge each moment, and gives a
safety manager a ranked board, a hotspot view per camera and a written briefing with actions.

Built at the VAST Builders Challenge, SF, 2 Oct 2026.

## How it works

```
VAST DataEngine pipeline (pre-built)          Near-Miss Agent (this repo)
───────────────────────────────────          ─────────────────────────────────────────────
video chunks in VAST S3                       5 near-miss queries, run in parallel
  → YOLO11 detects objects                      → VSS hybrid search over VastDB (Cosmos Embed vectors)
  → Cosmos Reason describes each clip           → dedupe to one moment per video
  → Cosmos Embed vectors                        → W&B Inference LLM scores each moment:
  → VastDB row per segment                          risk (high/medium/low/none), actors, what happened, action
                                                → hotspot board per camera + clip playback at the moment
                                                → LLM safety briefing for the ops manager
```

Re-ingest prompt used so captions record distance, motion and evasive moves (see below).

## Sponsor stack

| Sponsor | Used for |
|---|---|
| **VAST Data** | AI OS / DataEngine ingest pipeline, VAST S3 video storage, VastDB vector + metadata search |
| **NVIDIA** | Cosmos Reason (clip descriptions), Cosmos Embed (search vectors) |
| **CoreWeave** | GPUs serving Cosmos and YOLO, Kubernetes the app is deployed on |
| **Weights & Biases** | Serverless Inference (`api.inference.wandb.ai`) for risk scoring and the briefing |
| **Cursor** | Built with the Cursor agent and the challenge skills |

Plus Ultralytics YOLO11 for object counts.

## Run it

On the workshop VM (credentials come from `/config/<team>.config`, never committed):

```sh
bash deploy.sh                 # deploys to http://<team host>/app
```

Local UI test with fixture data, no credentials:

```sh
MOCK=1 python3 main.py         # http://localhost:8080/#scan
python3 test_main.py
```

On the VM without Kubernetes:

```sh
set -a; source /config/*.config; set +a
python3 main.py                # http://localhost:8080
```

Env: `VSS_URL`/`VSS_USERNAME`/`VSS_PASSWORD` (or `INGRESS_URL`/`USERNAME`/`PASSWORD`),
`WANDB_API_KEY`, `WANDB_TEAM`, `WANDB_MODEL` (default `meta-llama/Llama-3.1-8B-Instruct`).
With no W&B key it falls back to a keyword scorer so the board still renders.

## Re-ingest prompt

The default captions don't say how close people are to vehicles, so search can't find near
misses reliably. We re-ingested with:

> Describe every person and every vehicle (car, truck, bus, bicycle, forklift) in the clip.
> For each vehicle say whether it is moving, turning, reversing or stopped. Say how close
> the nearest person is to any moving vehicle (touching, within one metre, within a car
> length, farther). Say whether anyone makes an evasive move: stepping back, stopping
> abruptly, jumping aside, braking hard or swerving. End with a near-miss risk: high,
> medium, low or none.

## Files

- `main.py`: server, VSS client, W&B scoring, briefing, video stream proxy (stdlib only)
- `index.html`: the board
- `deploy.sh`: Kubernetes deploy (ConfigMap + Secret + Deployment + Ingress `/app`)
- `fixtures_hits.json`, `test_main.py`: offline mode and tests
