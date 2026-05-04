---
name: GPU Provider Strategy — Vast.ai 4090 Primary, Free Tier Rotation
description: Comprehensive GPU pricing research. Vast.ai RTX 4090 at $0.17-0.34/hr is optimal. AWS/Azure free credits available. Modal for existing infra.
type: reference
originSessionId: 47bf3dd8-df75-4271-9ce1-428c19c2eb32
---
## Primary: Vast.ai RTX 4090
- $0.17-0.34/hr on Vast.ai marketplace
- 4-5x faster than T4 for our workload (scorer CNN forward/backward passes)
- Per-second billing, Docker-based, full CLI/API
- `pip install vastai`, API key from cloud.vast.ai
- Filter: reliability > 0.95, verified datacenter hosts
- On-demand for 4-8hr runs (interruptible can be reclaimed)
- 10 parallel 4090s × 1 hour = ~$3.40 total

## Secondary: Modal (existing infrastructure)
- T4 at $0.59/hr (4x more expensive than Vast.ai)
- $30/mo free credits (~50 T4-hours)
- Keep for existing deployments (renderer training, TTO, auth eval)
- Serverless autoscaling, zero idle cost

## Free Tier Rotation (~78+ GPU-hrs/week):
- Kaggle: 30 hrs/wk (T4/P100 lottery, unreliable)
- Google Colab: 15-30 hrs/wk (T4, sessions drop)
- SageMaker Studio Lab: 28 hrs/wk (4hr/day T4) — UNTAPPED
- Lightning.ai: ~5 hrs/wk (T4, SSH working)

## Free Credits (authorized to use — DO NOT OVERSPEND):
- AWS: **$100** free credits (~450 T4 spot hours). Hard budget cap. Track spend carefully.
- Azure: $200 = ~1,800 T4 spot hours (NC4as_T4_v3 at $0.11/hr spot)
- Vast.ai: $25 credits = ~100 hours RTX 4090
- Modal: $30/mo free credits (~50 T4-hours)
- GCP: not claimed yet
- Oracle: not claimed yet

## GPU Price/Performance for Our Workload (287K params, 800MB VRAM):
| GPU | Vast.ai $/hr | Speed vs T4 | $/experiment |
|-----|-------------|-------------|--------------|
| RTX 4090 | $0.17-0.34 | 4-5x | $0.20 | ← OPTIMAL
| RTX 3090 | $0.15-0.20 | 2.5x | $0.32 |
| T4 | $0.15 | 1x | $0.60 |
| A100 | $0.80-1.10 | 5x | $0.88 | overkill

## Vast.ai Setup
- CLI: `uv pip install vastai`
- API key: ~/.vast_api_key or VAST_API_KEY env
- Deploy: `vastai search offers`, `vastai create instance`, SSH + rsync code
- Custom deploy script: scripts/vastai_deploy.py (to be built)
