---
name: Use Upstream Venv for Scoring — Local vs Auth Divergence
description: Local scorer (torch 2.11) disagrees with auth on PoseNet by 29x. Use upstream venv (torch 2.10) for reliable signal.
type: feedback
originSessionId: ecc348c1-2829-48d3-9d4c-10ed2aa188a8
---
CRITICAL FINDING (2026-04-12): Same archive scores 1.33 on auth, 1.93 locally. PoseNet is 29x worse locally (0.063 vs 0.002). SegNet and rate are nearly identical.

**Root cause:** PyTorch version mismatch. Our venv: torch 2.11.0. Upstream venv: torch 2.10.0. Auth CI uses `uv sync` which gets a specific torch version. Different versions produce different floating point results in PoseNet's BatchNorm/attention layers.

**How to apply:**
- ALWAYS use upstream venv for scoring: `workspace/upstream/comma_video_compression_challenge/.venv/bin/python evaluate.py ...`
- runner.py should be updated to use the upstream Python for the score stage
- Our venv is fine for training (different precision is OK during optimization)
- For local signal: SegNet and rate are reliable. PoseNet is not. Use relative A/B comparisons, not absolute numbers.
- For definitive signal: submit PR to get auth score
