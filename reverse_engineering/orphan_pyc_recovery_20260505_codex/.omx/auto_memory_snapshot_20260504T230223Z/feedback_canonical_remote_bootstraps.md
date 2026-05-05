---
name: Canonical Remote Bootstraps + Provenance + Records
description: Two reusable canonical scripts for any Vast.ai 4090/A100 deploy. Provenance.json + run_record.json are now part of every run.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
**BINDING NON-NEGOTIABLE.** All Vast.ai/remote deploys go through one of these two canonical scripts. Ad-hoc /tmp scripts are blocked by preflight (`_scan_text_for_dangerous_patterns`).

**Two scripts, both two-mode (default = launch tmux, --inner = work):**

1. **`scripts/remote_train_bootstrap.sh <profile> [output_subdir]`**
   - For training a renderer from scratch via `train_renderer.py --profile <profile>`.
   - Chains: bootstrap deps (use `/opt/conda/bin/python` from container, NOT a fresh venv) → determinism check → train_renderer → checkpoint probe → pipeline.py compress (FP4 self-compress + QAT + pose TTO + archive + auth eval).
   - Used for: DEN, SHIRAZ-style fresh training profiles.

2. **`scripts/remote_pose_tto_bootstrap.sh <checkpoint.pt> <masks.mkv> <profile> [output_subdir]`**
   - For running the FULL post-training stack on an existing float `.pt` checkpoint.
   - Chains: bootstrap deps → determinism check → pipeline.py compress (same step_export → step_qat → step_pose_tto → step_archive → step_eval as above, but starting from the provided checkpoint instead of training first).
   - Used for: SHIRAZ-class re-runs where we already have a trained renderer and just need to re-do post-training with new masks/poses.

**Both scripts emit:**
- `<output_dir>/provenance.json` — git_hash, gpu_name, driver_version, torch_version, cuda_version, profile, output_dir, pipeline path, CUBLAS_WORKSPACE_CONFIG. Written BEFORE any work so the record exists even if the run dies mid-stage.
- `<output_dir>/heartbeat.log` — per-minute GPU util + memory snapshot with profile-cookied label (`pgrep -f` safe).
- `<output_dir>/train.log` (or `pipeline.log`) — full stdout/stderr.
- `<output_dir>/run_record.json` (post_tto only) — finished_at_utc, auth_eval JSON dump, list of artifacts.

**Determinism:** both scripts export `CUBLAS_WORKSPACE_CONFIG=:4096:8` BEFORE any cuBLAS call (must be set before first import torch.cuda). `tools/check_determinism.py` validates the env + asserts cudnn.deterministic + benchmark are correctly set after `configure_reproducibility()`.

**Container Python (NOT a fresh venv):** Vast.ai pytorch image at `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel` already has working torch+CUDA. Creating a fresh `uv venv` pulled torch wheels for newer CUDA that the host driver couldn't satisfy (Error 804: forward compatibility on non-supported HW). Always use `/opt/conda/bin/python`.

**Why:** Hardened 2026-04-26 after the SHIRAZ deploy disasters. Records ensure (a) reproducibility (provenance pins versions), (b) auditability (heartbeat detects silent death), (c) usability as actual records (run_record.json is the "what happened" snapshot for each experiment).
