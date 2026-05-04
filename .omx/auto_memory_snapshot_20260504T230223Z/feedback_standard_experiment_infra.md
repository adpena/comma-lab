---
name: Standard Experiment Infrastructure for ALL tac Experiments
description: Periodic checkpointing, volume commits, resume support, results persistence, replicability manifest must be standard in every tac experiment, not per-experiment boilerplate.
type: feedback
originSessionId: ecc348c1-2829-48d3-9d4c-10ed2aa188a8
---
All tac experiments must include as STANDARD infrastructure:
- Periodic volume commits (results survive crashes/credit exhaustion)
- Resume from checkpoint (--resume flag with full state restoration)
- Results persistence to durable storage (Modal Volume, S3, etc.)
- Replicability manifest (git hash, config, environment, timestamp)
- Auto-kill on divergence (configurable thresholds)
- Full telemetry logging (loss components, metrics, timing, VRAM)

**Why:** The user explicitly said "all of that should be included standard in all tac experiments" after seeing the Modal deploy pattern. Experiment infrastructure should not be reimplemented per-experiment.

**How to apply:** Build a shared `tac.deploy.experiment_runner` module or base class that all experiments inherit. The Modal deploy scripts, Lightning scripts, and Kaggle kernels all use the same runner with platform-specific volume/storage backends.
