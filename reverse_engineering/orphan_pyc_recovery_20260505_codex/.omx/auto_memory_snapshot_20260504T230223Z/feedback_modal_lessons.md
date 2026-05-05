---
name: Modal Deployment Lessons (DEPRIORITIZED)
description: Hard-won lessons from Modal failures. Modal credits likely exhausted. Vast.ai is now primary.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## STALE NOTICE
Modal is deprioritized as of 2026-04-15. Credits likely exhausted. Vast.ai 4090 is primary compute.
Modal may still be useful for auth eval (short jobs that fit free tier).

## Lessons (still valid if using Modal)

Modal has never completed a full 2500-epoch run from scratch. Every successful result came from resume.

**Why:** Setup (clone upstream, LFS pull, decode video) eats 10-30 min of the 8-hour timeout.
Output streaming disconnects after ~5 min but the GPU keeps running.

**How to apply:**
- Always bake heavy assets into the Docker image (upstream repo, LFS models)
- Always implement resume from training_state.pt on volume
- Use `--force` on volume uploads (avoid "already exists" errors)
- Save training state every 50 epochs (atomic writes)
- Don't trust `modal run` output — check volume for actual results
- `modal app list` shows if function is still running (ephemeral + tasks > 0)
- Download checkpoints with: `.venv/bin/modal volume get comma-lab-weights weights/ ./modal_weights/ --force`

## Vast.ai is primary now
- RTX 4090 at $0.25/hr, 4-5x faster than T4
- $25 budget with $24 hard cap
- All experiment definitions in `src/tac/deploy/vastai/experiments.py`
- Deploy via `scripts/check_vastai.py`
