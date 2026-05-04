---
name: Modal Usage Reference
description: How to deploy GPU experiments to Modal A10G — auth, volumes, deploy scripts, cost estimates.
type: reference
originSessionId: ecc348c1-2829-48d3-9d4c-10ed2aa188a8
---
## Modal Setup
- Installed: `uv pip install modal`
- Auth: `modal setup` (browser-based, one-time)
- Profile: `adpena` workspace, already authenticated
- CLI: `.venv/bin/modal` (use venv path)

## Key Commands
```bash
# Check auth
.venv/bin/modal profile list

# Run a deploy script
.venv/bin/modal run src/tac/deploy/modal/modal_renderer_smoke_deploy.py

# Check running functions
.venv/bin/modal app list
```

## Deploy Pattern
```python
import modal
app = modal.App("tac-experiment-name")
image = modal.Image.debian_slim(python_version="3.12").pip_install("torch", ...)
vol = modal.Volume.from_name("tac-precomputed", create_if_missing=True)

@app.function(gpu="A10G", timeout=3600, volumes={"/data": vol})
def train(profile: str):
    ...

@app.local_entrypoint()
def main():
    train.spawn(profile="dp_sims_smoke")  # async
    train.remote(profile="dp_sims_smoke")  # sync
```

## Cost Estimates
- A10G: ~$1.10/hr
- Smoke test (200 ep): ~15-30 min = $0.25-0.55
- Full training (2500 ep): ~3-5 hr = $3.30-5.50
- Precomputed volume: free (persists across runs)

## Lessons (from earlier sessions)
- Always bake precomputed data into a volume (skip 5-min decode)
- Resume from checkpoint is REQUIRED (sessions can be preempted)
- Use wall_clock_timeout in tac profiles for graceful save
- Signal handlers (SIGTERM) catch Modal preemption
