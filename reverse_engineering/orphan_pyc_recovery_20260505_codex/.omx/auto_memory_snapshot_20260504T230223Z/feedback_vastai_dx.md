---
name: Vast.ai DX Lessons — First Deployment
description: Sharp edges, gotchas, and improvements discovered during first Vast.ai deployment
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## Sharp Edges Discovered (2026-04-15)

### SSH Key Must Be Account-Level BEFORE Instance Creation
- `vastai create ssh-key "$(cat ~/.ssh/id_ed25519.pub)"` must run FIRST
- Instances created before key registration get "Permission denied (publickey)"
- `vastai attach ssh <id> "<key>"` on existing instances is unreliable
- **Fix**: Always register key at account level, then create instances with `--ssh`

### Python Output Buffering Eats Logs
- `nohup python3 script.py > log 2>&1 &` produces empty log files
- Python's stdout is fully buffered when redirected
- **Fix**: Always use `python3 -u` (unbuffered) for background jobs

### PYTHONPATH Must Include Repo Root for Cross-Module Imports
- `experiments/tto_step_curve.py` imports `from experiments.renderer_tto import load_renderer`
- `PYTHONPATH=src:upstream` is not enough — need `PYTHONPATH=src:upstream:$PWD`
- **Fix**: Always include repo root in PYTHONPATH

### Instance "loading" Can Stall Indefinitely
- Instance A (offer 30101221) was stuck in "loading" for 10+ minutes
- Solution: destroy and pick a different offer
- **DX improvement**: Add a timeout to instance creation — if SSH isn't ready in 5 min, destroy and retry

### Onstart Script Runs Async
- `--onstart-cmd` installs packages WHILE the instance reports "running"
- SSH connects but packages (ffmpeg, etc.) may not be installed yet
- **Fix**: Poll for required binaries with `until which ffmpeg; do sleep 5; done`

### rsync for Code Deployment Works Well
- `rsync -avz -e "ssh -p PORT" /tmp/bundle/ root@host:/workspace/pact/` is fast and reliable
- 7.8MB bundle deploys in ~2 seconds
- **DX improvement**: Create a `scripts/deploy_vastai.sh` helper

## Cost Tracking
- Instance B: $0.294/hr, running since ~16:35 UTC
- Instance A: destroyed after stall, wasted ~$0.05 (12 min loading)
- Instance A2: created, status pending

## What Worked Well
- Vast.ai search API is fast and reliable
- Instance creation is instant
- 4090s are plentiful at $0.24-0.30/hr
- Code deployment via rsync is clean
- Upstream git clone + LFS pull takes ~30s on fast instances
