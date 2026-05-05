---
name: Training Profiles
description: Named training profiles in src/tac/profiles.py — council_v1 is the recommended default for all new runs
type: reference
originSessionId: ecc348c1-2829-48d3-9d4c-10ed2aa188a8
---
## Usage
```bash
# Local
.venv/bin/python experiments/train_tac.py --profile council_v1 --tag my_run --precomputed experiments/precomputed_local

# Lightning
PROFILE=council_v1 TAG=my_tag nohup bash run.sh > train.log 2>&1 &

# Modal (in deploy script)
PROFILE=segnet_attack TAG=modal_attack ...
```

## Profiles (src/tac/profiles.py)

**council_v1** — Full council consensus (2026-04-10). Use for all new runs.
- dilated h=64, KL distill T=5→0.5, boundary_weight=150, boundary_anneal=True
- hard_frame_ratio=0.3, error_replay_every=200, eval_every=5
- PoseNet gradient cap active (in losses.py)

**segnet_attack** — Aggressive SegNet-focused variant
- Same as council_v1 but: T→0.2, bw=200, hard_frame=0.5, replay=100

**proven_baseline** — The settings that produced the 1.33 checkpoint
- dilated h=64, standard loss, no curriculum, bw=1.0

**h96_council** — Width scaling with council settings
- Same as council_v1 but hidden=96

**smoke** — Quick smoke test (50 epochs, h=16)

CLI args override profile values. Profile sets defaults only.
