---
name: Bootstrap script is `remote_setup_full.sh` NOT `setup_full.sh`
description: 2026-04-28 parent-side metabug. I instructed 3 Cycle 1 launch subagents to use `bash scripts/setup_full.sh` in onstart, but the actual file is `scripts/remote_setup_full.sh`. Lane Ω-V2 first launch caught it cleanly (no spend); Lane EC + SAUG-V2 still in flight may hit the same. Hardening: every onstart command in subagent prompts must use the verified-correct path.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## The bug

When I dispatched 3 Cycle 1 launch subagents (Ω-V2, EC, SAUG-V2), I told each to put this in the onstart command:
```
bash scripts/setup_full.sh
```

The actual script is at:
```
scripts/remote_setup_full.sh
```

(verified via `ls scripts/remote_setup*` after Lane Ω-V2 launch failed clean.)

Other related canonical bootstraps:
- `scripts/remote_setup_full.sh` — full env setup (canonical)
- `scripts/remote_train_bootstrap.sh` — training-specific bootstrap
- `scripts/remote_pose_tto_bootstrap.sh` — pose TTO bootstrap
- `scripts/bootstrap.sh` — local-only bootstrap (creates .venv, installs comma-lab)
- `scripts/probe_nvdec.sh` — NVDEC capability probe (Stage 0)

## Why this happened

I confused the local bootstrap (`scripts/bootstrap.sh`) with the remote bootstrap. On a remote Vast.ai instance with a PyTorch container, you don't want the `.venv` setup from `bootstrap.sh` — you want `remote_setup_full.sh` which uses the container's `/opt/conda/bin/python` (memory: `feedback_canonical_remote_bootstraps`).

## How to apply

For ANY future Vast.ai launch subagent prompt, the onstart MUST be:
```
cd /workspace && git clone https://github.com/adpena/pact.git && cd pact && bash scripts/remote_setup_full.sh && bash scripts/remote_lane_<X>.sh > /tmp/lane.log 2>&1 &
```

Or use the canonical launcher pattern in `src/tac/deploy/vastai/client.py` (which presumably uses the right path).

## Why launch-and-return-early caught it

Lane Ω-V2's first attempt (subagent a4ef1eaf6378e7475) detected the missing file BEFORE creating any Vast.ai instance — the contract requires verifying setup paths exist before spending money. **Zero spend wasted.** This validates the launch-and-return-early pattern from `feedback_oneshot_vastai_subagent_failure_pattern`.

## Cross-references
- `feedback_oneshot_vastai_subagent_failure_pattern` — why launch-and-return-early
- `feedback_canonical_remote_bootstraps` — use container Python, not venv
- `feedback_vastai_launch_returns_success_before_lane_starts` — broader launcher hardening
