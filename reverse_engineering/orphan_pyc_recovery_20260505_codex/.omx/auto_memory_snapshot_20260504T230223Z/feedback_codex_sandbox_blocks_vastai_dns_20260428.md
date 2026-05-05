---
name: codex:rescue subagents CANNOT launch Vast.ai instances — sandbox DNS blocks console.vast.ai
description: 2026-04-28 Lane SAUG-V2 launch via codex:rescue subagent failed with errno 8 (name resolution error) on console.vast.ai. The codex sandbox is network-isolated. ALL Vast.ai launches (and any 3rd-party API calls) MUST come from the parent shell where Bash has internet access. This is a permanent operational rule.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## The bug

When dispatching a `codex:rescue` subagent with a prompt that includes `vastai create instance ...` or any API call to `console.vast.ai`, the subagent's codex CLI cannot resolve the hostname. Error message:
```
Cannot resolve console.vast.ai: errno 8 (name resolution error)
```

This was observed in Lane SAUG-V2 launch (subagent a3a01856a294003f7).

## Why

Codex CLI sandboxes run with restricted network access (likely allowlist for github.com + anthropic.com + a few core services). Third-party APIs like vast.ai are NOT in the allowlist.

## The rule

**Vast.ai launches MUST come from the parent Claude Code Bash tool**, NOT from codex:rescue subagents. The parent has unrestricted Bash with internet access; subagents do not.

This applies broadly to ANY 3rd-party API call:
- vastai (vast.ai)
- AWS / Azure / GCP CLIs
- Modal / Lightning / Kaggle CLIs
- HuggingFace Hub uploads
- gh CLI (mostly works since github.com is allowlisted)

## How to apply

For Vast.ai launches:
1. Parent does `vastai create instance ...` directly via Bash
2. Parent registers to `.omx/state/vastai_active_instances.json` directly
3. Parent SSHs the instance to verify heartbeat
4. ONLY delegate to subagents the work that doesn't need vast.ai API: e.g., "write the deploy script content", "design the lane training recipe", etc.

For canonical launches, prefer:
```bash
PYTHONPATH=src .venv/bin/python -m tac.deploy.vastai.cli launch <experiment_name>
```
(if the experiment is registered in `tac.deploy.vastai.experiments.EXPERIMENTS`)

## What this caught (2026-04-28)

Lane SAUG-V2 launch failed cleanly — no spend, no zombie instance. The sandbox bug was detected and reported by the subagent without creating an instance. **Launch-and-return-early contract caught this correctly.**

## Cross-references
- `feedback_oneshot_vastai_subagent_failure_pattern` — broader subagent-launch failure pattern
- `feedback_remote_setup_script_correct_path_20260428` — separate metabug from same launch wave
- `feedback_vastai_launch_returns_success_before_lane_starts` — server-side hardening (heartbeat verify)
