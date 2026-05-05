---
name: Never Store Artifacts in /tmp — Use Repo Paths
description: macOS reboot wiped /tmp, losing CRF sweep archives and eval state. All artifacts must be in repo or persistent paths.
type: feedback
originSessionId: ecc348c1-2829-48d3-9d4c-10ed2aa188a8
---
RULE: NEVER store important artifacts in /tmp. System reboots (including auto-updates at 3AM) wipe /tmp.

**What was lost (2026-04-12):**
- 6 CRF sweep archives (had to re-encode)
- Eval state from in-progress scoring run
- Proxy scorer logs

**How to apply:**
- CRF configs: `experiments/crf_sweep/` (in repo)
- CRF archives: `experiments/crf_sweep/archives/` (in repo or persistent dir)
- Eval runs: `submissions/robust_current/eval_runs/` (in repo, gitignored data)
- Training logs: `experiments/results/` (in repo)
- NEVER use /tmp for anything that takes more than 5 minutes to recreate
- runner.py eval_runs/ is correct — persistent, in project dir
