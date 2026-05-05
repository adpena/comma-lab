---
name: Remote Code Parity Required Before Eval
description: BINDING. Before any remote eval, verify the deployed code matches local HEAD. Stale code on remote killed SHIRAZ today. The auth_eval_renderer.py NameError I fixed today was the version deployed yesterday on A100.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
The bug pattern (verified today, 2026-04-25):
1. SHIRAZ trained on A100 over 16 hours
2. SHIRAZ Phase 3 finished, exported renderer_distilled.bin (99KB)
3. Pipeline tried to auth-eval the result
4. auth_eval_renderer.py raised NameError on `expected_raw` (line 499) — the bug I fixed locally TODAY
5. Cleanup crashed; results never saved
6. Tmux session shows "alive" — appears successful
7. We discovered hours later because we SSH'd in to check

THE CONTRACT going forward:
- `deploy_vastai.py launch()` MUST run `git pull --ff-only` on the remote BEFORE starting any work
- Sync code BEFORE the model trains, BEFORE the eval runs, BEFORE the pipeline begins
- If git pull fails (uncommitted changes, conflict), abort the launch
- Add a heartbeat mechanism: every N minutes the running script writes timestamp to `/tmp/heartbeat_<session>.log`
- A separate watchdog (preferably local) reads heartbeats; alerts if stale > 30 min
- Tmux session existence is NOT a heartbeat — the process inside can crash

PREFLIGHT CHECK TO ADD (#7): "remote code parity"
- For any deployment target, verify HEAD commit matches remote HEAD
- Pre-launch: SSH in, get `cd /workspace/pact && git rev-parse HEAD`, compare to local HEAD
- Block launch on mismatch unless `--allow-stale-remote` is passed (with warning)
- Should be a one-line addition to deploy_vastai.py preflight chain

THE COST:
- 16 hours of A100 time on idle (the eval crashed; we kept paying)
- $0.62/hr × 5h before noticed = $3.10 wasted
- Plus the next-action delay (we discovered too late to start retraining today)
- Plus the SHIRAZ score we never got — could have informed today's plan

HOW THE FIX WAS LOST IN THE FIRST PLACE:
- I committed the auth_eval fix locally this morning (commit c3efc172)
- I pushed to A100 ONLY when manually triggered just now
- There is no automatic mechanism to keep remote in sync
- The deploy_vastai.py launch path runs the OLD synced code

SPECIFIC FILES THAT NEED PARITY GUARANTEE:
- experiments/auth_eval_renderer.py (today's fix would have caught this)
- experiments/pipeline.py (the orchestrator)
- src/tac/preflight.py (the validators themselves)
- submissions/robust_current/inflate_renderer.py (the inflate path)
- src/tac/submission_archive.py (the archive builder)
- All Python files in src/tac/experiments/ (training scripts)

If any of these is stale on remote, the next deployment is INVALID.
