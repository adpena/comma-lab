---
name: No Wasted Resources Ever
description: BINDING NON-NEG. Every GPU dollar must produce a measured score. Tmux deaths, idle instances, stale code, untested deployments are ALL wasted resources. Audit before launching, monitor during, gate measurement at end.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
The SHIRAZ A100 ran 16h, completed Phase 3 with great proxy (0.49, PoseNet 0.0028 = 88x baseline), then sat idle for hours because the same `auth_eval_renderer.py:499` NameError I fixed locally TODAY was the version deployed on the A100 yesterday. **The training succeeded; the auth eval crashed silently; nobody noticed because tmux still showed the session as "alive".**

**Why:** `git pull` was never run on the A100; deploy_vastai.py uploaded the broken version weeks ago and never re-synced.

**Why we need to fix this PERMANENTLY:**

1. **Always re-sync code BEFORE eval kickoff.** `git pull` on the remote is not optional. Add it to the deploy script's `launch()` function as a mandatory step.

2. **Tmux ≠ alive.** A tmux session can persist after the script crashes inside it. Verify the actual process via `ps`, not `tmux ls`. The "Always tmux" memory said use tmux, but didn't say "verify the process inside tmux is still running." Update that contract: heartbeat to a file, not just session existence.

3. **Auth eval must be the FIRST thing to fail loudly.** SHIRAZ saved checkpoints, exported renderer.bin, then crashed during cleanup of a 3.6GB .raw file. The score was COMPUTED but the cleanup-crash buried it. Cleanup must NEVER share scope with measurement output.

4. **Monitor cost burn rate.** A100 at $0.62/hr × 16h idle = $10 wasted on a dead process. Add a periodic `vastai show instances --raw` cron that alerts if `gpu_util < 0.05` for > 30 min on a billed instance.

5. **Pre-flight check 7: stale checkpoint detection.** Before launching ANY new training run, verify the LAST run's checkpoint was actually auth-eval'd through CUDA `evaluate.py`. If not, re-eval first OR launch with `--force-stale` flag.

**The pattern that recurs:** local code is fixed → remote is stale → silent failure → bills accrue → discovery weeks later. The fix surface is the deploy_vastai.py launch path: it must enforce code parity before allowing a run to start.
