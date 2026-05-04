---
name: optimize_poses.py is silent — fix progress logging
description: optimize_poses.py runs at 100% GPU for 100+ min with ZERO log output. Confirmed running via process check, but no way to gauge progress. Add periodic stdout heartbeat.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
Observed 2026-04-25 21:53 → 23:40+ on A100 SXM4:
- `optimize_poses.py --steps 500 --batch-pairs 10` at 100% GPU
- 1h47m elapsed, ZERO progress lines in stdout
- Only "=== SHIRAZ pose re-TTO ===" header from the wrapper script
- Process confirmed running via ps, GPU confirmed working via nvidia-smi

This violates the "no wasted resources" rule because:
- Operator cannot tell if it's making progress vs stuck
- Cost is accruing ($0.62/hr × N hours of opacity)
- Can't decide "kill" vs "let it finish" without information

REQUIRED FIX (council can approve as bug fix):
1. Add `print(f"  [step {step}/{total}] loss={loss:.6f}", flush=True)` every N steps (recommend N=25 for 500-step run = 20 progress lines)
2. Add `tqdm` wrapper with file=sys.stdout, mininterval=10 for terminal-friendly progress
3. Write per-step metrics to a sidecar JSON (per-step pose/seg distortions)
4. Add `--log-interval N` CLI flag with sensible default

Pattern to copy: train_renderer.py line 832 has the right idiom — every N steps print a structured loss line + occasional FP4 eval.

WHY THIS MATTERS:
- During contest sprint, EVERY hour of GPU time should produce visible progress
- Silent runs trigger the "is it dead?" diagnosis ritual that wastes more cycles
- A 500-step run at 100 min is fine; not knowing whether you're at step 50 or step 450 is not

CHARTER: any new training/optimization script must default to verbose progress logging at >0 stdout output every 60s. Without this, silent runs auto-fail the no-wasted-resources rule.
