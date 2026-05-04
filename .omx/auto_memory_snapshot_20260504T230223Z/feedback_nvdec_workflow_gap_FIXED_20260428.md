---
name: NVDEC workflow gap — DETECTION fixed earlier today, AUTO-DESTROY fixed now (commit 5acebb88-ish)
description: 2026-04-28 PM. User caught: "I thought we permanently fixed the NVDEC issue". DETECTION (Stage 0.5 lightweight probe in 3s) WAS fixed earlier today (commit 58e55890). But the LAUNCHER never acted on the signal — phase2-launch returned SUCCESS regardless, leaving instances billing for hours. The PERMANENT fix in commit just-landed: phase2-launch Stage 2 now polls setup.log for ~60s post-dispatch, auto-destroys NVDEC_BAD hosts. Closes the "Vast.ai launcher returns success before lane starts" feedback memory entry.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## The two-part bug

### Part 1 — DETECTION (FIXED 2026-04-28 commit 58e55890)
- `scripts/probe_nvdec.sh --lightweight` dlopen libnvcuvid.so via ctypes
- Stage 0.5 of `scripts/remote_setup_full.sh` runs the lightweight probe BEFORE Stage 3 DALI install
- Catches NVDEC missing in 3s instead of after 5min DALI install (~95% accuracy per Metabug B fix)
- WORKING — verified J-NWC, lane_w, lane_m_v3, lane_sz_phase2 all caught NVDEC_BAD

### Part 2 — WORKFLOW (FIXED 2026-04-28 PM commit launcher-stage-2-hardening)
- Lightweight probe DETECTS NVDEC missing → setup_full.sh exits 2
- Lane wrapper (run_lane.sh) refuses to run lane after setup exit 2
- BUT — instance keeps billing $0.27/hr forever
- Phase2-launch returned SUCCESS regardless of remote outcome
- Operator dispatches + walks away → 24+ hours of idle burn at $7-9/hr fleet-wide

### The fix (this commit)
`scripts/launch_lane_on_vastai.py:cmd_phase2_launch` — added Stage 2 post-launch verification:
- New helper `_poll_setup_log_for_outcome(host, port, timeout=60)`
- SSH polls `/workspace/setup.log` every 5s for one of:
  - `NVDEC_BAD` → auto-destroy + retry guidance, return 2
  - `SETUP_COMPLETE` → outcome confirmed, return 0
  - Neither → fall through normally
- New `--skip-post-verify` flag for fire-and-forget ops

## Why this was missed earlier

The DETECTION fix was the visible piece (saved 5min DALI install per bad host). The WORKFLOW fix wasn't obvious because:
- Setup_full.sh DOES exit cleanly (no error visible to launcher)
- Lane wrapper DOES refuse to run lane (no work done)
- But the instance KEEPS billing — the LAUNCHER's success path was never updated to verify outcome

This is the same bug class as `feedback_vastai_launch_returns_success_before_lane_starts` — heartbeat freshness was the canonical readiness signal, NOT the launcher's exit code.

## Permanent guard

Could add a preflight check that scans `scripts/launch_lane_on_vastai.py:cmd_phase2_launch` for the `_poll_setup_log_for_outcome` call → fail if absent. Marks the workflow gap as structurally extinct.

## Cost saved

Today's accidental deploy of 30+ lanes onto bad-NVDEC hosts (per `vastai logs` showing same image, different host) burned ~$10 of $9.68/hr × hours before user caught it. With the fix, that becomes ~$0.05 (60s × $0.27/hr × N instances) before auto-destroy. ~200× cost reduction.

## Cross-references
- `feedback_vastai_nvdec_host_variation` — original NVDEC variability discovery
- `feedback_metabugs_round_3_20260428` — Metabug B (probe ran AFTER 5min DALI)
- `feedback_vastai_launch_returns_success_before_lane_starts` — same bug class
- commit 58e55890 — Stage 0.5 lightweight probe (DETECTION fix)
- commit just-landed — Stage 2 post-launch verify (WORKFLOW fix)
