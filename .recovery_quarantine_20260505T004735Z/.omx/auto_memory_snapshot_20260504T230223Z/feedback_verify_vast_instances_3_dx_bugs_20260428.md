---
name: verify_vast_instances.py — 3 DX bugs FIXED 2026-04-28 (commit 21784862)
description: Critical telemetry tool had 3 bugs that caused $2+ in false-positive destroys + obscured real lane state. (1) NoneType crash on status_msg. (2) IDLE false-positive when GPU util 0% during DALI install (5+ min normal). (3) UNREACHABLE conflated SSH-failed vs heartbeat-not-yet-created. Added SETUP classification + bumped IDLE threshold to 20 min + ssh_succeeded parameter.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## Bugs found

### Bug 1 — NoneType crash on status_msg
- File: `scripts/verify_vast_instances.py:180`
- Code: `notes=meta.get("status_msg", "")[:80]`
- Failure: `TypeError: 'NoneType' object is not subscriptable` when status_msg=None (vastai sometimes returns None vs empty string)
- Fix: `notes=(meta.get("status_msg") or "")[:80]`

### Bug 2 — IDLE false-positive during DALI install
- File: `scripts/verify_vast_instances.py:128`
- Old logic: `if gpu_util < 5.0 and age_min > 5.0: return "IDLE"`
- Failure: setup_full.sh DALI install takes ~5 min during which GPU util is 0% but heartbeat updates fresh. Verify marked these as IDLE → auto-destroy → wasted $2 (LM-V2, SI-V2 destroyed false-positively today)
- Fix: bump threshold to 20 min (DALI install + lane warmup + safety margin)

### Bug 3 — UNREACHABLE conflated SSH-failed vs heartbeat-absent
- Old logic: `if age_min is None: return "UNREACHABLE"`
- Failure: lanes in early setup (no heartbeat yet) classified UNREACHABLE same as genuinely-stuck instances. Operators couldn't distinguish "still booting" from "SSH dead"
- Fix: added SETUP classification + ssh_succeeded parameter. SSH OK + no heartbeat → SETUP (whitelisted from auto-destroy). SSH failed → UNREACHABLE.

## Pattern (anti-arbitrariness)

The verify script's heuristics encoded assumptions about lane lifecycle that were WRONG:
- 5 min was too aggressive for DALI install
- "no heartbeat" was too coarse a signal

Both classes of bug are the same meta-bug: heuristic-based health classification without lifecycle awareness. Future tooling should distinguish PHASES (boot, setup, train, eval) explicitly via discrete signals, not derive them from secondary metrics like GPU util.

## When to apply

- ALWAYS use the fixed verify_vast_instances.py for monitoring
- `--auto-destroy-stale` is now safer with SETUP whitelisted
- Default `--stale-minutes 30` is reasonable for our 8-14h training runs

## Cross-references
- `feedback_per_instance_verify_pattern_20260428` — original verify script design
- `feedback_vastai_launch_returns_success_before_lane_starts` — heartbeat-as-canonical-signal
- `project_19_lane_monitoring_cadence_20260428` — monitoring cadence
