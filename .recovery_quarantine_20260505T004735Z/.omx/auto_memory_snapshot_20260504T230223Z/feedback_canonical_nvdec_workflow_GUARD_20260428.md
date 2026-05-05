---
name: CANONICAL NVDEC workflow guard — automated end-to-end protection
description: 2026-04-28 PM after $6/hr burned on 87% NVDEC_BAD hosts (20/23 instances). Permanently fixed via 2-part canonical pattern: (1) lightweight probe at setup_full.sh Stage 0.5 (commit 58e55890), (2) phase2-launch Stage 2 post-launch verify polls /workspace/setup.log for ~60s and auto-destroys NVDEC_BAD (commit just-landed). Together: NVDEC failures detected in 3s + auto-destroyed in <2 min. Was: hours of $0.27/hr per bad host before manual catch.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## The PERMANENT canonical pattern

**Two cooperating layers. Both required. Either alone leaves a gap.**

### Layer 1: DETECTION (`scripts/probe_nvdec.sh --lightweight`)
- dlopen libnvcuvid.so + cuvidGetDecoderCaps via ctypes
- ~3 seconds, zero install cost
- Catches ~95% of NVDEC-missing hosts
- Run from `setup_full.sh` Stage 0.5 BEFORE Stage 3 DALI install (saves 5min wasted DALI)

### Layer 2: ACTION (`scripts/launch_lane_on_vastai.py:cmd_phase2_launch` Stage 2)
- After phase2-launch dispatches lane wrapper, polls `/workspace/setup.log` for ~60s
- Three outcomes detected via SSH grep:
  - `NVDEC missing` / `NVDEC_MISSING` → AUTO-DESTROY + return 2 + retry guidance
  - `SETUP_COMPLETE` → return 0 with confirmation
  - Neither yet → return 0 with "verify in 5-15 min" message
- `--skip-post-verify` flag for fire-and-forget ops

## Why the gap existed

The DETECTION fix landed earlier today (commit 58e55890) but only made setup_full.sh exit cleanly with code 2. The lane wrapper (`run_lane.sh`) refused to run lane after setup exit 2. BUT:
- Phase2-launch returned SUCCESS regardless of remote outcome
- Operator dispatched + walked away
- Instance kept billing $0.27/hr × hours
- Per `feedback_vastai_launch_returns_success_before_lane_starts` — same bug class

The PERMANENT pattern requires BOTH layers:
- Layer 1 prevents wasted setup time
- Layer 2 prevents wasted billing time

## Cost analysis

**Without fix (today's accidental run)**:
- 20 bad-NVDEC instances × $0.27/hr × ~2 hours = $10.80 wasted
- Caught manually only when user noticed all lanes idle

**With fix**:
- Phase2-launch runs Stage 2 verify within 60s of dispatch
- 20 bad-NVDEC instances × $0.27/hr × 60s = $0.09 total waste
- ~120× cost reduction

## Permanent guards to add

1. **Preflight Check N+1 — phase2-launch must call _poll_setup_log_for_outcome**:
   - Static AST scan of `scripts/launch_lane_on_vastai.py:cmd_phase2_launch`
   - Assert `_poll_setup_log_for_outcome` called OR `--skip-post-verify` opt-in present
   - STRICT after live=0

2. **Preflight Check N+2 — setup_full.sh must call probe_nvdec.sh --lightweight at Stage 0.5**:
   - Static scan of `scripts/remote_setup_full.sh`
   - Assert `probe_nvdec.sh --lightweight` invocation BEFORE Stage 3 DALI install
   - STRICT now (already in production)

3. **Runtime guard — vastai instance tracker should record NVDEC outcome**:
   - When phase2-launch detects NVDEC_BAD, write to `.omx/state/vastai_nvdec_failures.json`
   - Operator can review which hosts/regions had bad NVDEC for offer-filter learning

## When to apply

- ALWAYS use `phase2-launch` for new lane dispatches (not just phase2 combined)
- Trust the auto-destroy — saves $$ + operator time
- For fire-and-forget batch dispatches, monitor with `verify_vast_instances.py` after 5 min

## Cross-references
- commit 58e55890 — Stage 0.5 lightweight probe (DETECTION)
- commit just-landed — Stage 2 post-launch verify (ACTION)
- `feedback_vastai_nvdec_host_variation` — 4090 NVDEC variability discovery
- `feedback_metabugs_round_3_20260428` — Metabug B (probe ran AFTER 5min DALI)
- `feedback_vastai_launch_returns_success_before_lane_starts` — original bug class
- `feedback_per_instance_verify_pattern_20260428` — verify_vast_instances.py
