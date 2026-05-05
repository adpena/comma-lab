---
name: Preflight Check 42 — pose-projection train/inference parity (BUG-1 class extinct)
description: 2026-04-28 added Check 42 to permanently prevent the BUG-1 class (Lane M-V2 audit). Scans for pose-projection helpers used asymmetrically (optimizer-side without inflate-side parity). 8-test regression suite. STRICT @ 0 violations after 2 waivers. 42 strict checks total in preflight_all.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## What Check 42 catches

Pattern: any function in `experiments/` or `src/tac/` (excluding tests) matching:
- `_project*pose*`
- `project_*_pose`
- `*_pose_pad*`

…that is NOT called from `submissions/robust_current/inflate_renderer.py` AND lacks a `# PROJECT_PARITY_WAIVED:` comment within 15 lines of the def.

## Why this prevents Lane M-V2 BUG-1

Lane M-V2 audit (memory: `project_lane_m_v2_audit_council_findings_20260428`) found that `_project_to_renderer_pose` in `experiments/optimize_poses.py` zero-padded pose dims 1-5 during optimization, but inflate used raw saved tensor (frozen-baseline pad). The optimizer was solving a different problem than what inflate evaluated. Result: 0.076 PoseNet (15× Lane A) — signal of THE BUG, not the architectural premise.

Cost: ~$1.50 + 5h GPU before audit.

## Live state (2026-04-28)

2 helpers detected:
1. `experiments/optimize_poses.py:759` `_project_to_renderer_pose` — **WAIVED** with reference to Lane M-V3-clean fix (queued). Removal of waiver tracks Lane M-V3 completion.
2. `src/tac/scorer_exploits.py:1099` `project_segnet_grad_to_posenet_null_space` — **WAIVED** as different domain (gradient-domain projection, not pose-input).

After waivers: 0 violations → STRICT in `preflight_all`.

## 8 regression tests

- `test_strict_passes_on_real_codebase`
- `test_detects_optimizer_only_helper`
- `test_passes_when_inflate_calls_helper`
- `test_passes_with_waiver_marker`
- `test_waiver_window_extends_back_15_lines` (multi-line waiver comments)
- `test_pose_pad_helper_also_detected` (regex coverage)
- `test_strict_raises_metabugviolation`
- `test_test_files_excluded`

All pass.

## Total preflight checks: 42 STRICT

Today's additions: 37 (resource-fork purge), 38 (test-imports-resolve), 39 (undeployed-producer scan), 40 (FP4 hardware disclosure), 41 (lane heartbeat loop), 42 (pose-projection parity).

## Cross-references
- `project_lane_m_v2_audit_council_findings_20260428` — the motivating BUG-1
- `project_lane_m_v2_landed_1_84_regression_20260428` — the wasted-GPU result
- `feedback_dead_flag_wiring_pattern` — same metabug class (asymmetric use)
- `feedback_proxy_auth_math_useless` — auth eval still required even with parity
