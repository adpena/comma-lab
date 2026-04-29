# Lane M-V2 council audit — BUG-1 train/inference asymmetry

**Date:** 2026-04-28
**Lane:** Lane M-V2 (radial-zoom 1-DOF pose representation on the dilated-h64 baseline)
**Result:** 1.84 [contest-CUDA] (regression vs Lane A 1.15)
**Audit verdict:** the regression is NOT a refutation of the rank-1 hypothesis.

## CRITICAL finding (BUG-1)

The optimizer fed the renderer `[zoom, 0, 0, 0, 0, 0]` (zero-padded pose vector)
during training, while the inflate path fed `[zoom, baseline_1..5]` (frozen
baseline pad). The renderer therefore saw a different distribution at train
vs. inference — a classic train/inference parity bug.

This means the rank-1 claim ("a 1-DOF radial-zoom pose representation lives
in the renderer's input subspace") was never actually tested. Lane M-V2 and
Lane M-V1 are both contaminated.

## Council vote (4/5 approve clean retest)

- **Yousfi**: APPROVE — the asymmetry confounds the test; rerun with the
  projection helper threaded through both code paths.
- **Fridrich**: APPROVE — the BUG-1 mechanism is plausible and worth $0.30 to
  resolve before the rank-1 hypothesis is buried.
- **Hotz**: APPROVE — clean retest is cheap; ship it.
- **Quantizr**: APPROVE — the leaderboard leader's pose representation is
  also low-DOF, so the hypothesis remains live.
- **Contrarian**: WAIT — wants a smoke test at 100 iterations before
  committing to the full retest. (Override: smoke test included in Lane M-V3
  protocol.)

## Lane M-V3 (clean) protocol

- Pass `init_poses[:, 1:6]` through `_project_to_renderer_pose` at BOTH train
  and inflate time.
- Smoke auth eval at iteration 100 (per the pose-TTO smoke rule).
- Predicted band: [1.05, 1.20] [contest-CUDA].
- Cost cap: $0.30, 2 GPU-h.

## Structural remediation

Preflight Check 42 (`check_42_train_inference_parity`) scans for
pose-projection helpers used asymmetrically across train/inflate code paths.
STRICT @ 0 violations after two waivers (BUG-1 marked WAIVED until V3-clean
lands; gradient-domain projection marked different domain).

## References

- `src/tac/optimize_poses.py` — pose projection code
- `src/comma_lab/preflight/strict_checks.py:check_42_train_inference_parity`
- Memory: `feedback_check_42_train_inference_parity_20260428`
- Memory: `project_lane_m_v2_audit_council_findings_20260428`

## Provenance

- compiled_at: 2026-04-29
- compiler: tools/ara_compile.py
- ara_version: 0.1
- claim_id: C9
