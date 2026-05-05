---
name: Lane GP v3 LANDED 89.67 — Runge phenomenon kills polynomial pose-fit
description: 2026-04-29 ~10am. Lane GP v3 with Fix A (baseline_poses=baseline) scored 89.67 [Modal-T4-CPU], identical to v2 (89.66). Off-manifold dims 1-5 was NOT the dominant cause — degree-10 polynomial CAN'T represent dim 0 trajectory.
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
Lane GP v3 (Modal call_id fc-01KQCTA49BTVP1XBATRGNS5XRJ) landed at 14:43Z with Fix A applied:
- final_score: 89.67 [Modal-T4-CPU]
- avg_posenet_dist: 149.95 (50,000× Lane A baseline 0.003)
- avg_segnet_dist: 0.505 (126× Lane A baseline 0.004)
- archive_size_bytes: 692,568

**Why:** Fix A (council 2026-04-29 PM) added `baseline_poses=baseline` to `reconstruct_poses()` to preserve dims 1-5. This worked — `optimized_poses.pt` was 15.3KB (a 600×6 fp32 = 14.4KB confirms full 6-DOF was written). But the score barely moved (v2 89.66 → v3 89.67). The off-manifold hypothesis was a red herring.

The actual root cause is in `fit_pose_gp.log`:
```
pose_gp: degree10 coeffs=[-186304, 913031, -1.89039e+06, 2.14975e+06, -1.46232e+06,
                           608516, -152934, 22400.4, -1810.34, 60.8654, 33.3414]
pose_gp: reconstructed dim0 RMSE vs baseline = 1.011365 over 600 pairs
```

This is textbook **Runge phenomenon**: degree-10 polynomial through 600 equispaced points develops massive endpoint oscillations. Coefficients alternate signs at 1e6 magnitude with destructive cancellation. RMSE 1.01 ≈ pose signal magnitude itself, so dim 0 reconstruction is essentially noise.

**How to apply:**
- Lane GP polynomial-fit approach is structurally broken at degree=10. Don't redispatch v4 with same architecture.
- If Lane GP rate-saving (~14KB poses → ~22B sidecar) is still desired, switch to DCT basis (low-frequency cutoff captures slow trajectory) or B-spline fit (avoids Runge by using piecewise-low-degree).
- Net rate gain is small: ~14KB / 700KB archive ≈ 2% rate reduction. Not worth chasing without a proven low-error fit.
- **Diagnostic-log-blindness lesson**: `fit_pose_gp.log` clearly printed "RMSE vs baseline = 1.011" — that single line predicted the catastrophe. Future lanes should surface fit-quality diagnostics into RESULT_JSON (not just buried in stage logs) so the recovery tool flags them automatically.

The Lane GP v2 audit memory (`project_lane_gp_v2_audit_20260429.md`) attributed the failure to off-manifold dims 1-5. That attribution was incorrect — Fix A preserves dims 1-5 and the score still pegs at 89.67. The trajectory cannot be modeled by a degree-10 polynomial. Update the audit memory to reflect the corrected post-mortem.
