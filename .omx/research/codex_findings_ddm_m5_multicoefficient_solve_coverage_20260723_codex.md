---
title: Codex findings - DDM M5 multicoefficient solve coverage
utc: 2026-07-23T11:01:00Z
lane_id: lane_ddm_m5_multicoefficient_solve_coverage_20260723
research_only: true
score_claim: false
main_landing_review_required: true
---

# Outcome first

The exact n600 receiver replay does **not** support a numeric certified
`#366 TRUE scope`.

The receiver-closed v19b integer-lattice stack is inside C1's 200,000 B box
(`137,825 B`, `+3,884 B` versus control) and has real reach: it fixes `232,540`
baseline errors. It also creates `129,218` errors, so its net `103,322` cannot
be labeled zero-collateral solve reach. Every one of the five target-label
strata has nonzero collateral and none is fully inverse-solved.

## Exact table

| stratum | baseline errors | helpful | harmful | net | residual | zero collateral |
|---|---:|---:|---:|---:|---:|---|
| Road | 2,210,770 | 114,814 | 31,990 | 82,824 | 2,127,946 | no |
| Lane | 300,563 | 4,636 | 2,633 | 2,003 | 298,560 | no |
| Undrivable | 236,896 | 11,426 | 36,617 | -25,191 | 262,087 | no |
| Movable | 425,853 | 84,409 | 57,035 | 27,374 | 398,479 | no |
| MyCar | 66,446 | 17,255 | 943 | 16,312 | 50,134 | no |

## Adversarial disposition

- **Road:** `PARTIAL_REACH_MEASURED_FULL_SOLVE_NOT_CERTIFIED`. It is not honest
  to answer either “fully solvable” or “joint descent proved necessary.”
- **Lane:** `JOINT_N600_REACH_MEASURED_ZERO_COLLATERAL_FULL_SOLVE_NOT_ADMITTED`.
  G2G2 failed because its six-pair, 20-coordinate construction was
  under-parameterized and failed the semantic/pose admission predicates. This
  was never infeasibility.
- **Certified residual:** `null` for every stratum. The current inputs name no
  finite complete program set, exhaust no such set at 200,000 B, and provide no
  class-isolated zero-collateral solutions.
- **#366:** `TRUE scope = NOT CERTIFIABLE`; interval stays
  `[0, 2,377,273]`. The measured M3 counterfactual `2,303,328` remains a point
  estimate under one stack, not a universal lower bound.

Claiming the candidate residual as “certified infeasible” would repeat the
exact error M3 was designed to prevent: absence of an admitted proposal is not
infeasibility.

## Durable artifacts

- executable audit:
  `tools/audit_ddm_m5_multicoefficient_solve_coverage.py`;
- typed config:
  `.omx/research/configs/ddm_m5_multicoefficient_solve_coverage_20260723.json`;
- receipt and 38 stage checkpoints:
  `.omx/research/ddm_m5_multicoefficient_solve_coverage_20260723T103457Z/`;
- equations:
  `.omx/research/ddm_m5_multicoefficient_solve_coverage_canonical_equations_20260723.md`;
- DAG/FEED:
  `.omx/research/ddm_m5_multicoefficient_solve_coverage_DAG_FEED_20260723.md`.

R1 quarantine waiver was honored: only `d_pose=0.001610` and `7,195 B` were
cited; no R1 bytes were consumed. No launch, #366 config mutation, score claim,
promotion, or pointer movement occurred.

