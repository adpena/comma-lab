---
title: DDM MENU1 realized-flip menu DAG feed
date_utc: 2026-07-23
lane_id: lane_ddm_menu1_realized_flip_menu_20260723
research_only: true
execution_allowed: true
score_claim: false
evidence_axis: "[macOS-CPU frozen-scorer advisory]"
verdict: MENU1_MEASURED_BOX_NOT_REACHED
verdict_scope: "FORMULATION: V19C x scalar/temporal/local paint pool x fixed composed geometry x coarse top-cluster sidecar; families remain open"
pointer: "0.1910828242 [contest-CPU]"
pointer_moved: false
main_landing_review_required: true
---

# Purpose

MENU1 turns the complete 2,649-row SN1 error-source solve menu and six governed
fix families into 15,894 typed rows. It then prices the required unmeasured
paint and targeted rungs on the actual V19C endpoint with frozen SegNet and
PoseNet. Historical PT1, E2, and PA1 deltas remain cross-control evidence; none
is imported as a V19C price.

The delegated `2,265,811` V19C count is the SN1 residual
Road/Undrivable/MyCar bucket. Exact total V19C errors are `2,923,991`; only
that total governs `d_seg`.

# Executable DAG

```text
SHA-bound SN1/PT1/E2/DR2B/C1/V19C inputs
  + AT1X manifest
  + PA1 pose-amplitude receipt
  |
  +-> verify exact content hashes and V19C archive bytes
  |
  +-> compile 2,649 SN1 clusters x 6 fixes
  |     -> stable cluster_id and row_id
  |     -> mechanism_bucket + composition_pool_id
  |     -> COUNTED/FREE/NULL partition on every row
  |     -> historical prices remain cross-control or unpriced
  |     -> explicit SDWL1<->E2 bridge blocker
  |
  +-> receive exact V19C archive
  |     -> render two-frame camera pairs
  |     -> exact frozen SegNet + PoseNet n600 replay
  |     -> preserve every 16-pair JSON+NPZ checkpoint on SSD
  |
  +-> fit frame1 amplitude ladder from V19C to SHA-bound gt_f1
  |     -> scalar gain/bias: 12 counted bytes
  |     -> temporal RGB affine: 16 knots, 204 counted bytes
  |     -> class x row-band RGB affine: 974 counted bytes
  |
  +-> exact same-pool alternative measurements from V19C
  |     -> scalar
  |     -> temporal
  |     -> local class x row-band
  |     -> local statistics + hard placement + analytic coverage
  |     -> choose exactly one pool winner by measured joint S
  |
  +-> top SN1 cluster coarse targeted prototype
  |     -> current Undrivable->target Road, ANNULUS_2_TO_5
  |     -> counted sparse mask sidecar
  |     -> exact joint and byte-budget gates
  |
  +-> PA1 cross-pool import
  |     -> pose_amplitude remains distinct from paint_amplitude
  |     -> FREE rows become FREE_candidate pending receiver survival
  |     -> no PA1 delta is summed into the V19C curve
  |
  +-> final receipt and rs1/#366 route
```

# Measured exits

The exact V19C base is 137,827 B, `d_seg=0.024786978827582466`, and
`d_pose=163.06121002915629`.

The 12-byte scalar and 204-byte temporal rows both improve Seg and Pose:

| row | counted bytes | net error correction | d_seg | d_pose | disposition |
|---|---:|---:|---:|---:|---|
| scalar gain/bias | 12 | 40,366 | 0.02444479200575087 | 159.39533299820565 | same-pool dominated |
| temporal 16-knot affine | 204 | 78,888 | 0.0241182369656033 | 150.74417503260625 | same-pool dominated |
| class x row-band statistics | 974 | -23,980,002 | 0.22806797451443142 | 27.418160360123842 | same-pool dominated |
| statistics + hard + analytic | 974 | -5,394,796 | 0.07051923116048177 | 36.6181847780574 | joint pool winner |

The composed arm wins the joint action because its Pose gain outweighs its Seg
damage. Its endpoint is 138,801 B with `S=26.28022355199344`.

The coarse top-cluster row is rejected by both gates: 227,369 B exceeds the
200,000 B cap and `S=30.104108909525713` is worse than its parent. It
introduces 5,069,958 net errors in this formulation.

The accepted curve misses the box by 8,181,948 errors and
`d_seg=0.06935923116048177`. MyCar is binding at 4,072,489 errors. Route:
`ROUTE_RS1_366_MYCAR_RESIDUAL_AND_POSE_FINISH`.

# Resumability and custody

- The SSD root is
  `/Volumes/VertigoDataTier/pact/ddm_menu1_realized_flip_menu_20260723T214943Z_audit_ext_v2`.
- Every scorer arm has 38 immutable 16-pair JSON+NPZ checkpoints.
- Scalar, temporal, local-statistics, and target-mask payload bytes and hashes
  are preserved separately.
- Earlier invalidated receipts were retained on SSD with round labels; none was
  overwritten or promoted.
- The target cache and V19C archive were read-only. No frontier archive,
  remote/paid dispatch, training, or contest exact evaluation occurred.

# Triality

- DSL/data:
  `.omx/research/configs/ddm_menu1_realized_flip_menu_20260723.json` and
  `.omx/research/ddm_menu1_realized_flip_menu_20260723T214943Z/ddm_menu1_realized_flip_menu_receipt.json`
- DAG: this file and the immutable SSD stage tree
- equations:
  `.omx/research/ddm_menu1_realized_flip_menu_canonical_equations_20260723.md`
- implementation: `src/tac/optimization/ddm_realized_flip_menu.py` and
  `tools/measure_ddm_menu1_realized_flip_menu.py`
- regression: `src/tac/optimization/tests/test_ddm_realized_flip_menu.py` and
  `tools/tests/test_measure_ddm_menu1_realized_flip_menu.py`

# MAIN landing review

MAIN must independently review the full branch diff; verify all input hashes,
the residual/total count split, receiver and scorer custody, frame1-only
application, payload parse-back, 15,894-row completeness, pool competition,
transition and objective arithmetic, COUNTED/FREE/NULL partitions, AT1X and
PA1 blocker semantics, verdict scope, SSD preservation, and pointer immobility
before merge.
