---
title: DDM MC1 static-hood reassert DAG feed
date_utc: 2026-07-24
lane_id: lane_ddm_mc1_hood_static_reassert_20260724
research_only: true
score_claim: false
evidence_axis: "[macOS-CPU frozen-scorer advisory]"
verdict: MC1_MEASURED_INSTANCE_NOT_JOINT_POSITIVE
verdict_scope: "INSTANCE: exact V19C base-byte hood reassert after the MENU1 frame1 winner; pose-stat-preserving static-field formulations remain open"
pointer: "0.1910828242 [contest-CPU]"
pointer_moved: false
main_landing_review_required: true
---

# Executable feed

```text
MENU1 receipt SHA 2fc12eb505...
  + exact V19C receiver
  + preserved MENU1 v19c_base / winner checkpoints
  |
  +-> reconstruct winner batch camera
  |     -> require camera SHA == preserved MENU1 winner SHA
  |     -> require frame_0 == V19C frame_0
  |
  +-> derive hood class from V19C base argmax
  |     -> argmax_c bottom_share(c) * static_iou(c)
  |     -> detected class 4 without hard-coding
  |     -> single-static mean frame IoU 0.9959164190
  |
  +-> price three support formulations
  |     -> single-static stored: COUNTED 139 B
  |     -> per-frame stored: COUNTED 58,026 B
  |     -> receiver semantic support: FREE 0 new B
  |
  +-> ordered composition on frame_1 only
  |     -> MENU1 paint winner
  |     -> restore V19C bytes on support
  |     -> exact support-locality + frame_0 identity assertions
  |
  +-> frozen CPU-torch n600 receiver measurement
  |     -> SegNet total + per target class
  |     -> official PoseNet two-frame path
  |     -> official preprocessed-input coupling telemetry
  |     -> 38 immutable JSON+NPZ checkpoints per candidate
  |
  +-> choose one static_reassert pool row by exact joint S
        -> all three Delta S > 0
        -> reject measured instance, keep family open
```

# Measured rows

Parent MENU1 winner: `B=138,801`, errors `8,318,787`,
`d_seg=0.07051923116048177`, `d_pose=36.6181847780574`,
`S=26.28022355199344`.

| support | partition | errors | d_seg | d_pose | S | Delta S |
|---|---:|---:|---:|---:|---:|---:|
| single static | 139 B COUNTED | 6,571,730 | 0.05570924546983507 | 64.85599367436599 | 31.13027893413343 | +4.850055382139988 |
| per frame | 58,026 B COUNTED | 7,593,268 | 0.06436893039279513 | 67.4773211303703 | 32.54434925581808 | +6.264125703824636 |
| decoder semantic | 0 B FREE | 8,038,719 | 0.06814506530761719 | 65.7222283407805 | 32.54327533941567 | +6.263051787422231 |

The best single-static row recovers `1,519,350` net MyCar errors and
`1,747,057` total errors, but the official PoseNet input changes in
`45,397,200` coordinates with L1 `1,849,418,743.604599`; `d_pose` rises by
`28.23780889630859`. The pose loss dominates the Seg gain.

# Pool and route

`static_reassert` is a new pool. It is ordered after the MENU1 paint arm and
overlaps that paint support, so its interaction is measured directly and is
not treated as additive. No row enters c1 waterfill because every measured
`Delta S` is positive.

The best rejected instance leaves `2,553,139` MyCar errors. That is the honest
rs1/#366 residual scope, not a claim that all are reachable by this family.
The first rung is a one-time static hood field constrained to preserve the
official PoseNet YUV6 sufficient statistics (or an exact trust-region partial
reassert), followed by the same n600 joint gate.

# Triality and custody

- DSL/data: `.omx/research/configs/ddm_mc1_hood_static_reassert_20260724.json`
- DAG: this FEED and the immutable SSD checkpoint tree
- equation: `ddm_mc1_static_hood_reassert_joint_action_v1`, executable at
  `tac.canonical_equations.ddm_mc1_hood_static_reassert_20260724:evaluate_hood_reassert_joint_delta`
- receipt:
  `.omx/research/ddm_mc1_hood_static_reassert_20260724T003346Z/ddm_mc1_hood_static_reassert_receipt.json`
  (SHA `458043413339551fe785e605d54751c46fe0d8b24c7c4ee59a67426872e320e8`)
- SSD:
  `/Volumes/VertigoDataTier/pact/ddm_mc1_hood_static_reassert_20260724T003346Z`

No paid dispatch, training, exact contest evaluation, archive promotion, or
frontier mutation occurred.

# MAIN landing review

MAIN must independently review the full base-to-branch diff and rederive the
support partition, exact parent-camera hashes, frame-0 identity, transition
conservation, official Pose coupling, joint objective, equation registration,
negative scope, and pointer immobility before merge.
