---
title: Codex findings - DDM v19c correction saturation
date_utc: 2026-07-23
lane_id: ddm_v19c_correction_saturation
research_only: true
execution_allowed: false
score_claim: false
evidence_axis: "[macOS-CPU frozen-scorer advisory]"
verdict: V19C_CORRECTION_SATURATION_ADMITTED_N600_ADVISORY
verdict_scope: "INSTANCE:V19C x SHA-bound v19b start x finite represented coordinates; no contest-axis, global-family, score, or promotion verdict"
pointer_moved: false
main_landing_review_required: true
---

# Finding

The fixed ten-move v19b endpoint was not correction-saturated on the broader
v19c coordinate inventory. DEV admitted 153 moves before the preregistered
64-consecutive-failure stop; exact sequential n600 replay retained 104.

The strict final n600 endpoint is 137,827 B,
SHA `dc767b59c9e8671b6870e0f9f17a24cfe900dd0f2ae2a251825e41566b52e4c9`,
with `d_seg=0.024786978828`, `d_pose=163.061210029156`, and
`Delta S=-0.18073912464057892` versus sealed v19b. The archive costs 2 B more
than v19b. Its exact decomposition is `-0.180744595` Seg,
`+0.000004138641514828123` Pose, and `+0.0000013317179062443428` rate.

# Saturation certificate

This is a measured coordinate-stream asymptote, not a global family optimum.
The typed inventory contains 231 unique coordinates across six families. The
recursive DEV stream tested 1,002 proposals and stopped at exactly 64
consecutive failures, before exhausting the 200,000 B correction budget.

n600 replay disposition:

- 104 strict admissions;
- 47 measured nonnegative rejections;
- 2 compile-infeasible worldsheet failures whose `joint_delta` is deliberately
  `null`, not fabricated.

The two infeasible proposals were
`cycle_000_worldsheet_pair_346_x_+1` and
`cycle_001_worldsheet_track_059_x_+1`; each left the scorer grid only on the
state-dependent n600 accepted path.

# n600 admitted curve

The receipt preserves all 104 admitted points. Selected exact knees are:

| admission | bytes | d_seg | d_pose | incremental Delta S | cumulative Delta S vs v19b |
|---:|---:|---:|---:|---:|---:|
| 1 | 137,825 | 0.026594382392 | 163.061176881086 | -0.000004204389335280001 | -0.000004204389335280001 |
| 2 | 137,823 | 0.026334152222 | 163.061161969938 | -0.02602619503295735 | -0.02603039942229263 |
| 5 | 137,823 | 0.026140730116 | 163.059414537643 | -0.0039061870384488884 | -0.045588979630809115 |
| 10 | 137,827 | 0.026112416585 | 163.059327969132 | -0.000005947928094288124 | -0.04842838836408492 |
| 20 | 137,828 | 0.025551868015 | 163.059972259764 | -0.000007613178749837912 | -0.10440280234581047 |
| 40 | 137,827 | 0.025475362142 | 163.060153256486 | -0.00008400091549529703 | -0.11203164421188162 |
| 60 | 137,827 | 0.024870427450 | 163.061136897477 | -0.000002523825007760827 | -0.17240331768686684 |
| 80 | 137,828 | 0.024817801581 | 163.061038176386 | -0.000005111049873537943 | -0.1776774624870423 |
| 100 | 137,827 | 0.024787021213 | 163.061210531860 | -0.0000008368475753889326 | -0.18073482389521636 |
| 104 | 137,827 | 0.024786978828 | 163.061210029156 | -0.0000008868623872365644 | -0.18073912464057892 |

The full curve is in the final receipt; the selected table is not a substitute
for its per-admission custody.

# Family attribution

All six families contribute at n600:

| family | measured/classified | admitted | compile infeasible | summed strict gain |
|---|---:|---:|---:|---:|
| inverse-solved row-band | 17 | 9 | 0 | 0.10824782614480749 |
| worldsheet pair event | 7 | 5 | 1 | 0.03267366926651212 |
| scorer-template swap | 4 | 2 | 0 | 0.02534419237950912 |
| worldsheet track event | 50 | 25 | 1 | 0.01320209817923183 |
| pre-uint8 Q8 region | 73 | 62 | 0 | 0.0012414474961441602 |
| grammar-event template swap | 2 | 1 | 0 | 0.000029891174374216522 |

The row-band family supplies the largest score gain; Q8 supplies the largest
admission count but a shallow aggregate tail.

# c1 role/residual attribution

Relative to exact v19b, v19c realizes 213,215 additional net Seg flips:

- Lane+Movable role bucket: 38,859;
- Road+Undrivable+MyCar residual bucket: 174,356;
- residual share: 81.77473442300026%.

The final bucket errors are 658,180 role and 2,265,811 residual, for 2,923,991
total errors and 2,787,152 errors above c1's integer target. The correction
line remains residual-dominant; #366/c1 must consume this exact endpoint and
must not import additive credit from v19b or DEV.

# Atom-order gauge

The shared-template payload is 140 B as emitted and 140 B in canonical order:
`Delta B_order=0`. The current fixed-width ZIP-stored representation therefore
has no atom-order rate actuator. Future c1 CODE work may reopen the lever only
with an order-sensitive coder, placement-index remap, and exact camera-byte
identity proof. No scorer permutation or training-landscape claim was imported.

# Custody

- final receipt SHA:
  `506fb1dfed849beb06358d3a30d624fa8cbdad3c6e0da6cf1bf1ec14960472ae`;
- n600 curve SHA:
  `b23873f45ed001e9e02a54e0e5dc071b3374293d9fd081ff98a4037f76c8b979`;
- final archive SHA:
  `dc767b59c9e8671b6870e0f9f17a24cfe900dd0f2ae2a251825e41566b52e4c9`;
- strict final replay digest:
  `3efcc943769209a5393ee03ee59b80d4287b689d16f03dfcc29589f52ded6cc3`;
- strict final replay: 38 preserved 16-pair batches, identical to sequential
  acceptance rows;
- durable output: 8,025 files and 173,118,320 B, below the 512 MiB local
  durable threshold; no destructive cleanup;
- axis: `[macOS-CPU frozen-scorer advisory]`;
- `score_claim=false`;
- pointer: `0.1910828242 [contest-CPU]`, unmoved.

# MAIN review required

MAIN must independently re-derive the SHA-bound v19b start, all typed
coordinates and recursive ordering, strict `Delta S < 0` decisions, both
compile-infeasible classifications, exact support/camera-identity reuse,
strict final replay equality, per-admission curve, bucket arithmetic,
atom-order no-op, artifact hygiene, false-authority labels, and pointer
immobility before landing.
