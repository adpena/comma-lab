# DDM PA1 PoseNet amplitude twin — DAG FEED — 2026-07-23

`research_only=true` · `score_claim=false` ·
`[macOS-CPU frozen-scorer advisory]` · pointer unchanged.

## Feed edge

`E2 strict raw (SHA 4871b1c1...)`
→ literal upstream resize + `rgb_to_yuv6`
→ n600 12-channel moment scan
→ `{GT-video target: COUNTED, frozen-BN inverse target: FREE candidate}`
→ frame-0 camera residual
→ frozen PoseNet + mandatory frozen SegNet
→ `menu1:pose_amplitude`.

The scorer-only target is derived from the two frozen first-stem convolution
branches and their AT1x BN running-stat tables. Candidate moments are computed
from already-counted decoded E2 content. No GT-video statistic, Pose6 target,
scorer weight tensor, or extra payload byte is introduced. Governed E2
`inflate.py` composition and exact archive parse-back remain an explicit
promotion blocker.

## Measured menu rows

| Rung | Target home | Δ bytes | d_pose | d_seg | Joint ΔS | Verdict |
|---|---|---:|---:|---:|---:|---|
| E2 control | existing packet | 0 | 162.580958694146 | 0.028614807129 | 0 | control |
| frame0 GT-stat | COUNTED | 24 | 162.423102394524 | 0.028614807129 | -0.019563561860 | positive |
| frame0 scorer-stat | FREE candidate | 0 | 147.491042043395 | 0.028614807129 | -1.916766686214 | selected |
| joint scorer-stat | FREE candidate | 0 | 161.726887970608 | 0.035720909966 | +0.604562771375 | reject |

The frame-0 arms changed zero frame-1 channel values and therefore preserved
all 3,375,540 Seg errors exactly. The joint rung improved pose but added
838,270 Seg errors; the Seg term dominated.

## Consumer directive

`menu1` should consume pool `pose_amplitude` with the frame-0 scorer-stat row
as the first receiver-closed measurement candidate. The next measurement is
not another statistics ladder: compose the generic two-pass transform into the
governed E2 runtime, preserve resume/checkpoint custody, price exact archive
bytes, and remeasure on the same advisory axis before any contest replay.

## Triality

- DSL: `DDMPA1PoseNetAmplitudeTwinConfigV1` seals E2 raw/archive, target cache,
  scorer sources/weights, AT1x manifest, byte homes, and falsifier margin.
- DAG: the feed edge above; frame 0 and frame 1 are separate ownership nodes.
- Equations:
  `ddm_pa1_pose_amplitude_moment_match_v1`,
  `ddm_pa1_scorer_only_bn_inverse_target_v1`,
  `ddm_pa1_free_null_counted_target_partition_v1`, and
  `ddm_pa1_shared_rgb_joint_price_v1`.

Canonical code:
`src/tac/canonical_equations/ddm_pa1_posenet_amplitude_twin_laws_20260723.py`.
Canonical receipt:
`.omx/research/ddm_pa1_posenet_amplitude_twin_20260723T221923Z/ddm_pa1_posenet_amplitude_twin_receipt.json`.

MAIN landing review is required.
