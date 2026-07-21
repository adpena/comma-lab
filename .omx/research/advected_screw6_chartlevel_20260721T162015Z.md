# Full-screw advection with chart-level coefficient pricing

UTC 2026-07-21T16:20:15Z · lane
`lane_advected_screw6_chartlevel_20260721` · `research_only=true` ·
`[macOS-CPU advisory]` · hard CPU Torch · seed 1234 · `$0` ·
`score_claim=false` · `promotion_eligible=false` · pointer
`0.19108 [contest-CPU] UNMOVED`.

## Verdict

**N16_TWO_AXIS_GATE_FAIL_STOP_BEFORE_N64.** The complete screw is no longer a
near-identity actuator, and it does carry a small amount of pose, but the tested
scene framing and five-class RGB-offset coefficient family are a net loss:
full-screw `d_pose` improves 3.17%, `d_seg` worsens 2.31x, the chart coefficient
stream grows 29.37%, and the composed action loses `+0.272859`. The governed
n16 gate therefore forbids n64 and n600.

Verdict scope: stored-PoseNet-to-SE(3) calibration (`s_t=0.16`, `s_r=1.0`), one
ground homography with off-ground persistence, full-screw one-hot scene-chart
transport, per-class RGB-offset coefficients, the first 16 pairs of this clip,
and macOS CPU advisory execution. This rejects one DOF/level/framing
combination; it does not kill screw advection or chart coefficients as families.

## D1 — full six-coordinate screw

All six stored source coordinates are consumed by

`xi = log_SE3(T(exp_SO3(s_r*[p3,p4,p5]), s_t*[p2,p1,p0]))`.

Every source and xi coordinate is nonzero on all 600 pairs. Full-n600 `xi_l2`
quantiles are `3.7220 / 4.8700 / 4.9830 / 5.1065 / 5.6088`; the planar
predecessor max was `0.05226`. The actuator is now physically moving in its
declared chart. Motion adds zero video-derived bytes beyond the stored pose
sidecar, and the receiver remains scorer/pose blind.

Constants are custodied by `ego_motion_cumulative_se3_bspline_v1`: `s_t=0.16`
is the measured pose-carry anchor, `s_r=1.0` is the derived identity conversion
to `tac.lie` radians, and ground pitch comes from the hash-pinned G1 receipt.

## D2 — n16 hard-oracle result

All n16 rows fall in q4 of the full-n600 magnitude partition (`xi_l2`
5.1320–5.5494); q1–q3 are honestly empty at this prefix.

| Arm | d_seg mean | d_pose mean | Pose delta vs matching static |
|---|---:|---:|---:|
| Static base | 0.007289569 | 187.789519 | — |
| Full-screw base | 0.016856194 | 181.832164 | -5.957355 (-3.17%) |
| Static + chart coefficients | 0.007288297 | 187.789762 | — |
| Full-screw + chart coefficients | 0.016764005 | 181.987147 | -5.802614 |
| #549 solved target | 0.000132879 | 0.000040871 | target reference |

The screw carries pose only weakly toward the target tube, while its Seg side
effect is decisive. The chart RGB offsets slightly help Seg inside the screw arm
but slightly harm pose; they do not reconcile the geometry.

## D3 — chart/coefficient price

The new strict packet stores `int8[pair,5,RGB]` offsets plus one float16 scale
per pair. The decoded PPCS scene chart is the receiver basis. Header, SHA-256,
CRC, parse/re-encode, Brotli decompress, and receiver application all close.

| Arm | Raw packet | Brotli-11 terminal | d_pose | d_seg | action at lambda |
|---|---:|---:|---:|---:|---:|
| Static chart | 632 B | 269 B | 187.789762 | 0.007288297 | 44.063725 |
| Full-screw chart | 632 B | 348 B | 181.987147 | 0.016764005 | 44.336584 |

At `lambda*=6.658589531221714e-7`, the 79 added bytes cost only
`5.2603e-5`; the total `+0.272859` loss is dominated by Seg damage. Against the
planar exact-pixel baseline of 19,739,340 B, both chart packets are over 99.998%
smaller. That comparison is **neither equal fidelity nor equal prefix**: the
chart packet is lossy n16, whereas the pixel exception was target-exact n64.
The authority is the measured `(bytes,d_seg,d_pose)` point, not the attractive
byte ratio alone.

## D4 — U1 / G-pose route

- Einstein-Kolmogorov U1: ingest `(348 B, 181.987147, 0.016764005)` as a
  rejected R-D point, not a positive coefficient.
- P0 G-pose #603: register `FORMULATION_NEGATIVE`; do not waterfill this arm.
- Next crux: the target/framing remains wrong before declaring pose intrinsically
  expensive. Use depth-stratified action (ground homography, rotation-only sky,
  identity hood, explicit movable/object motion) and Fisher-margin
  curvelet/shearlet boundary coefficients. The current single ground-depth chart
  expends coherent motion in the wrong strata.

## Reuse manifest

| Surface | Disposition | Exact use |
|---|---|---|
| `tac.lie` SE(3) exp/log | REUSED | complete xi conversion and homography |
| predecessor `advect_motion_base` | EXTENDED | unchanged transport plus additive full-screw custody |
| #549 reconstruction | REUSED | deterministic scorer-plane targets and factor-2 realization |
| PPCS seed/chart | REUSED | decoded five-class coefficient basis |
| G1/openpilot geometry | REUSED | hash-pinned pitch and EON plane geometry |
| chart RGB coefficient packet | NEW, NARROW | missing strict PPCS-chart coefficient codec; see failed-search receipt |

No quarantined archive, Fourier residual, scorer-in-decoder, or literal pixel
payload was reused.

## Reproduction and custody

```bash
PYTHONPATH=src /Users/adpena/Projects/pact/.venv/bin/python \
  tools/measure_advected_screw6_chartlevel.py \
  --seed /Volumes/VertigoDataTier/pact/evidence/seed_compose_20260721/seeds/seed_compose_b2_loose.ppcs \
  --gt-cache /Users/adpena/Projects/pact/experiments/results/mlx_fleet_gt_cache/gt_n600.npz \
  --upstream /Users/adpena/Projects/pact/upstream \
  --output-dir /Volumes/VertigoDataTier/pact/evidence/advected_screw6_20260721 \
  --pair-end 16 --chunk-size 8 --threads 4
```

Full SSD receipt:
`/Volumes/VertigoDataTier/pact/evidence/advected_screw6_20260721/advected_screw6_chartlevel/receipt.json`,
19,873 B, SHA-256
`e257bc6b9fe3a899e6b1d9b6a4d5b0496129dc4831a3aa25b40d7cf27346b450`.
Sixteen pair stages and two chunk checkpoints are preserved. Three earlier
source-custodied attempts are also preserved losslessly beside the final
evidence; no bytes were deleted.

## STORES CONSULTED

PPCS B2 seed/decoder; `gt_n600` stored RGB and six-coordinate pose; #549 target
memo and source-plane reconstruction; predecessor planar receipt; hash-pinned
G1 worldsheet receipt; `tac.lie`; native upstream CPU-Torch scorer; lane registry;
delegation inbox; SSD evidence tree. MAIN landing review is required.
