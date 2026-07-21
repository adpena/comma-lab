# Task #578 findings — canonical G1 projective xi advection

`lane_id=lane_predictor_upgrade_xi_chart_578_20260721` · `research_only=true` ·
`[macOS-CPU advisory]` · `score_claim=false` · `promotion_eligible=false` ·
`pointer=0.1910828242 [contest-CPU] UNMOVED` · `receiver_closed=false` ·
`MAIN_REVIEW_REQUIRED=true`

## Exact formulation and verdict

`PREDICTOR_TARGET_MISSED`.

The active `xi_advected_prior_per_class_charts.v2` formulation executes G1's
canonical plane-induced homography path. `s_t` and `s_r` resolve through the
existing LawRefs; `pitch_rad` is loaded from the exact hash-validated G1 receipt
at `g1/transport_assumptions/calibration/pitch_rad`. For transition `t>0`,
`gt_poses[t]` is consumed directly by `xi_from_pose_calibration`, the prior
categorical field is converted to five one-hot channels, and
`warp_frame0_native_numpy` applies `GroundHomographyGeom` with persist fallback.
Deterministic first-argmax decodes the warped labels. No absolute trajectory is
fabricated and no already-relative twist is re-differenced.

The G1 receipt explicitly lacks exact cross-pair PoseNet targets, so
`gt_poses[t]` is only the nearest-target-pair proxy. This result is therefore an
oracle-prior, proxy-motion cell-description measurement, not receiver-closed or
score evidence.

## D1/D2 measured satisfaction

| stage | Road | Lane | Undrivable | Movable | MyCar | overall |
|---|---:|---:|---:|---:|---:|---:|
| n64 `MEASURED_DEVELOPMENT_PREFIX` | 0.965392866314 | 0.755978998141 | 0.991880686176 | 0.998188881041 | 0.997565054154 | 0.985902229945 |
| n600 `MEASURED [macOS-CPU advisory]` | 0.924312018158 | 0.707667826462 | 0.986206253335 | 0.996571995960 | 0.998831013737 | 0.973533749051 |

The n600 correct/total rows are Road 25,332,662/27,407,046; Lane
488,743/690,639; Undrivable 57,607,543/58,413,281; Movable
1,455,319/1,460,325; and MyCar 29,958,447/29,993,509. All 3,122,086 misses
have exactly one causal bucket in the sequential policy receipt.

The Task #595 packet was hash/raw/Brotli/zlib validated, decoded, and rendered
per pair before Lane reconciliation. Custody is 159,386 raw bytes, 41,303
binding Brotli-11 bytes, raw SHA-256
`d2b2a62eeb6ebe45cbf908dafa7e081eabddaca0f424faac970b41eea650d810`,
and diagnostic zlib-9 47,546 bytes with SHA-256
`ef16824eea59415e71435b94c450c2d554e0db08c0981fae6b392ab08170d287`.
Only the 41,303 binding bytes are counted in D3.

## D3 measured all-counted-section curve

Each total is exactly `PPCS raw + 221,195 PXCH raw + 41,303 Lane Brotli`.

| point | PPCS bytes | exceptions | total counted | d_seg after | rate term | advisory non-score objective | >216,222 B |
|---|---:|---:|---:|---:|---:|---:|---|
| loose | 544,710 | 1,655 | 807,208 | 0.026452221341 | 0.537486673832 | 3.182708807892 | yes |
| knee | 879,071 | 3,306 | 1,141,569 | 0.026438225640 | 0.760123939257 | 3.403946503276 | yes |
| tight | 1,546,709 | 6,528 | 1,809,207 | 0.026410912408 | 1.204676679001 | 3.845767919778 | yes |

All three v2 PPCS objects parse/re-serialize byte-identically and remain nested.
The zlib Lane diagnostic is reported but not added to these totals.

## D4 and custody boundary

The SAME n600 boundary inventory contains 938,050 records: 9,380,500 raw bytes
at 10 bytes/record, with independent chunk zlib-9 sizes summing to 2,686,108
bytes. The measured exception-coding baseline corrects all 938,050 records and
leaves zero errors on this mask-only inventory. With no R1 execution, the
top-level comparison correctly leaves `corrected_cells=0` and
`remaining_errors=938050`; every R1 result field remains `null`. Equal-fidelity
through-R R1 custody remains absent, so the exact verdict is
`FORMALIZATION_PENDING_THROUGH_R_REALIZED_SCORE_RECOVERY`.

The fully fresh corrected generation is rooted at
`/Volumes/VertigoDataTier/pact/evidence/predictor_upgrade_20260721/canonical_g1_d4_fixed_20260721/`.
Both stages ran with `--no-resume` in strict n64 then n600 order. Its config
hashes are n64
`d4860734107b066246a0cc1e305be76f1539ba91e8f722856778a4a502244be6`
and n600
`0859b6f17000820f436e9dd5831b886f6f750511ab237ae1f2716d77ef16bc58`.
The measured source SHA-256 is
`a429f0ca855d1d7abb6481ba8f7298f07504987fd8d6c3a063bdc692ab34cdae`.
The receiver remains untouched at
`6f3704726c57f9e02e628a792127d0ce16f1979e27097bbede003542942dbd1f`.

The entire rejected flat-shift generation is preserved under
`superseded_parent_rejected_noncanonical_xi_advection_20260721/` with manifest
SHA-256 `066a87c750c84411df9a1cd296ce264f727d2896cc4ff993ba9a1834753fe6d6`.
All earlier superseded generations remain preserved.

Machine receipt: `.omx/research/predictor_upgrade_xi_chart_measurements_20260721.json`,
SHA-256 `7df2a4adb7bfc62d48baec829d87296b4156ea2b433347a3dddd093ee69af236`.

## STORES CONSULTED

Task #578 spec plus Round 2 amendment; exact G1 receipt and LawRefs; canonical
`warp_real_luma_frame0`; Task #595 LBND2 packet/canonical renderer; predecessor
B2 seeds; #139 hood and #234 movable components; frozen n600 cache; current and
all superseded SSD evidence trees; CLAUDE/AGENTS vehicle contracts.
