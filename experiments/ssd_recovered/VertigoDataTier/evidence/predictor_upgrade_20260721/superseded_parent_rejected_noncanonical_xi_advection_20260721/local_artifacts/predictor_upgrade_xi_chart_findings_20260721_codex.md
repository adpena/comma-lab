# Task #578 findings — xi-advected prior plus executed per-class charts

`lane_id=lane_predictor_upgrade_xi_chart_578_20260721` · `research_only=true` ·
`[macOS-CPU advisory]` · `score_claim=false` · `promotion_eligible=false` ·
`pointer=0.1910828242 [contest-CPU] UNMOVED` · `MAIN_REVIEW_REQUIRED=true`

## Verdict

`PREDICTOR_TARGET_MISSED`.

The exact measured Task #578 formulation decoded the finite Task #595 packet,
rendered its coherent arc-length/dash mask for each pair with the canonical
`analytic_lane_render_band` helpers, and reconciled Lane before MyCar and
Movable. The packet is
`coherent_slot_none_dash.lbnd2`: 159,386 raw bytes, 41,303 binding Brotli-11
bytes, SHA-256
`d2b2a62eeb6ebe45cbf908dafa7e081eabddaca0f424faac970b41eea650d810`.
Its custody is `executed_in_task578_measurement_external_counted_custody` with
`execution_scope=task578_measurement_only` and `receiver_closed=false`.

This is cell-description evidence. The prior field is the explicitly labeled
oracle stand-in `lstars[t-1]`; it is not serialized. No camera-RGB realization,
receiver closure, frozen scorer run, contest score, promotion, or pointer move
is claimed. The negative is scoped to this measured sequential reconciliation
policy and does not close the broader xi/chart or sibling PROJECT families.

## D1/D2 measured satisfaction

| stage | Road | Lane | Undrivable | Movable | MyCar | overall |
|---|---:|---:|---:|---:|---:|---:|
| n64 `MEASURED_DEVELOPMENT_PREFIX` | 0.960871030110 | 0.748580615988 | 0.991633226621 | 0.998143476945 | 0.998371843217 | 0.984907865524 |
| n600 `MEASURED [macOS-CPU advisory]` | 0.922695718466 | 0.699462381939 | 0.986157668493 | 0.996465855203 | 0.999132512305 | 0.973161476983 |

The ≥0.99-every-class target is missed at both stages. At n600 the exact
correct/total rows are Road 25,288,364/27,407,046; Lane 483,076/690,639;
Undrivable 57,604,705/58,413,281; Movable 1,455,164/1,460,325; and MyCar
29,967,490/29,993,509. Every one of the 3,166,001 n600 misses has one exclusive
cause in the receipt. Causes follow the executed stage trace: absent Lane masks
are chart misses; later priority overwrites are adjacency/tie misses; Movable
misses are track misses; critical sites remain critical; and
Road/Undrivable/MyCar distinguish absent chart support from later overwrite.

The n64-trained PXCH payload is 221,195 raw bytes, 835 diagnostic zlib-9 bytes,
SHA-256 `2b3665c47f7a404e7ac8ea1b30cad768d4ce2a84fd998e167d230b522e18ba43`.

## D3 measured all-counted-section curve

The final advisory objective prices `PPCS raw + PXCH raw + 41,303 Lane-chart
bytes`. The earlier undercounted objective is superseded and is not a final row.

| point | PPCS bytes | PXCH | Lane | total counted | rate term | d_seg after | advisory non-score objective | >216,222 B |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| loose | 544,804 | 221,195 | 41,303 | 807,302 | 0.537549264573 | 0.026824476454 | 3.219996909973 | yes |
| knee | 877,741 | 221,195 | 41,303 | 1,140,239 | 0.759238346849 | 0.026810548570 | 3.440293203891 | yes |
| tight | 1,542,222 | 221,195 | 41,303 | 1,804,720 | 1.201688969879 | 0.026783362495 | 3.880025219336 | yes |

All PPCS objects parse and reserialize byte-identically, name
`xi_advected_prior_per_class_charts.v1`, and carry the packet SHA plus the
measurement-only/non-receiver-closed execution scope.

## D4 measured inventory and blocker

The n600 mask-only boundary-normal R0 stream is 9,570,800 raw bytes; the sum of
the 38 independently resumable chunk zlib-9 sizes is 2,723,787 bytes. An
equal-fidelity through-R R1 execution is absent, so the exact D4 verdict remains:

`FORMALIZATION_PENDING_THROUGH_R_REALIZED_SCORE_RECOVERY`

Its scope is the equal-fidelity through-R comparison only; no mask F1 is
promoted into d_seg and both predictor/R1 families remain open.

## Resumability and boundaries

- n64 config SHA-256: `3f0bb25e9ad3208a9c4253084edf5907ac6317612389dca214923f396d490f43` (4 chunks).
- n600 config SHA-256: `53005263f381e2ada4bb75faa46aefbd889386db293976ff540c201f35437adc` (38 chunks).
- Durable root: `/Volumes/VertigoDataTier/pact/evidence/predictor_upgrade_20260721/`.
- The rejected unexecuted-Lane run is preserved losslessly in
  `superseded_parent_rejected_lane_unexecuted_20260721/`; earlier superseded
  trees remain preserved too.
- `src/tac/optimization/predict_project_receiver.py` is untouched at SHA-256
  `6f3704726c57f9e02e628a792127d0ce16f1979e27097bbede003542942dbd1f`.

Machine receipt: `.omx/research/predictor_upgrade_xi_chart_measurements_20260721.json`,
SHA-256 `431fc66a3055e38033b12b8c154a6ab2ae8b16ae1fb8726c101d33abd178479f`.

## STORES CONSULTED

Task #578 build spec and current receipt; Task #595 LBND2 packet and canonical
renderer; predecessor B2 artifacts; G1 LawRefs; #139 hood and #234 movable
components; frozen n600 cache; current and superseded SSD evidence trees;
CLAUDE/AGENTS vehicle contracts.
