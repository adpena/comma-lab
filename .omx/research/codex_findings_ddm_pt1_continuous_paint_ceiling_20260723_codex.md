---
title: Codex findings - DDM PT1 continuous paint ceiling
date_utc: 2026-07-23
lane_id: lane_ddm_pt1_continuous_paint_ceiling_20260723
research_only: true
execution_allowed: false
score_claim: false
evidence_axis: "[macOS-CPU frozen-scorer advisory]"
verdict: PREPARED_NOT_EXECUTED
verdict_scope: "BUILD/PREPARE only; no PT1 arm was measured"
pointer_moved: false
main_landing_review_required: true
---

# Outcome

**PASS build; NO empirical PT1 verdict.** The exact-n600 measurement surface is
implemented and behavior-tested, but the delegated authority and checked-in
typed config both have `execution_allowed=false`. No frozen scorer was run, no
`d_seg <= 0.0142` claim was made, and no formulation was confirmed or
falsified.

| Row | FIRST-RUNG | Errors | Sites | d_seg | Status |
|---|---:|---:|---:|---:|---|
| E2 native-grid flat paint | yes | 3,349,482 | 117,964,800 | 0.028393910726 | MEASURED inherited anchor |
| PT1 hard camera placement | yes | — | 117,964,800 | — | PREPARED_NOT_EXECUTED |
| PT1 analytic coverage blend | yes | — | 117,964,800 | — | PREPARED_NOT_EXECUTED |
| PT1 global amplitude-statistics match | yes | — | 117,964,800 | — | PREPARED_NOT_EXECUTED |
| PT1 stratum-spectrum diagnostic | yes | — | 117,964,800 | — | PREPARED_NOT_EXECUTED |

Receipt:
`.omx/research/ddm_pt1_continuous_paint_ceiling_20260723/ddm_pt1_prepared_receipt.json`.

# Controlling mechanism split

MAIN's later directives supersede the original composite-AA framing. The meter
keeps these rows separate:

1. **Primary — hard-amplitude camera placement.** The exact target partition
   is converted to signed-distance fields. The existing #275
   `gt_tie_targets_numpy` margin-ratio localizer moves eligible zero crossings
   before fields are evaluated at `(874,1164)` camera pixel centres. The
   renderer selects one class and writes an exact full-amplitude palette
   prototype; no blend crosses uint8.
2. **Secondary — analytic coverage blend.** The same fields drive an analytic
   signed-distance coverage blend before one uint8 round. This is a separate
   v14-adverse control. Brute supersampling is absent.
3. **Third arm — global amplitude statistics.** Geometry remains the E2
   flat-paint geometry. One global RGB mean/variance affine is stored in an
   exact parse-backed 30-byte payload (six float32 scalars plus magic).
4. **Spectrum diagnostic.** Per-stratum RGB coefficients are fit against GT
   over the existing texture-trunk period-4/6/8 basis, restricted to the
   measured through-R/stem-surviving passband. Its 45 float32 coefficients
   have an exact parse-backed 186-byte payload. The global-statistics affine
   is composed first, geometry remains unchanged, and this isolates
   local-spectrum/region-ERF sensitivity from the third arm.

The E2 control uses the pinned first half of R—bicubic up to camera resolution,
round/clamp/uint8—and every candidate delegates the bilinear downsample to the
real frozen `SegNet.preprocess_input`.

# De-arbitraried falsifiers and diagnostics

- **Mechanism bar:** compare primary placement recovery with an independently
  measured, SHA-bound survival-wall receipt from this vehicle. The checked-in
  config deliberately leaves that receipt null; even an authorized successor
  execution fails closed without it. The 16% SINE-family prior is not reused.
- **Box bar:** report primary post-paint `d_seg <= 0.0142` and signed
  `d_seg - 0.0142`. This is secondary evidence.
- **Four-way decomposition:** disjoint operational corrected-error counts for
  `sub_cell_placement`, `bn_se_amplitude_statistics`,
  `texture_prior_or_region_erf`, and `class_interaction`. A
  texture-dominated residual routes to the texture family rather than becoming
  a geometry-family negative.
- **Curve provenance:** every transition is split by
  `already_described_curve_sites` versus `freshly_fitted_curve_sites`.
- **Scorer-native profile:** every evaluated variant reports per-layer
  `layer`, `channel_group`, `spatial_band`, `delta_norm`,
  `delta_norm_relative`, and Fisher-weighted delta against the SHA-bound GT
  frame. Within-batch and preserved cross-batch feature-trajectory errors cover
  every consecutive sample.
- **Depth of first divergence:** every final-argmax failure identifies the
  earliest captured layer with an exact static or feature-trajectory
  divergence. Stem entry is labeled as having no downstream-only correction;
  later entry identifies the preceding feature-relay surface.

Every stream×stage row closes
`errors_after = errors_before + errors_introduced - errors_corrected`.

# Description ownership and byte honesty

DV1-near target boundaries reuse the already-counted spline owner. Every
remaining target boundary is charged as a complete SDWL1 typed-section,
causal-delta, outer-zlib object with exact parse-back. The prepared receipt uses
`null`, never `0`, for that future fitted-geometry byte count. Both streams
carry the four-clause audit triple: descriptive/compact/coder decomposition,
scorer visibility, sensitivity-priced tolerance, single ownership, and no
duplicate correction stream.

# Pose secondary

The ξ branch remains `PREPARED_NOT_ENABLED`. It requires a separately
custodied decoder-side ξ-to-camera warp and deterministic
amplitude-structured paint. Flat advected paint is explicitly rejected as
pose-blind. A future enabled receipt must compare static versus ξ-advected
structured paint using PoseNet embedding delta as the primary currency and
per-layer consecutive-frame trajectory stability as supporting evidence.

# Typed implementation and triality

- `src/tac/optimization/ddm_continuous_paint_ceiling.py`: margin-localized
  signed-distance geometry, camera-centre placement, analytic coverage,
  amplitude statistics, measured-passband spectrum/warp preparation, exact
  description debt, four-way attribution, scorer-native profiles, and stage
  conservation.
- `tools/measure_ddm_pt1_continuous_paint_ceiling.py`: strict SHA-bound config;
  immutable prepared receipt; exact-n600-only future execution; full target
  cache hash; frozen scorer custody; 38 preserved batch checkpoints plus
  activation endpoints; execution and independent-wall fail-closed gates.
- **DSL:** `DDMPT1ContinuousPaintCeilingConfigV1`.
- **DAG:** SHA-bound target/description inputs → owner-split margin-localized
  geometry → non-composed paint arms → camera uint8 → official scorer →
  layer/trajectory profiles → exact stage ledger and falsifiers.
- **Equations:** `E_after=E_before+E_introduced-E_corrected`;
  `t=M_p/(M_p+M_q)`; `d_flip=|margin|/||Delta w||`;
  `S=100*d_seg+sqrt(10*d_pose)+25*B/37,545,489`.

# Verification

- Python compile: PASS.
- Ruff focused lint: PASS.
- Focused behavior suite: **16 passed**.
- Focused plus AA-SDF, phase, texture-trunk, SDWL1, and through-R component
  suites: **135 passed, 4 skipped**.
- Covered: exact source-partition preservation, canonical sub-cell tie
  placement, prototype-only hard writes, separate analytic blend, exact
  30-byte amplitude payload, deterministic 186-byte spectrum payload, four-way
  decomposition, scorer-native profile/trajectory and first-divergence depth,
  curve-owner
  partition, stage conservation, positive SDWL1 debt, execution refusal, and
  absence of fake prepared outcomes.
- No n600 scorer, PoseNet, paid dispatch, GPU, or candidate evaluation ran.

# STORES CONSULTED

- `CLAUDE.md`
- `AGENTS.md`
- `PROGRAM.md`
- `docs/operating_manual_craft_handoff.md`
- `.omx/research/t5_crucible/SPEC_v75_optimal_single_trunk_20260708.md`
- canonical state ledgers and latest E2, DV1, DV2/SDWL1, v14, and v19c records
- per-arm inbox through `2026-07-23T19:11:51Z`
- fleet broadcast through `2026-07-21T13:15:53Z`

# MAIN landing review required

Before merge, MAIN must independently verify:

1. hard placement emits palette prototypes only, while analytic AA, global
   amplitude matching, and spectrum matching remain non-composed rows;
2. #275 tie-position and pinned R coordinate laws are used as declared;
3. the externally measured survival wall is truly independent and SHA-bound
   before any execution bit is enabled;
4. DV1-reused versus target-fitted boundary ownership is exhaustive, disjoint,
   and fitted geometry can never receive a zero-byte claim;
5. activation hook names match the frozen scorer, first-divergence depth uses
   declared layer order, and all consecutive-frame endpoints remain preserved
   across resume;
6. the mechanism labels are operational attribution, not causal Shapley
   claims; and
7. no score, promotion, or pointer mutation is inferred from this prepared
   receipt.
