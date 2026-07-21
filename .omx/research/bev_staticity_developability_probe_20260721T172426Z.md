# BEV staticity / developability probe — confound-gated n64 → n600

**Verdict:** `NO_VERDICT_C1_HOOD_CONTROL_FAILED` at the load-bearing n600 scale.  The rigid
MyCar/hood full-silhouette positive control is not static in its canonical ego-identity chart:
MEASURED p50 ruling residual **7.75 scorer px** and only **0.143475** of samples lie within the
registered **1 px** floor.  Per C1, Road/Lane residuals are diagnostic only and MUST NOT be read as
evidence that ground geometry is non-static or non-developable.  D3 therefore emits no byte ratio.

Axis is `[macOS-CPU advisory]`; seed 1234; `score_claim=false`; `promotion_eligible=false`; pointer
`0.1910828242 [contest-CPU]` UNMOVED.  MAIN landing review is required.

## Exact custody and method

- Frozen target: `gt_n600.npz`, SHA-256
  `cf8d83605d2198ef56786c6be23d3470033ad2763f59559f06a79cedfb7b8cd6`, fields `lstars` and
  `margins`; this is the frozen CPU-Torch scorer target cache used by the #549 solved-cell line.
- ξ is NOT fitted here.  The probe decodes the already-solved PPCS spline/AR trajectory with
  `counted_planar_xi_series`; source seed SHA-256
  `a21dde38128bed7ff62860ef005b994b74202e0bd00a37d1df8824ee325e856b`.
- #327 reconciled geometry is fixed at MEASURED `v_h=174` and openpilot device height `1.22 m`;
  scorer intrinsics come from `clip_profile.camera_for_resolution(1164,874)`.
- Class indices are self-detected on the n64 frozen stack with `detect_class_order`; the detected
  map is the canonical `Road/Lane/Undrivable/Movable/MyCar` order and is reused unchanged at n600.
- Each unlike-label edge is oriented by the Fisher-margin field.  Only its shallow side enters
  D1-D3, and its subpixel coordinate is `t=M_p/(M_p+M_q)` (registered #275 law); deep-side counts
  remain in custody.
- D2 deliberately does not emit raw numerical Gaussian K.  The C3-safe statistic is directrix +
  frozen-ξ ruling reconstruction residual, compared with the 1 px discretization floor.
- Every frame is an atomic resumable JSON stage on the SSD.  No scorer forward, GPU, paid dispatch,
  archive mutation, or run-pointer mutation occurred.

## MEASURED D rows

### C1 positive control

| scale | authority | hood p50 / p90 residual (px) | static fraction ≤1 px | residual dynamics | events / static segments | gate |
|---|---|---:|---:|---:|---:|---|
| n64 | directional only | 6.50 / 57.2780 | 0.142222 | 0.857778 | 0 / 1 | FAIL |
| n600 | load-bearing | 7.75 / 79.00 | 0.143475 | 0.856525 | 32 / 19 | **FAIL** |

The second n64 attempt used one scorer row per bin and the complete MyCar silhouette; the earlier
48-bin attempt is preserved at `canonical_v1` but invalidated because it introduced an ~8-row
aggregation confound.  The corrected failure is therefore not that binning artifact.  The remaining
C1 root cause is scoped to the control surface: incorrect ego-canonicalization and frozen-argmax
MyCar-boundary dynamics are not disambiguated here.

### n600 oriented shallow-side diagnostics — NOT Road/Lane verdicts

| stratum | shallow / excluded-deep samples | p50 / p90 ruling residual (px) | ≤1 px fraction | residual-dynamics fraction | events / static segments | polynomial order |
|---|---:|---:|---:|---:|---:|---|
| Road | 722,312 / 789,922 | 39.6465 / 181.8127 | 0.047354 | 0.952646 | 14 / 8 | >3 or unmatched |
| Lane | 448,715 / 373,347 | 46.5501 / 188.8249 | 0.044258 | 0.955742 | 3 / 4 | >3 or unmatched |
| Undrivable | 192,083 / 199,201 | 12.2571 / 150.7500 | 0.091122 | 0.908878 | 20 / 13 | >3 or unmatched |
| Movable | 96,828 / 94,478 | 8.0336 / 52.9774 | 0.203785 | 0.796215 | 39 / 28 | >3 or unmatched |
| MyCar oriented | 159,980 / 162,968 | 8.5000 / 84.3549 | 0.117369 | 0.882631 | 27 / 17 | >3 or unmatched |

The Movable row is directionally consistent with the expected non-static control, but C1 prevents it
from ratifying the ground transform.  These residual-dynamics fractions retain curvature changes,
forks, dash rhythm, per-pair separatrix asymmetry, and independent object motion; none is relabeled as
noise or as “the video is static.”

## D1 / D2 / D3 dispositions

- **D1:** `BLOCKED_C1_HOOD_CONTROL_FAILED`.  No Road/Lane near-static verdict is authorized.
- **D2:** `BLOCKED_C1_HOOD_CONTROL_FAILED`; raw K remains unreported.  The diagnostic ruling
  fractions above are not developability verdicts.
- **D3:** `ESTIMATE_ONLY_NOT_RUN_NO_HOOD_GATED_D1_D2`.  No lossy-cheap versus lossless-expensive
  comparison and no fake matched-distortion ratio were emitted.  A future positive still owes a
  g2g2-style real-homography decode + frozen-scorer admission on the same #549 cells.
- **Routing:** U1/U2, U5, and P0 activation are all false.  The only route is the blocker/reactivation
  edge in the sibling DAG FEED.

## Triality / apparatus

- **Equations:** `t=M_p/(M_p+M_q)`; openpilot IPM
  `f=H f_y/(v-v_h), lateral=-(u-c_x)f/f_x`; cumulative translation-first SE(3) from `tac.lie`;
  C3 ruling residual `|boundary - (directrix ⊗ ξ)|` in scorer-pixel equivalents.
- **DAG:** `.omx/research/bev_staticity_developability_probe_DAG_FEED_20260721T172426Z.md`.
- **DSL:** research-only measurement CLI; no trainer flag or launch config exists, so no DSL lever is
  invented.  Any future decoder actuator must enter through the governed typed DSL after admission.
- **Pointer delta:** exactly zero.

## Evidence and git proof

- n64 SSD receipt: `canonical_v3/receipt_n64.json`, file SHA-256
  `184707e736c019c3b19ef5ff80ff8bb9254b41130b685113eaaf68573a7dc62e`, internal canonical SHA
  `e833b28c8bd4ac2f8064958b9bcd9fcd837e6888780189703c91a9b7c4025186`.
- n600 SSD receipt: `canonical_v3/receipt_n600.json`, file SHA-256
  `8177eec78152209dcf8def9cb7a6c73c79dcd82a79e7622b397084fb5d0d9ea6`, internal canonical SHA
  `74e7b5d46a2763a33fde225ee3bf4388944b8aca38b0805624b572345ae2bccd`.
- SSD root: `/Volumes/VertigoDataTier/pact/evidence/bev_staticity_developability_20260721/`.
- Base git SHA before landing: `31184ccdf38c5192054017d1cabbe6342f2d7dfa`.

## STORES CONSULTED

`CLAUDE.md`; `AGENTS.md`; v7.5/v8 SPECs; `reports/latest.md`; lane/subagent registries; #325/#327
openpilot geometry sources; #549 joint-solve/C1 receipts; `gt_n600.npz`; seed-compose receipt/reuse
manifest; `clip_profile`; `lane_sdf_component`; `ground_frame_chart`; `ego_xi_trajectory`;
`predict_project_receiver`; canonical equations registry (#275/Fisher-margin); latest live inbox and
broadcast directives.

`verdict_scope`: the failed positive-control surface at n64/n600.  This is NOT a negative on static
ground polynomials, ruled-worldsheet representations, a corrected transform, a physical hood mask,
or any admitted receiver implementation.
