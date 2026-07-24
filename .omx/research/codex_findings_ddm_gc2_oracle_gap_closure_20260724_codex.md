# Codex findings — DDM GC2 scorer-value oracle gap closure

Date: 2026-07-24  
Lane: `lane_ddm_gc2_oracle_gap_closure_20260724`  
Implementation commit: `e77f30900dab7cb4b0aefe0235cdc1ec257fc309`  
Verdict: **BUILT_AND_LOCALLY_VERIFIED; MAIN LANDING REVIEW REQUIRED**

## Outcome

All seven historical `TYPED-GAP` rows in `ScorerValueOracle` now have
machine-readable, SHA-bound producers and typed accessors. Live coverage is:

```text
21 WRAPPED / 0 TYPED-GAP / 0 stale-advisory
```

Each new artifact records its schema, authority scope, exact source paths,
source hashes, source byte counts, and any measurement-axis limitation. The
manifest is
`.omx/research/ddm_gc2_scorer_value_oracle_gap_closure_20260724/MANIFEST.json`.

## Seven closures

| Row | Durable result | Epistemic scope |
|---|---|---|
| gain (normalization affine) | Frozen Pose input mean/std and all eight learned scalar AllNorm inference affines | Exact source/weight extraction; no score claim |
| frequency / R passband | Exact finite-matrix bicubic-up then bilinear-down mode gains and tangent/normal comparison | Linear unclamped R geometry; measured 3.2x deficit is separate representation evidence |
| YUV6 luma phases | Ordered `Y00,Y10,Y01,Y11,Ubox,Vbox` preprocessing law | Exact pre-network law; clamp-active derivatives are piecewise |
| chroma pose-null | Exact six-dimensional post-resize 2x2 preprocessing kernel plus scoped n6 uint8 readback | RGB-input-visible, not argmax-changing; camera preimage and receiver closure are NULL |
| null/gauge energy | #580 ker(A) geometry and #519 gauge/null measurements carried together without conflation | macOS-CPU advisory n32 measurements; joint intersection is NULL |
| pose dims 7-12 | Exact upstream first-six-only objective exclusion | Structural exclusion, not network-null semantics |
| score axes + weights | Upstream-SHA-bound score formula and canonical helper linkage | Authoritative scores still require exact upstream eval and contest axes |

## Adversarial corrections

- The sampled R-chain anisotropy is below `0.056%`, so the historical
  `3.125` (~3.2x) along-tangent deficit cannot be attributed to R attenuation.
- The phrase "52% gauge energy" overloaded two quantities. #519 measures a
  gauge fraction of norm of `0.52356`, whose energy fraction is `0.27412`;
  separately it measures rendered output energy in `ker(A)` of `0.52425`.
- The chroma certificate is exact only at the post-resize RGB/YUV6 read surface.
  The scoped n6 fixture finds `442716/589824` bounded-feasible blocks and
  `118304/442716` bit-exact YUV6 readbacks, but this does not certify a
  camera-grid preimage or a receiver-closed actuator for j8f, pa2, or F2.

## Verification

- `75 passed`: `test_scorer_value_oracle.py`,
  `test_ddm_costate_organ.py`, and `test_ddm_campaign_costate.py`.
- Seven-row parameter matrix verifies fresh schema/content reads.
- Every new row has a mutation test proving SHA drift fails closed.
- Ruff and `git diff --check`: clean.
- Two clean entity-level review passes per touched Python file:
  `source-math-custody` and `tests-fail-closed`.
- The GC2 lane is canonically L1 with `impl_complete=true`. Global
  `lane_maturity validate` remains red on 110 pre-existing missing evidence
  paths; none names this lane, so that repository-wide custody debt was not
  laundered into a GC2 gate.

## Triality and pointer honesty

- DSL: no control or launch configuration changed.
- DAG: pinned source receipts -> seven immutable producers -> hash/schema
  admission -> costate consumers, with an explicit camera-authority refusal.
- Equations: exact R composition, YUV6 kernel, first-six Pose slice, and contest
  score functional are source-bound; no canonical score law changed.
- Pointer: **UNMOVED**. No job, GPU, archive, score, or promotion dispatch.

STORES CONSULTED: delegated authority; CLAUDE.md; AGENTS.md; operating manual;
DDM-366 contract; OF1 facade memo; upstream evaluate/modules/frame_utils;
PoseNet weights; contest-score helper; exact R-chain matrices; #580; #519;
3.2x source receipt and scoped follow-up; canonical lane/checkpoint state; both
directive inboxes.

MAIN must independently rehash the seven producers, rerun the 75-test focused
suite, and review the three fail-closed nulls before merging:
`camera_preimage_certified_fraction`, `receiver_closed_fraction`, and
`measured_joint_intersection_energy`.
