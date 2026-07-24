# DDM CO4 Road-local ranking and propagated-precision preregistration

UTC: 2026-07-24  
Lane: `lane_ddm_co4_road_local_and_precision_20260724`  
Authority: local advisory `_dev`; `research_only=true`;
`execution_allowed=false`; `actuation=NONE`; `score_claim=false`;
`promotion_eligible=false`; MAIN landing review required.

This selection rule is sealed before fitting or scoring either CO4 candidate.
The landed CO3 receipt and its historical held-out metrics are known inputs.
They remain immutable controls rather than being overwritten.

## Population, target, and folds

- Population: exact source pairs `0..599`, one row per pair, from the same
  SHA-bound G3 x EV1 x scorer-value-oracle join used by CO3.
- Target: nonnegative realized receiver-closed Seg/Pose distortion closure from
  EV1. Shared V19 rate bytes remain excluded from pair targets and features.
- Outer and inner folds remain the CO3 pair-held-out SHA256 folds. Every
  fitted scaler, ridge coefficient, hyperparameter, and any expert route is
  learned without the outer held-out target.
- The historical `Road` audit slice is the 288-pair EV1 dominant-closure
  diagnostic. That outcome-derived label may be used to score the held-out
  slice only. It is forbidden as a candidate feature or inference-time router.

## Frozen Road-conditional feature construction

CO4 extends the selected CO3 factorized-plus-MS4D feature vector. It adds only
source-preexisting, target-free scorer-recursive quantities:

1. **PF2/MS4D Road Fisher spectrum.** For each nonempty PF2 bucket containing
   Road, allocate the bucket's measured rank-4 margin-Fisher Gram to a pair by
   its exact MS5/MS6/RG3 `pair_support_counts[pair_id] / event_count`. Sum those
   blocks per pair. Features are log trace, log largest eigenvalue, effective
   rank, Road-boundary trace share, and log Road support.
2. **G3 hard-pair rank.** Consume the existing full-n600 `score_rank` under
   `derived_exact_flip_pose_score_mass`; encode normalized rank and reciprocal
   log rank. No new target ordering is constructed.
3. **G4 within-Road flip frequency.** From the same exact pair-support vectors,
   emit the Road transient fraction and Road static-in-image fraction.
4. **SN1 boundary versus interior.** Emit Road boundary and cell/interior
   support fractions from the PF2 class-stratum partition, plus the interaction
   of Road Fisher trace with the existing scorer-margin gap.

No generic spatial menu, Euclidean norm, or held-out response-derived feature
is admissible.

## Frozen two-candidate race

1. `global_road_conditional`: one ridge model on all training pairs using the
   CO3 selected features plus the frozen Road-conditional features. Ridge
   strength is selected inside each outer training fold from the existing
   CO3 grid.
2. `g3_stratum_experts`: separate ridge experts on the target-free G3 dominant
   pre-outcome class-flip stratum inside each outer training fold. Each held-out
   pair is routed by its already-existing G3 dominant class-flip stratum.
   Strata with insufficient training support fall back to the corresponding
   fold's global Road-conditional model. The target-derived EV1 closure stratum
   is never the router.

Selection is by held-out Road NDCG@4, then global held-out NDCG@4, then held-out
Road Spearman rho, then lower complexity. The selected CO4 prediction replaces
the CO3 prediction only if:

- held-out Road NDCG@4 is at least `0.60`; and
- global held-out NDCG@4 is at least the existing CO3 admission floor `0.75`.

Otherwise CO4 records a formulation-scoped failure and retains the historical
CO3 selected predictions and duty authority. No family-negative verdict is
authorized.

## Wallace/MML pair-precision propagation

The 15 pairs with measured positive MS4D direct blocks remain `DIRECT`.
For every other pair:

1. compose the pair Fisher Gram as the sum of
   `(pair_support_count / bucket_event_count) * bucket_gram` over its nonempty
   PF2 buckets;
2. require exact 600-entry support vectors, nonnegative support, event-mass
   conservation, finite symmetric positive-semidefinite Grams within numerical
   tolerance, and at least one positive pair contribution;
3. label the interval `PROPAGATED`, never `DIRECT`, with assumptions:
   within-bucket exchangeability of event Fisher, additive independent bucket
   blocks, and no unmeasured cross-bucket covariance;
4. apply a Kish-style observed contribution-heterogeneity design effect
   `1 + CV(contribution)^2` to reduce nominal propagated precision. The emitted
   interval must therefore be wider than the same nominal propagated Fisher
   interval without the assumption penalty.

Pairs failing any condition are `UNRANKED` with the exact reason. Pair ordering
is `TIED` whenever adjacent 95% intervals overlap or either interval is
missing. The receipt reports exact counts for `DIRECT`, `PROPAGATED`, and
`UNRANKED`.

## Required diagnostics and authority walls

- Recompute Kalman-style innovations on the actually selected/retained OOF
  prediction; lag-one remains only a scoped diagnostic.
- Pontryagin/Bellman and matched organ/RD1 dual consistency remain typed
  `AWAITING_J8F` until real ordered J8F state transitions and M34 per-state
  duals exist.
- Every DECIDE row carries a canonical Rudin falling-rule explanation.
- The #611 scorer-recursive proposal route is represented as a typed blocked
  DECIDE row; this lane does not create proposal bytes or actuate it.
- MS2R immutable-stage drift is diagnosed against exact organ inputs and
  represented by a typed row; no immutable checkpoint is overwritten.
- All outputs share one state digest with existing consumers. No provider,
  GPU, scorer replay, run mutation, archive mutation, or frontier-pointer
  action is authorized.

## Verdict scope

`INSTANCE:V19_N600_EXACT_RECEIVER_REPLAY_X_G3_X_G4_X_PF2_MS5_MS6_RG3_X_MS4D`;
the Road gate can reject these two frozen formulations only. The Road-local
ranker family and scorer-recursive proposal family remain open.
