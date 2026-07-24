# DDM CO3 N600 lambda-ranker preregistration

UTC: 2026-07-24  
Lane: `ddm_co3_lambda_refit_full_join`  
Authority: advisory `_dev`; `actuation=NONE`; `score_claim=false`;
`promotion_eligible=false`; MAIN landing review required.

This selection rule was written before fitting or evaluating any candidate
below. The already-landed full-join control values (Spearman
`0.7476669456024575`, NDCG@4 `0.19557065696692438`) are known inputs, not
candidate results.

## Exact population and folds

- Population: the one fresh, exact EV1 receiver-closed join of source pairs
  `0..599`, joined one-to-one to the fresh G3 atlas. No subset, resampling,
  synthetic row, site-as-pair duplication, or old eight-pair result is
  admissible.
- Unit of holdout: `source_pair_id`. All rows/features derived from one pair
  stay in one fold.
- Five deterministic folds:
  `fold(pair_id) = SHA256("ddm-co3-n600-v1:" + decimal(pair_id))[0:8] mod 5`.
  A fold is scored only by a model fitted on the other four folds.
- Reported selection metrics are computed only from the 600 concatenated
  out-of-fold predictions. In-sample metrics may be diagnostic but have no
  selection or admission authority.
- Primary metric: held-out NDCG@4. Tie-break: held-out Spearman rho. Final
  tie-break: lower form complexity in the order listed below.

## Frozen candidate race

All learned forms are labeled `[advisory-heuristic]`.

1. `factorized_refit`: ridge refit of the existing factor family in a
   log-stabilized coordinate system. Inputs are exact gap, usable support,
   receiver helpful/changed ratio, and inverse baseline-byte price. Ridge
   values are selected inside each training fold from the fixed grid
   `{1e-6,1e-4,1e-2,1,100}` by deterministic inner three-fold Spearman.
2. `factorized_ms4d_interactions`: candidate 1 plus explicitly named
   scorer-metric interactions: gap x margin-Fisher stratum mass, support x
   stationarity, helpful ratio x pose-tube activity, and hardness x
   margin-decile. Only fresh oracle-backed quantities or exact joined
   first-rung counts may populate them.
3. `g4_regime_conditional`: candidate 2 plus G4 temporal-regime intercepts
   and regime x factor interactions. Unknown or unjoinable regimes remain a
   typed missing-feature row; they are never imputed from labels.
4. `small_monotone_gb`: a deterministic small boosted-stump ranker on the
   same candidate-3 features, limited to 32 depth-1 stumps, shrinkage `0.05`,
   fixed quantile split grid `{0.1,...,0.9}`, and monotone directions fixed
   from the DDM factor law before fitting. It is raced only if candidates
   1-3 all have held-out NDCG@4 below `0.75`.

No candidate may consume the held-out target during feature construction,
normalization, hyperparameter choice, missing-value handling, or regime
assignment. The exact receiver-closed positive Seg/Pose distortion closure is
the target. Shared V19 rate bytes remain one global home and are excluded from
pair targets and pair features.

## Admission and duty gate

The selected form is the candidate with greatest held-out NDCG@4, then
Spearman, then lower complexity. It may upgrade campaign duty ranking only if
its held-out NDCG@4 is at least `0.75`. Otherwise:

- the organ remains weak-advisory;
- no campaign duty is reranked by the learned output;
- a formulation-scoped blocker records the exact held-out result;
- J8F realized-verdict telemetry remains independently blocking.

No outcome in this lane moves a contest pointer, launches work, or changes
promotion authority.

## Pre-fit Pantheon addendum

Directive consumed at `2026-07-24T22:30:57Z`, after the initial
preregistration was written and before any candidate was fit or scored.

- Pontryagin/Bellman: emit a standing adjacent-lambda residual row. It is
  measured only when ordered adjacent J8F lambda estimates exist; otherwise it
  stays `AWAITING_J8F_MEASUREMENT`.
- Dual consistency: compare organ and RD1 estimates only on identical typed
  homes with two non-null estimates and a measured comparison band. Current
  null RD1 prices remain a typed awaiting row, never a zero.
- Wallace/MML: attach Fisher-derived uncertainty when the fresh MS4D producer
  supplies a valid pair-level precision join. Duty comparisons with
  overlapping intervals are `TIED`; missing precision blocks interval claims.
- Kalman innovations: every outer-fold prediction emits its held-out
  innovation. Report lag-1 correlation and typed decompositions by fold,
  stratum, G4 class, margin decile, and pair-hardness decile. These are
  diagnostics, not new selection objectives.
- Rudin falling-rule explanations: reuse the landed explain surface if its
  callable contract accepts this typed feature schema without lossy
  adaptation. Otherwise emit a reasoned exclusion and a compact native rule
  chain; do not create a parallel generic explainer.
- COVER/bandit allocation: design note only. No queue actuator or exploration
  spend is authorized in this lane.
- Compression progress per effort: surface as the primary *future* duty
  currency when J8F supplies the required measured wall-clock and score rows.
  Missing J8F leaves the current queue unchanged.
- Close-form mixture: candidate selection is nested inside every outer
  pair-held-out fold. If the best two inner held-out NDCG@4 values differ by
  at most `0.02`, their outer predictions are combined by equal-weight average
  normalized rank. The mixture itself is then scored only on its outer-held-out
  rows. No full-population candidate score may choose an OOF mixture weight.
