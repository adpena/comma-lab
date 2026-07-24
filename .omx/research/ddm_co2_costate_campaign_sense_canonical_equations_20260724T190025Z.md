# DDM CO2 campaign costate — canonical equations

Status: advisory `_dev`; `actuation=NONE`; MAIN review required.

## E1 — exact campaign score

For exact receiver-realized quantities:

`S = 100*d_seg_R + sqrt(10*d_pose_YUV6_R) + (25/37,545,489)*B_archive`

No proxy quantity may be labeled `S`.

## E2 — endpoint marginal score per byte

For a measured endpoint transition with positive counted-byte increment:

`r_D = (D_left - D_right)/(B_right - B_left)`

`lambda_score_per_byte = 25/37,545,489 - r_D`

This is a restricted non-additive scalarization control. It is not a bucket
price unless the candidate-delta x dimension x counted-byte foreign keys are
closed.

## E3 — typed bucket price admission

`lambda_bucket = NULL`

unless all of:

1. scorer-metric block is MS4D-custodied;
2. candidate delta is assigned to the exact G4/dimension bucket;
3. receiver-closed uint8 effect is measured;
4. counted-byte home is exact and not duplicated across pairs.

Null is evidence, not zero.

## E4 — same-regime noise alarm

Given `n >= 2` measured noise samples in the latest `noise_regime_id` and a
preregistered family-wise error `alpha`:

`k = Phi^{-1}(1 - alpha/(2*n))`

`alarm_threshold = k * sample_stdev(delta_S_noise)`

No default `alpha`, `k`, or threshold exists.

LawRef: `ddm_campaign_familywise_noise_alarm_v1`.

## E5 — evaluator-band top-K

Let `b*` be the evaluator interval of the candidate with lowest measured
center `delta_S`. Then:

`K = |{i : evaluator_band_i intersects b*}|`

No candidate bands means `K=NULL`.

LawRef: `ddm_campaign_evaluator_band_overlap_top_k_v1`.

## E6 — scoped trust radii

- G2F native-pixel amplitude knee is the SHA-guarded measured value 1.0.
- V16 coupled-solve linearization is invalid for radius promotion.
- V17 boundary-normal radius is the largest consecutive lattice prefix with
  both positive realized reduction and positive realized/predicted ratio:
  q=1,2 pass; q=4 fails; therefore the scoped radius is q=2.
- MS7 PF3 class-birth radius is the measured instance value q=1.

These units and formulations are not interchangeable; no universal minimum is
formed.

## E7 — state and consumer identity

`state_digest = SHA256(canonical_json(core_campaign_state))`

Every consumer view carries that exact digest and refuses a mismatch. Receipt
content hashes and future J8F event-stream hashes participate in the core
state, and the organ resume checkpoint carries those source hashes.

## E8 — plateau routing

`decision = FEED603[plateau_type](typed_scorer_residual)`

where the residual has `{residual_type, metric_id, value, units}`. Unknown
types block; known types emit advisory F1–F7 rows; neither path actuates.
