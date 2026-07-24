# DDM CO3 N600 lambda refit — canonical equations

Status: advisory `_dev`; `actuation=NONE`; MAIN review required.

## E1 — exact instance target

For pair `i`, using receiver-realized EV1 endpoints:

`DeltaD_i = 100*(d_seg_after_i - d_seg_before_i)
            + sqrt(10*d_pose_after_i) - sqrt(10*d_pose_before_i)`

`y_i = max(0, -DeltaD_i)`

The shared V19 candidate byte delta is not multiplied into pair targets.
`y_i` is an advisory ranking target, not a contest score.

## E2 — deterministic pair-held-out split

`fold(i) = int(SHA256("ddm-co3-n600-v1:" || decimal(i))[0:8], 16) mod 5`

Every reported selection metric is computed once from the concatenated
out-of-fold predictions. Pair IDs are exactly `0..599`.

## E3 — fold-local ridge forms

For each candidate feature map `phi_c`, training fold `T`, and ridge alpha:

`z_i = (phi_c(x_i) - mean_T(phi_c)) / std_T(phi_c)`

`w = (Z_T^T Z_T + alpha I)^(-1) Z_T^T (y_T - mean_T(y))`

`yhat_i = mean_T(y) + z_i^T w`

Feature normalization and margin/hardness decile thresholds are fitted only
inside the training fold. Alpha is selected by deterministic inner three-fold
Spearman, then NDCG@4, then smaller alpha, exactly as preregistered.

## E4 — held-out selection and admission

`winner = argmax_c (NDCG4_OOF(c), Spearman_OOF(c), -complexity(c))`

`duty_upgrade_eligible = [NDCG4_OOF(winner) >= 0.75]`

Observed winner:

`factorized_ms4d_interactions`

`NDCG4_OOF = 1.0`

`Spearman_OOF = 0.8607149751465011`

No in-sample metric has admission authority.

## E5 — nested close-form mixture

Within each outer training split, order forms by inner held-out NDCG@4, then
Spearman. If the best two NDCG values differ by at most `0.02`, map their
outer-test predictions to training-reference percentile ranks and average:

`yhat_mix = (rank_T(yhat_a) + rank_T(yhat_b))/2`

Otherwise the mixture uses only the inner winner. The outer test targets do
not select mixture members.

## E6 — aggregate-derived G4 mixture

For temporal class `c`:

`q_i(c) = [B_i*p_boundary(c) + I_i*p_interior(c)] / (B_i + I_i)`

where `B_i,I_i` are G3 pair boundary/interior counts and the class fractions
come from aggregate G4 custody. The dominant label is diagnostic only; it is
not an exact pair-level G4 measurement.

## E7 — Wallace/MML pair precision

Only for a pair with positive direct MS4D Fisher trace `F_i`:

`se_i = sigma_OOF / sqrt(F_i)`

`CI95_i = [yhat_i - 1.96*se_i, yhat_i + 1.96*se_i]`

Overlapping adjacent intervals emit `TIED`. Missing `F_i` emits
`UNRANKED_PRECISION_OWED`. Current coverage is `15/600`; pair duty ranking is
therefore blocked even though aggregate held-out admission passed.

## E8 — innovation health

With pair IDs in ascending order:

`e_i = y_i - yhat_i`

`rho_lag1 = corr(e_0..e_598, e_1..e_599)`

Observed `rho_lag1 = 0.06679580189709491`. The verdict is scoped to lag one;
it is not a complete whiteness proof.

## E9 — Bellman and dual consistency standing rows

When adjacent J8F costates and transitions exist:

`r_Bellman,t = lambda_t - (partial L_t/partial x_t
                           + J_t^T lambda_(t+1))`

Until then the Bellman row is null.

When a pair/dimension crosswalk and non-null RD1 dual exist:

`r_dual = lambda_organ - lambda_RD1`

and consistency requires a measured uncertainty band. Current RD1 has 0/162
actionable prices, so the row is null rather than zero.

## E10 — one state identity

`receipt_content_sha = SHA256(canonical_json(receipt_without_content_sha))`

`state_digest = SHA256(canonical_json(core_campaign_state))`

The ranker receipt file hash and compact validated state participate in the
campaign digest. Digest, dashboard, duty, and nag consumers must carry the
same `state_digest` or fail closed.
