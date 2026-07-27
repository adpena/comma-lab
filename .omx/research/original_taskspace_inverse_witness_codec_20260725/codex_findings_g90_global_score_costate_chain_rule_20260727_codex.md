# G90 — costates must differentiate the population score, not per-pair proxies

Date: 2026-07-27  
Axis: exact evaluator-equation review plus deterministic unit proof  
Verdict scope: pair ranking and actuator pricing built from
`compute_pair_scorer_gradient(component="pose"|"combined")`. This does not
invalidate VJPs, costates, or scorer-gradient selection as families.

## Finding

`upstream/evaluate.py` computes:

`D_pose = (1/N) * sum_i d_i`

`S_pose = sqrt(10 * D_pose)`

The existing local diagnostic instead differentiates `sqrt(10*d_i)` separately
for each pair. That changes relative pair prices by a factor proportional to
`1/sqrt(d_i)` and therefore overweights already-low-distortion pairs. It is not
the costate of the contest score.

The exact multiplier for a raw per-pair pose-MSE VJP is shared across all pairs:

`dS_pose/dd_i = 5 / (N * sqrt(10 * D_pose))`

For the exact G85 n600 object (`N=600`,
`D_pose=163.06130981`), the multiplier is
`0.00020636844449905425`.

`src/tac/optimization/scorer_gradient_sparse_residual.py` now exposes:

- `compute_pair_pose_mse_vjp`: differentiates the upstream per-sample pose MSE
  before any square root;
- `global_pose_score_costate_scale`: derives the exact complete-population
  chain-rule multiplier; and
- `scale_pair_pose_mse_vjp_for_global_score`: forms the scorer-native pose
  costate without dtype drift.

Finite-difference tests prove the derivative after the population mean, and
fail-closed tests reject zero/negative/nonfinite distortion and non-exact
sample counts.

## Segmentation and rate boundaries

The evaluator's segmentation term is an argmax disagreement count, not
cross-entropy. `seg_ce_weight=0.05` is therefore an arbitrary proposal heuristic
and must not be combined with the pose VJP as if it were score units. The
scalable path is:

1. compute target-vs-current logit-gap VJPs at mismatched cells;
2. project complete realized-through-R actuator interventions into those gaps;
3. use the projection only to screen or order a compact shortlist;
4. measure exact argmax flips and exact ZIP bytes on each complete n600 state;
5. admit only by the nonlinear score.

Rate has no honest per-pair smooth derivative. Its costate is the exact serialized
byte delta of the composed physical stream, including shared dictionaries and
outer coding.

## Triality

DSL:

`POSE_COSTATE(pair_i) = VJP(d_i) * 5/(N*sqrt(10*mean(d)))`

DAG:

`complete base row -> raw pair-MSE VJPs -> global score scale -> actuator projection -> exact n600 archive replay`

Equations:

`delta S(theta) ~= <lambda_pose, delta R(theta)> + exact_seg_flips(theta) + 25*delta_bytes/37_545_489`

The approximation is a shortlist mechanism only. G83 exact whole-state
arbitration remains the verdict.

## Pointer-delta honesty

No candidate or score was produced. The effective frontier remains `0.172`.
This landing removes a controller mispricing bug before the expensive
population costate materialization.

## Stores consulted

- `upstream/evaluate.py`
- `upstream/modules.py`
- exact G85 n600 evaluator report, SHA-256
  `75e85ae2a75748423f4c74e592982a9bf61d07e3b0ef7fee294934a71a6bde7c`
- `src/tac/optimization/scorer_gradient_sparse_residual.py`
- `tools/run_scorer_gradient_sparse_residual_smoke.py`
