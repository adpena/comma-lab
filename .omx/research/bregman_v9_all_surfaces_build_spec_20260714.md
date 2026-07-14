# Bregman v9 all-surfaces build specification — 2026-07-14

Status: `BUILD + LOCAL-VERIFY ONLY`; `$0`; no teacher call, evaluator, live-run
mutation, paid dispatch, or heavy launch. Pointer is immutable in this lane.

## Authority and ownership

- Consume, do not duplicate, `argmax_native_vjp_fidelity_v1`,
  `categorical_bregman_geometry_summary`,
  `extended_kl_mc_summary_from_log_ratios`, and
  `compile_v9_optimal_metric_binding`.
- Do not edit `src/tac/scorer_surrogate/vjp_fidelity.py`.
- Do not edit provenance-owner hot surfaces (`witness_autoconfig`, V9 config,
  trainer bijection, or `spec_v9_cgauge.py`). The new typed binding is handed to
  that owner for composition into the sole V9 DSL path.
- New code is argv-inert and research-only unless its receipt prerequisites are
  satisfied. No standalone helper is represented as a live trainer actuator.

## Build contract

1. Repair the real naïve-MC-KL defect by routing the existing float-returning
   fallback through the canonical extended nonnegative estimator. Preserve
   explicit iid-from-p custody and refuse nonfinite/negative extended results.
2. Extinguish current spatial/generic `batchmean` reduction violations and add
   explicit flat-tensor waivers only where shape custody proves them valid.
3. Add an exact affine-Legendre gauge helper for
   `Fbar(theta)=lambda F(A theta+b)+<c,theta>+d`, with a content-bound receipt
   verifying `B_Fbar=lambda B_F` at transformed coordinates. This proves the
   divergence identity only; it must emit
   `GAUGE_IDENTITY_VERIFIED_NOT_MODEL_FACTORIZED` until the V9 latent model has
   an executable `(xi,R)` factorization receipt.
4. Add a deterministic finite-support Caratheodory reducer and exponential-
   family sigma-point KL verifier. Require positive normalized weights,
   `m<=D+1`, expectation-coordinate parity, and the exact MC-error identity.
5. Add a categorical Chernoff-bisector solver whose endpoint hashes and support
   identifiers bind the exact input bytes. Keep the seg/pose V9 binding custody-
   gated because scalar score terms are not normalized distributions on common
   support; retain the existing score-gradient operating point meanwhile.
6. Canonicalize the already-existing Euclidean waterfill as the one-projection
   instance of the curved-centroid law. Do not add a duplicate solver or claim a
   speedup. Keep TerminalSolve and #423 head-offset substitutions unadmitted:
   the former is not built and the latter is Jensen-invalid.
7. Add one typed `bregman_geometry` V9 policy binding and Lever with no invented
   trainer flags. It must consume the canonical metric binding, enumerate every
   LawRef/consumer/receipt/status, and fail closed on missing live custody.
8. Add canonical equations, local deterministic probe receipts, focused tests,
   an adversarial round-1 review, a DAG FEED, findings, and session summary.

## Measurement labels

- Synthetic NumPy fixtures are `MEASURED [local-CPU math fixture]`, never n600
  score evidence.
- Retained real artifacts may be cited only with bytes/hash/axis custody.
- Full-n600 metric equivalence, Hessian/Gram coupling, sigma-point throughput,
  seg/pose Chernoff, and executable V9 gauge covariance remain explicit
  `NO_VERDICT_*_CUSTODY` states unless the required artifacts exist.
- The historical V9 row `d_seg=0.03482035319010417 @ ep150` scopes only that
  formulation. Missing affine covariance is not asserted as its cause.

## Verification gates

- Focused unit tests for gauge identity, sigma reduction/error identity,
  Chernoff bisector, typed DSL drift refusal, and MC-KL nonnegativity.
- Repository KL-reduction strict preflight returns zero violations.
- Deterministic probe emits stable JSON and repeats byte-identically. Wall-clock
  timing is opt-in, separately content-addressed, and never a canonical constant.
- Ruff/compile checks on owned files, then three clean focused passes.
- Own round-1 review before serializer submission; serializer allowlist only.
