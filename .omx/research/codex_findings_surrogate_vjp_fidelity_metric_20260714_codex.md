# Codex Findings — Surrogate VJP Fidelity Metric — 2026-07-14

`research_only=true`  
Lane: `lane_surrogate_vjp_fidelity_metric_20260714`  
Authority: `[macOS-CPU advisory; NumPy-fp32 training-gradient evidence; no score authority]`  
Verdict scope: `INSTANCE-OF-FIRST-CUT-FORMULATION x REAL-N600-SOURCE-HELDOUT-120`  
Pointer delta: **NONE**

## Adversarial verdict

The inherited raw RGB costate cosine is a measurement-locus artifact.  Pulling both
costates through the real 19-parameter renderer raises cosine from
`0.001416` to `0.017697` for round 2 and from about `0.00167` to about
`0.0857` for round 3.  The magnitude-sensitive same-learning-rate descent
fraction remains only `0.002304` and `0.004648`, respectively, with relative
L2 approximately one.  These retained first-cut instances remain below the
static license gate.  That is an instance observation, not a technique verdict.

The optimal centered-logit joint value+Jacobian decision/Fisher/functional
student was never fit: its prior campaign receipt has `n_pairs=0`,
`teacher_calls=0`, and no real-n600 cache manifest.  It remains live in the
explicit reformulation queue with measurement status `UNMEASURED`.

## Round-1 findings and dispositions

| Severity | Finding | Disposition |
|---|---|---|
| HIGH | Directly applying `P H` to an RGB input costate is type-invalid. | Replaced by renderer pullback `J_R^T` and typed decision operator `A_theta=W^(1/2) C D_x T D_theta R`. |
| HIGH | Raw cosine discards both reachable geometry and update magnitude. | Gate reports `rho`, norm ratio, relative L2, and `eta=rho*r`; exact identical-state functional parity remains final authority. |
| HIGH | Retained receipts cannot support decision/Fisher/KL/flip/one-step claims. | Every unavailable row is an explicit `NOT_MEASURED_*` custody blocker; nothing is reconstructed from hashes or scalar reductions. |
| HIGH | Initial probe recomputed row hashes but did not compare them with the original campaign seals. | Fixed: all 360 rows must match campaign path, SHA-256, and byte count. A one-byte mutation now fails closed. |
| MEDIUM | Zero surrogate VJP was rejected as malformed instead of recognized as catastrophic fidelity. | Fixed: `rho` is undefined, norm ratio and `eta` are zero, relative L2 is one, and replacement remains refused. |
| MEDIUM | Importance correction initially trusted declarative metadata. | Fixed: the policy receipt, support/density hashes, and exact ratio/mask array hashes are mandatory. Uniform or unsealed weights fail closed. |
| MEDIUM | Renderer uplift was initially described as null-space removal alone. | Corrected: anisotropic `J_R^T` also reweights reachable directions, so the measurement proves locus sensitivity but not that sole cause. |
| HIGH | A low checkpoint could be mistaken for feature poverty before optimization terminality. | Added a fail-closed terminality receipt requirement: fixed quadratics need a terminal-gradient/curvature bound, range-space/null custody, and an exact heldout ridge ladder; nonlinear charts need their own certificate. |
| HIGH | The legacy Metal fitter optimizes boundary-masked ambient RGB Sobolev cosine/L2, not the corrected objective. | Host handoff refuses `MODE=refit-measure` with `BLOCKED_IMPLEMENTATION`; no semantically false timing command is emitted. |

## Ranked gate

1. Identical-state, trust-radius-matched exact one-step/short-trajectory
   functional parity: licensing authority.
2. Low-margin winner–rival directional fidelity under the reachable decision
   operator: necessary argmax-native first-order gate.
3. Renderer/optimizer pullback `rho`, norm ratio, `eta`, and relative L2:
   strongest retained diagnostic and joint-Jacobian training target.
4. Finite-step KL and correctly typed Fisher primal/dual geometry: secondary
   quotient checks.
5. Ordinal/recos/sign concordance: basis-dependent diagnostic only.
6. Raw RGB cosine: historical baseline only.

No threshold was guessed.  A future governed host run must preregister its
repeat floor, trust radius, non-inferiority band, and holistic facet gates.

## Fresh-evidence custody

The `$0` probe authenticates the real source video, frozen n600 cache lineage,
campaign manifests, and each original heldout row seal.  It evaluates exactly
the deterministic n120 split `0,5,...,595`; it does not relabel that split as a
fresh n600 evaluation.  Full RGB costates, logits/probabilities, and applied
perturbation outcomes were cleanup-certified as deleted, so advanced metrics
remain `BLOCKED_DATA_CUSTODY`.

Canonical receipt:
`.omx/research/surrogate_vjp_fidelity_metric_remeasurement_20260714.json`,
SHA-256 `c4116ff0b9af3284b00e90980f693f98be3c11b30eada0ac13bb395cf50c3753`.

Fleet handoffs through 2026-07-14T11:57:05Z agree that full ordinal is not a
replacement metric, FORE/importance sampling must wait for real
density/support custody, and every first-cut negative is instance-only with the
optimal form named.  No duplicate estimator or fifth arm was created.

## Verification

- 89 focused and inherited scorer-surrogate regression tests passed.
- Ruff passed for every new Python surface.
- The default host command completed the `$0` authenticated remeasurement.
- No Metal, teacher, paid, heavy, live-run, or score evaluation was attempted.

## Exact remaining blocker

`BLOCKED_IMPLEMENTATION` (execution status, not a technique verdict): the
corrected centered-logit value+directional-
Jacobian refit driver and expanded real-n600 sufficient-statistic cache do not
exist.  Therefore corrected fidelity and `C_S,VJP`, update cost `U`, Fisher/KL,
winner–rival, and exact functional parity are still unmeasured.  Only after
that resumable, stage-checkpointed driver lands may MAIN run approved Metal
timing and apply the amortized budget gate.
