# Build spec — ORGAN / n=1 / RL warm-start cluster (2026-07-14)

**Lane:** `lane_warmstart_organ_n1_rl_20260714`  
**Scope:** `$0`, read-only real-artifact backtest, no trainer/provider dispatch, no score claim.  
**Pointer:** unchanged.  
**Coordination:** new organ-side files only. The provenance owner retains
`preflight.py`, `canonical_equations/**`, `witness_dsl/**`, autoconfig, trainer config, and
the provenance bijection.

## Premise and divergence

The real corpus is one contest clip with ten verdicts, nine dependent intervals, and seven
deployment-faithful walk-forward folds. Existing Q/P arms provide physics-structured prior means but
use one scalar ridge and expose no posterior covariance. Existing T is a useful total-forcing GP
forecaster but deliberately exposes no per-lever response field. The implementation target is therefore
not another neural network or another GP:

`U_hierarchical_physics_residual = fixed P/Q physics response prior + conjugate block-partial-pooling residual`.

The persistent response stream is centered on the fixed P/Q prior direction. A separate prediction
stream fits only the residual. This is the n=1 transfer of SPS's state/prediction separation and of the
actor-critic baseline/advantage decomposition; it is a predictive organ, not an identified actor or a
causal effect estimator.

## Mathematical contract

For interval target `y_t = dx_t / d_epoch` and the existing design row
`z_t = [1, x_t, sum_i u_ti phi_i]`, let `C0(q)` be the coefficient prior for fixed prior mode
`q in {Q_iso, P_aniso}`. Coefficients are grouped into intercept, state-drift, class-response,
nonclass-response, and byte blocks. With a preregistered block multiplier vector `g` and scalar
precision `alpha`:

```text
P(alpha) = alpha * blockdiag(g_intercept I, g_state I,
                             g_class_response I, g_other_response I, g_byte I)
Sigma_beta = (Z^T Sigma_eps^-1 Z + P)^-1
beta_hat = Sigma_beta (Z^T Sigma_eps^-1 Y + P C0)
y_hat_* = z_* beta_hat
Cov(y_hat_*) = z_* Sigma_beta z_*^T + Sigma_eps
```

`alpha` and P/Q mode are selected only from a tiny preregistered grid by inner prequential error on
the outer fold's past prefix. The first computable fold uses the declared Q/default precision, never
the target. NumPy-fp32 is the verdict reference. MLX implements the identical solve and reports parity;
absence of MLX is explicit, not silently coerced.

VR-GHAL transfers only a diagnostic clipped-difference transform
`clip(delta_t, L ||z_t-z_(t-1)||)` plus a median anchor. Its stochastic fixed-point convergence theorem
is not claimed because the organ has neither an unbiased multi-query stochastic operator nor repeated
seeds. FORE transfers a support certificate, not an occupancy-ratio estimate: no behavior/target
propensities means `BLOCKED_DISTRIBUTION_CUSTODY`. TOFU-POV transfers a prefix-frozen SVD basis and
effective-rank/conditioning certificate, not OFUL regret: the corpus lacks masked offered slates and
random coordinate reveal. HCM transfers partial pooling, while its causal gate is false because there
is no within-unit randomized treatment variation. Grokking transfers the eigensystem/effective-mode
clock of the closed-form ridge posterior; no iterative-GD late-generalization claim is made. Continual
learning remains an external posterior update until at least three independent trajectory lineages.

## Owned files

1. `src/tac/witness_control/costate_warmstart_cluster.py`
   - pure deterministic NumPy-fp32 posterior solve and optional MLX parity solve;
   - U model, prefix-only selector, support/causal/CL certificates, effective-rank diagnostics;
   - no imports from trainer, subprocess, provider, or witness DSL.
2. `src/tac/witness_control/tests/test_costate_warmstart_cluster.py`
   - posterior algebra, covariance PSD, deterministic ties, target-leak mutation, support fail-closed,
     optional MLX parity, and real-fixture walk-forward smoke.
3. `tools/probe_costate_warmstart_cluster.py`
   - read-only real-run probe; atomic JSON receipt outside the run directory; baseline and ablation rows.
4. Dated findings, standalone DAG FEED, receipt, and session summary under `.omx/research/`.

## Acceptance and adversarial gates

- identical seven outer folds and exact class weights as `lambda_net.backtest`;
- compare persistence, A, Q, P, T, and U; report aggregate and per-class WF MAE, fold wins/sign test,
  posterior uncertainty, effective degrees of freedom, support rank, and selected grids;
- mutate each held target while preserving the prefix and prove the fold prediction is byte-identical;
- distinguish predictive identifiability from causal attribution; HCM/FORE/TOFU/RL claims fail closed;
- a miss is `INSTANCE x FORMULATION` only, with structured-GP residual, change-point blocks,
  frozen #434/meta encoder + tiny readout, and independent-trajectory accrual left open;
- no adoption from one mean-only win; `GRADUATION_MIN_RECORDS=3` remains binding;
- final integration request names one v9 DSL factory, held LawRef, trainer/organ consumer, receipt schema,
  and provenance edge for serial wiring by the exclusive owner.

## Held v9 triality contract

- **Factory (held):** `costate_hierarchical_physics_residual_v1_spec()` returning the typed organ arm
  `U_hierarchical_physics_residual`, with no invented CLI flag.
- **LawRef (held):** `costate_hierarchical_physics_residual_v1`, whose payload binds the block precision
  grid, prior modes, prefix-only selection rule, NumPy/MLX tolerance, and receipt schema.
- **Consumer (held):** the existing costate-organ tournament/dispatcher invoked by v9 autoconfig after
  telemetry extraction; never the witness renderer or archive payload.
- **Receipt (held):** `costate_warmstart_cluster_backtest.v1`, content-bound to the trajectory log,
  fold definitions, seed, module hash, selected hyperparameters, and all baseline rows.
- **DAG (standalone):** warm-start queue to FEED-426/433/434/436; main serially appends it after owner
review. No shared DAG/hot-file mutation in this lane.

## Inbox amendments and value-provenance ladder

The 2026-07-14T15:48:11Z inbox amendment adds requential coding. The exact REC/PAC-Bayes
construction is not portable because the organ lacks normalized generative teacher/student
distributions and a prefix-free proposal/index stream. The held transfer is therefore a Gaussian
posterior disagreement curriculum, explicitly `NOT_REC_CODE`: all real prefix rows retain half of
their unit replay mass and the reallocated half is capped at two units. `0.5` and `2.0` are
**SPECIFIED research constants**, chosen before the amended receipt to preserve coverage and prevent
one of at most nine dependent intervals from becoming a repeated dataset; they were not optimized.

The 2026-07-14T15:52:31Z inbox amendment adds HSE/router diagnostics. HSE is computed exactly over
single-linkage cosine thresholds on held-out skill profiles. The primary bounded skill map
`exp(-error/persistence_error)` has no fitted scale; reciprocal and fold-minmax maps are retained as
representation-sensitivity ablations. Router perturbations are exactly one NumPy-fp32 ULP in each
direction, not an invented percent-noise ladder.

Other constants:

- `BLOCK_MULTIPLIERS=(0.25,8,1,16)` are **SPECIFIED**, not DERIVED: class response is the unit
  reference, intercept is allowed four-times more residual freedom, while state and nonclass response
  are shrunk eight/sixteen-times harder under n=1. The uniform `(1,1,1,1)` ablation is mandatory.
- `MIN_INNER_SELECTION_FOLDS=4` is a **POST-HOC FORMULATION REPAIR** after the one-fold selector
  produced a catastrophic next-interval miss. Every repaired result is therefore an adaptive
  development-WF result, not independent confirmation.
- `PRECISION_GRID=(0.01,0.1,1,10)` and Q-first tie law are inherited from the existing organ ridge
  scale; no continuous search is allowed.
- Gaussian disagreement variance is posterior predictive variance. Any value below fp32 epsilon is
  floored and counted; a floor-hit code proxy cannot be called a capacity floor.

New owned files added by the inbox amendments:

- `src/tac/witness_control/costate_requential_curriculum.py`
- `src/tac/witness_control/tests/test_costate_requential_curriculum.py`
- `src/tac/witness_control/costate_society_diagnostics.py`
- `src/tac/witness_control/tests/test_costate_society_diagnostics.py`
