# TOFU-POV for duty-to-measure ranking and exact-teacher allocation

**Date:** 2026-07-13 UTC
**Role:** SOL xhigh deep-math reader/designer
**Status:** `DESIGN (MEANS)`, `research_only=true`, uncommitted, no launch, no training
**Review status:** `UNREVIEWED`
**Authority:** controller/advisory only; never an evaluator score or surrogate promotion
**Pointer delta:** `UNMOVED`

## Answer first

| application | scoped verdict | linear reward | low-rank action subspace | paper-style random action-coordinate missingness |
|---|---|---|---|---|
| 1. Costate / duty-to-measure ranker | **`FEED-costate-controller`**; worth a default-off rank-adaptive comparator only after the three fit gates below | **YES locally and regime-conditionally** for the first variation of `-Delta S` from one frozen reference state; **NO globally** across topology/stage/argmax events | **UNESTABLISHED**: the measured/derived witness state dimension `~8` does not prove that lever-effect vectors have rank 8 | **BREAKS on the current store semantics**: the owed effect is unknown because the lever was not run; that is missing reward/outcome, not a random pre-pull mask of an action vector |
| 2. `#455` pay-or-trust exact-teacher allocation | **`FEED-455`** only as a frozen-epoch, randomized-audit, inverse-probability correction layer; direct TOFU-POV transplant is **NO-GO** | **NO** for raw argmax `d_seg` or the global trust/pay return; **YES only for a certified local smooth teacher-residual chart** | **UNESTABLISHED** for the online residual stream | **BREAKS for deterministic uncertainty querying** because missingness is chosen by the controller; it can be repaired for oracle unbiasedness with a predictable probability `p_t > 0`, but that is a new derived wrapper, not the paper's theorem |

**DERIVED, conditional dimension advantage for application 1.** Under the explicitly assumed
catalog-sized effect descriptor dimension `D = 72`, assumed lever-effect rank `r = 8`, and the
OFUL-style leading term
`R_T = O_tilde(d sqrt(T))`, replacing ambient OFUL by latent OFUL changes the leading dimension
factor by `D/r = 72/8 = 9x`. At the illustrative horizon `T = 72`, the unnormalized leading
proxies are `72 sqrt(72) = 610.9403` and `8 sqrt(72) = 67.8823`. This is **not** a 9x measured
run saving and **not** the paper's full bound: TOFU-POV also pays a missingness/subspace-estimation
term, and the current repo has not established either `D=72` as a feature dimension or `r=8` as
the lever-effect rank. If the relevant theorem normalization instead places dimension under the
square root, the same substitution gives only `sqrt(72/8) = 3x`; the paper body must be checked
before a canonical theorem constant is registered.

**DERIVED after reconciling the landed `#462` memo:** for the *current live* `#455` formulation,
theorem-certified teacher-forward calls saved are `0`, hence `0%`. The allocation identity
`1-mean(p_t)` below is a conditional design law for a repaired fixed-oracle formulation, not a
current saving. Likewise, `K=20 => 95%` is the pre-existing target cadence, not a consequence of
either paper.

## Evidence labels and source-access boundary

- **MEASURED** below means a read-only re-derivation from current repo code/state or an already
  custody-bearing sibling receipt. No new empirical run was made.
- **DERIVED** means algebra from named inputs and displayed assumptions.
- **INFERRED** means a structural mapping between the paper model and Pact.
- **ASSUMED** means an unproved bridge required to make that mapping.

The official arXiv records were resolved for:

1. Gautam Dasarathy, Vineet Gattani, and Lalit Jain, *Stochastic Linear Bandits with Partially
   Observed Actions*, arXiv:2607.08971v1 (2026),
   <https://arxiv.org/abs/2607.08971>.
2. Jelena Diakonikolas, *Solving Stochastic Fixed-Point Equations with High Probability*,
   arXiv:2607.09097v1 (2026), <https://arxiv.org/abs/2607.09097>.

**NO-FAKE access note.** The official abstracts were read. The PDF/TeX bodies were not retrievable
through this sandbox's web cache, network-disabled CLI, unavailable in-app browser, or disallowed
local browser control. Therefore this memo imports only claims exposed by the official abstracts
and the operator-supplied paper synopsis. It does not invent paper theorem constants, mask notation,
or conditioning exponents. The `9x` computation is a separately labeled standard OFUL-style
derivation, not represented as a verbatim TOFU-POV theorem evaluation.

## 1. Current Pact truth re-derived before analogy

### 1.1 What the rankers actually do

- **MEASURED from the live canonical readers:** `duty_to_measure_ranked()` currently returns `77`
  significance rows, of which exactly `72` are registered owed lever measurements. The separate
  curriculum pool has `28` owed rows. Thus the prompt's `~72` is current for the lever queue, but
  the combined lever-plus-curriculum obligation is `100`; they are different action types and must
  not be silently pooled.
- **MEASURED from `activation_ledger.py`:** the display rank is
  `est_delta_s / (S_current - 0.15)`, with P8 term-floor capping when a measured current term and a
  measured floor exist. Unknown estimates sink below estimated rows.
- **MEASURED from `producer_bridge.py`:** the actual never-fired EIG bridge refuses to fabricate
  `Delta S`; under an uninformative prior it ranks owed levers by known DSL `epochs_delta` cost
  ascending, i.e. `1/cost`.
- **MEASURED from `control_alphabet.py`:** a different PowerPlay acquisition surface ranks a
  feature-structured lambda field by `curiosity * blast_radius / cost`. Never-fired rows are
  explicitly `PARTIAL`, not measured effects.

TOFU-POV would therefore not replace one single settled rule. It would be a fourth, default-off
comparator joining (a) P8 relative significance, (b) honest cost-only EIG, and (c) partial
lambda-field PowerPlay. Its potential value is calibrated cross-lever generalization with an
uncertainty bonus; its failure mode is laundering assumptions into a more elaborate score.

### 1.2 The `~8` dimension does not yet establish the paper's subspace

- **MEASURED/DERIVED repo anchor:** the lane/witness orbit has an approximately eight-dimensional
  state chart; Whitney gives `2*8+1 = 17`, with two gauge-margin coordinates giving mod-19 for the
  SDF-like CGauge representation.
- **INFERRED negative:** this is a statement about the witness/state manifold and an embedding
  dimension. TOFU-POV needs the *action vectors presented to the bandit* to lie in a well-conditioned
  low-rank linear subspace. A nonlinear 8-manifold does not imply that the 72 heterogeneous lever
  response vectors span a linear rank-8 subspace.
- **INFERRED negative:** encoding 72 lever identities as one-hot vectors gives ambient dimension
  72 but rank 72, not rank 8. The `D=72, r=8` comparison is valid only if a separate 72-coordinate
  effect descriptor is shown empirically to have stable rank about 8. It cannot be obtained by
  naming the arm count twice.

`verdict_scope:` this negative rejects the unproved `Whitney state dimension => lever action rank`
implication. It does not reject low-rank lever effects as a family.

## 2. What TOFU-POV requires

The model exposed by the paper abstract/synopsis can be written schematically as

```text
full action:       x_t,a in R^D, with x_t,a = U z_t,a and rank(U)=r << D
observed action:   M_t,a x_t,a, where M_t,a reveals a random coordinate subset
reward:            y_t = <theta_star, x_t,a_t> + eta_t
representation:    estimate U from masked actions; freeze U within an epoch
decision:          impute z_t,a under frozen U; run OFUL in R^r
```

The epoch freeze matters: without it, the coordinates used in the confidence design matrix move
with the same adaptive history that OFUL is trying to control. Rank adaptation removes the need to
know `r` in advance. The official abstract says the regret is `sqrt(T)` and scales with intrinsic
rather than ambient action dimension, while an additional missingness-dependent cost remains and
the lower bound separates it from ordinary reward-learning uncertainty.

The minimum fit gates for Pact are therefore:

1. **Reward realizability:** one fixed parameter must make expected reward approximately linear in
   the latent action coordinates over the decision epoch.
2. **Subspace estimability:** the full action descriptors must truly share a well-conditioned,
   sufficiently excited low-rank linear subspace.
3. **Mask semantics:** coordinates must be observed before action selection under a random,
   bounded-away-from-zero observation process of the type used by the theorem. Unknown reward after
   selecting a lever is ordinary bandit feedback, not partial action observation.

## 3. Application 1 — costate / duty-to-measure ranker

### 3.1 Where the linear reward hypothesis survives

Freeze a common campaign state/checkpoint `q_e`, stage, evaluator geometry, and score weights for
decision epoch `e`. Let lever `i` induce a small state displacement `Delta q_i`, exact score change
`Delta S_i`, and known measurement cost `C_i > 0`. On one smooth chart,

```text
-Delta S_i
  = -<lambda(q_e), Delta q_i> + O(||Delta q_i||^2)
  = <theta_e, z_i> + epsilon_i,
```

where `z_i` is a latent lever-effect coordinate and `theta_e` absorbs the costate and the chart
Jacobian. The nonlinear pose term is locally differentiable for `d_pose > 0`, with score costate
`5/sqrt(10 d_pose)`; the rate term is exactly linear in bytes. Thus:

- **DERIVED:** local linearity is principled for small, isolated levers from one frozen state.
- **BREAK:** argmax cell crossings, target-class birth/death, lever interactions, curriculum-stage
  changes, optimizer-memory changes, and saturation at a measured P8 floor create discontinuities or
  a changing `theta_e`.
- **BREAK:** dividing the response by heterogeneous cost inside the regression changes the noise
  and can destroy linearity. Fit raw improvement `g_i=-Delta S_i`; divide the optimistic bound by
  known `C_i` only in the acquisition index.

`verdict_scope:` global linear reward is NO-GO for the whole mixed queue. Local, fixed-regime
first-variation reward remains open and is the only allowed arm.

### 3.2 Where action partial-observation holds or breaks

| hypothesis | current fit | reason |
|---|---|---|
| action coordinates visible with a random mask before pull | **BREAK** | current owed rows chiefly have static DSL metadata and an unobserved effect because no training run has occurred |
| reward observed only for selected action | **HOLD** | a chosen lever's expensive training/eval reveals its realized effect; this is ordinary bandit feedback |
| repeated/diverse action panels identify a subspace | **UNESTABLISHED** | one fixed heterogeneous catalog and `n=1` real trajectory do not certify excitation, eigengap, or conditioning |
| one stationary reward parameter during an epoch | **POSSIBLE only with common-checkpoint forks** | sequentially changing the source state makes the lever reward surface nonstationary |
| bounded, approximately centered noise | **UNESTABLISHED** | training outcomes contain seed, regime, and topology variation; the current organ explicitly reports 30x fold spread |

If all pre-pull lever descriptors are fully observed, ordinary rank-adaptive latent OFUL is the
right comparator and TOFU imputation adds nothing. If only post-run effects are missing, the exact
model is closer to a linear contextual bandit, active experimental design, or combinatorial
pure-exploration problem. TOFU-POV becomes literal only after a real pre-pull masked effect descriptor
is defined and its random-mask contract is measured.

### 3.3 Proposed guarded ranker law

Within epoch `e`, freeze the reference checkpoint and representation `U_e`. Let `z_hat_ei` be the
imputed latent action, `V_e` the OFUL design matrix, `beta_e` the reward-confidence radius,
`epsilon_imp_ei` the separately derived imputation radius, `H_ei` the measured P8 headroom for the
lever's target axis, and `C_i` its known measurement cost. A safe *proposal* index is

```text
U_ei = <theta_hat_e, z_hat_ei>
       + beta_e ||z_hat_ei||_(V_e^-1)
       + epsilon_imp_ei

I_ei = min(H_ei, max(0, U_ei)) / C_i.
```

Unknown or non-authoritative `H_ei` means “do not cap,” not a guessed floor. Missing cost, missing
mask propensity, failed subspace-conditioning, or out-of-regime action means fail closed to the
current P8/cost-only ranker. The P8 cap is a controller governor outside the paper theorem; no full
TOFU regret claim survives that nonlinear clipping without a new proof.

**Epoch transaction:** collect exact rows during epoch `e`, but do not update `U_e` or rewrite old
coordinates. At the boundary, atomically checkpoint the complete observation matrix, masks,
propensities, SVD/rank decision, `U_(e+1)`, design matrix, reward fit, and next-arm queue. Rank-adaptive
selection is preferred because `r=8` is not known for lever effects.

### 3.4 Intrinsic-versus-ambient regret derivation in the requested numbers

Use the standard OFUL-style leading term and isolate the paper's additional missingness debt:

```text
ambient comparator: R_amb(T)  = O_tilde(D sqrt(T))
latent proposal:    R_lat(T)  = O_tilde(r sqrt(T)) + R_missing(T; masks, conditioning, D, |A|)
```

Under **ASSUMED** `D=72`, **ASSUMED** `r=8`, and illustrative `T=72`:

```text
D sqrt(T) = 72 sqrt(72) = 610.9403
r sqrt(T) =  8 sqrt(72) =  67.8823
leading-factor ratio     =   9.0000x
coordinates removed      = 64 / 72 = 88.8889%
```

This is a dimension-factor comparison, not a finite-horizon performance prediction. With rewards
normalized to a bounded range, either displayed proxy can exceed the maximum possible cumulative
regret at `T=72`, so the asymptotic upper bound can be numerically vacuous even while the 9x symbolic
ratio is correct. `R_missing` can erase the gain when observation probability is small or the
subspace is ill-conditioned. The current heuristic has no comparable regret certificate, but a
theorem for a misspecified model is worse than no theorem.

### 3.5 Admission experiment — no empirical claim in this memo

No training was launched. A future arm is admissible only if all of the following are real:

1. Build a typed action descriptor and record its ambient dimension `D`; do not substitute arm
   count. On historical *real* measured lever effects, report singular values, stable rank,
   eigengap/conditioning, and reconstruction error. `r~8` is accepted only if measured here.
2. Identify which descriptor coordinates are available before a pull. Record masks and observation
   propensities. If the only missing item is reward, run ordinary latent OFUL and mark TOFU-POV N/A.
3. Use common-checkpoint, common-config A/B forks so one epoch has a stationary reward surface.
   Every selected arm's endpoint is the exact receiver-realized, NumPy-fp32, `n600` score components
   and archive bytes; proxy reward alone cannot graduate the ranker.
4. Chronologically compare current P8 significance, cost-only EIG, PowerPlay, rank-adaptive latent
   OFUL, and TOFU-POV only if gate 2 holds. Report cumulative reward per cost, simple regret, top-k
   discovery, and exact runs to first confirmed improvement with paired uncertainty.
5. Adopt only on a real held-out win that survives the current costate real-only walk-forward rule.
   Synthetic or simulator-only rankings remain `research_only=true`.

This is controller exploration efficiency feeding `#247/#426` SENSE -> DECIDE. It does not directly
change the 95%-kill wall clock, archive bytes, evaluator score, or pointer.

## 4. Application 2 — `#455` surrogate pay-or-trust allocation

### 4.1 Adversarial model fit

The proposed action is “trust the cheap provider or pay the exact frozen-SegNet teacher.” That is
not automatically TOFU-POV:

- **BREAK, raw reward:** exact `d_seg` is an argmax Hamming functional and is discontinuous at class
  boundaries. No global linear reward parameter exists.
- **POSSIBLE, re-scoped reward:** inside one certified activation/argmax chart, the exact-teacher
  residual or directional control regret can be locally smooth and approximately linear in a frozen
  latent residual representation.
- **BREAK, mask process:** querying precisely when uncertain makes exact-coordinate observation
  endogenous/MNAR. A theorem for exogenous random coordinate masks cannot be reused.
- **BREAK, stationarity:** the `#455` student changes the state distribution and its own provider.
  The current sibling re-derivation calls it a nonconvex moving-distribution regression loop, not a
  stationary fixed-point oracle.
- **MEASURED existing falsifier:** the tested `#455` formulation passed its fit gate but drifted above
  deterministic repeat floors for CE and `d_pose`; it skipped only `2/3` non-anchor teachers on a
  three-update window, not recurring K20/95%. That formulation remains scoped NO-GO.
- **MEASURED existing certificate blocker:** `#454b` has zero rigorous whole-ball reuse and its
  faithful HVP correction costs more than the `8.375` validation-per-teacher baseline. It cannot be
  used as a trust authorizer here.

`verdict_scope:` direct TOFU control of raw `d_seg` and deterministic query-on-uncertainty are NO-GO.
The local residual allocation family remains open.

### 4.2 Frozen-epoch pull-when-uncertain schedule

At epoch boundary `e`, freeze:

- surrogate/teacher-residual representation `U_e` and its rank;
- surrogate weights/EMA shadow used for decisions;
- confidence matrix `V_e`, thresholds, and query-probability law;
- source checkpoint, exact model/runtime hashes, and trust-region authority inputs.

For candidate step `t`, form only from cheap pre-query information:

```text
u_t = beta_e ||z_hat_t||_(V_e^-1)
      + epsilon_subspace_t
      + epsilon_surrogate_t

boundary_t = distance to the certified local chart / exact-metric admission boundary
ood_t      = ||(I - U_e U_e^T) x_t|| or another registered reconstruction residual.
```

Set a *predictable* probability `p_t` from these quantities, with `0 < p_min <= p_t <= 1`, for
example a monotone clipped law. Then draw a recorded Bernoulli audit. Force `p_t=1` at stage
boundaries, failed custody/admission gates, OOD events, or `u_t >= boundary_t`. Exact rows may be
buffered during the epoch, but `U_e` and the served surrogate remain frozen until the atomic epoch
checkpoint. A deterministic “never query in the trusted region” rule (`p_t=0`) is forbidden because
it destroys both missingness coverage and the unbiased correction below.

### 4.3 Exact-forward call saving

For `T` candidate steps and exact-query indicators `A_t`:

```text
Q_T              = sum_t A_t
exact-call saving = 1 - Q_T/T
E[Q_T | cheap history] = sum_t p_t
expected saving  = 1 - mean_t(p_t).
```

For a two-part rule with independent audit floor `q` and an uncertainty trigger firing on fraction
`p_uncertain` of non-audit steps,

```text
E[Q_T]/T = q + (1-q) p_uncertain
saving   = (1-q)(1-p_uncertain).
```

Therefore a 95% exact-call skip requires

```text
q + (1-q) p_uncertain <= 0.05,
```

which is impossible if the audit floor alone exceeds 5%. No value for `q` or
`p_uncertain` is chosen here; their distribution is unmeasured. This is the requested **DERIVED
symbolic saving**, not a projected win.

Call saving is not wall-clock saving. If exact and surrogate complete-step costs are `c_E` and
`c_S`, the idealized time saving is

```text
(1 - Q_T/T) * (1 - c_S/c_E).
```

Using the already measured but fidelity-failed `#455` complete-window ratio `c_E/c_S=1.83988695`
only as a **non-admissible sensitivity calculation**, even 95% exact-call skipping would imply at
most `0.95*(1-1/1.83988695)=43.37%` complete-window time saving. It cannot be promoted because the
provider failed fidelity and the window excluded validation/controller-search calls.

## 5. Composition with `#462` / VR-GHAL

### 5.1 What the paper gives

The official VR-GHAL abstract assumes a nonexpansive or contractive fixed-point operator accessed by
unbiased stochastic evaluations with bounded second central moment. Suppressing logarithms and all
parameters except target residual `epsilon` and contraction/Lipschitz constant `gamma`, it reports

```text
bounded variance:             min{epsilon^-5, (1-gamma)^-3 epsilon^-2}
Lipschitz-in-expectation:     epsilon^-3 in the nonexpansive case
samplewise nonexpansiveness:  epsilon^-2.
```

This answers **how many valid stochastic-oracle evaluations** are sufficient. It does not by itself
authorize replacing them with a biased learned teacher.

### 5.2 The honest rate-times-allocation composition

Let `G_t=G(x_t;xi_t)` be an exact stochastic-oracle sample satisfying
`E[G_t | F_t,x_t]=T(x_t)`, where `T` is the mean fixed-point operator. Let `G_hat_e,t` be the
frozen cheap proxy computed from pre-query information `C_t`, choose `p_t` measurably from `C_t`,
and draw `A_t ~ Bernoulli(p_t)` with fresh randomness conditionally independent of `G_t`. At the
same VR-GHAL-requested iterate, use

```text
G_tilde_t
  = G_hat_e,t + (A_t/p_t) * (G_t - G_hat_e,t).
```

Then, for `p_t>0`, first condition on the cheap information and the conceptual exact sample:

```text
E_A[G_tilde_t | C_t,G_t] = G_t.

E[G_tilde_t | F_t,x_t] = T(x_t)             (iterated expectation).
```

This inverse-probability residual correction is the clean composition:

- VR-GHAL supplies the required stochastic-oracle *rate*.
- The frozen uncertainty model supplies the per-request exact-teacher *allocation probability*.
- Exact teacher calls equal `Q = sum A_t`, while every VR request still receives an unbiased oracle
  estimate.

The landed `#462` memo identifies an important simpler comparator. For a fixed replay state and
deterministic frozen-SegNet target, compute the exact label once and cache it. For a frozen linear
head, the teacher label cancels from paired parameter-gradient differences, so VR-GHAL does not
automatically save another exact label. The inverse-probability wrapper is relevant only when the
valid fixed stochastic oracle still samples fresh, not-yet-labeled states and the proxy residual is
cheaper than exact labeling; it must beat exact-label caching in complete call accounting.

But allocation is not free. Conditional on `C_t,G_t`, the audit-randomness variance is
`(1-p_t)/p_t * ||G_t-G_hat_e,t||^2`; stochastic-oracle variance remains in addition. Driving `p_t`
toward 0 can therefore increase the effective second moment and the hidden constant/call budget in
VR-GHAL. The true composition is implicit:

```text
B_VR = B_VR(epsilon, delta, gamma, sigma_eff(p))
Q_exact(epsilon) = sum_{t=1}^{B_VR} A_t
expected exact saving versus exact-every-request = 1 - mean_{t<=B_VR}(p_t).
```

Thus “VR-GHAL says how many; TOFU says where” is valid only with the same-iterate unbiased correction
and a proved variance bound. Arbitrarily relocating VR-GHAL oracle calls to other steps, or setting
`p_t=0` in trusted regions, changes the algorithm and loses the theorem.

### 5.3 Landed `#462` reconciliation and scoped disposition

The sibling memo `.omx/research/vrghal_95kill_fixedpoint_20260713.md` landed during this memo's final
verification pass and was read without modification. Its result sharpens, rather than reverses, the
fit audit:

- **DERIVED by `#462`:** the live `#455` map fits the verified VR-GHAL hypotheses: **NO**. A moving
  on-policy gradient map has operator drift; a population best-response makes one teacher forward
  an invalid oracle for the response; and a coupled witness/surrogate map lacks both a full oracle
  evaluation and nonexpansive geometry.
- **DERIVED by `#462`:** theorem-certified teacher-forward saving for live `#455` is `0` calls,
  hence `0%`. One call per 20 steps gives the algebraic target `19/20=95%`, but no VR-GHAL theorem
  currently certifies it.
- **DERIVED by `#462`:** the nearest theorem-shaped formulation is frozen exact-labeled replay with
  a convex linear/ridge head and a proved Hessian spectral interval. In that formulation, exact
  label caching and label cancellation in paired gradient differences already perform the direct
  teacher-call elimination.

Therefore:

- **NO-GO, current formulation:** no VR-GHAL call budget may be multiplied by a TOFU allocation
  ratio for current `#455`.
- **FEED-455 reactivation:** first define and prove the fixed-point operator, nonexpansiveness or
  contraction, unbiased exact stochastic oracle, bounded variance, and a frozen-surrogate residual
  correction with `p_min>0`. Then show that it saves exact calls beyond the simpler cached-label
  baseline. Only then is the equation above a rate-times-allocation bridge.
- **No redundancy:** VR-GHAL controls convergence rate of a valid oracle method; the allocation law
  controls which requests pay for exact residual correction. They are complementary only after the
  common oracle contract exists and only when fresh exact labels remain necessary. On frozen
  already-labeled replay they conflict economically: caching dominates the proposed allocator.

## 6. Scoped verdicts and reactivation gates

### Application 1 — `FEED-costate-controller`

`verdict_scope:` default-off design arm for the 72 registered owed lever rows, within a fixed
reference checkpoint/regime. It is not approved for the mixed 100-row lever-plus-curriculum pool,
live actuation, or pointer claims.

Reactivation/admission gates:

1. measured action descriptor dimension and stable low-rank effect geometry;
2. honest pre-pull random masks, or downgrade to ordinary latent OFUL;
3. fixed-regime common-checkpoint exact `n600` walk-forward win against all current rankers.

### Application 2 — `FEED-455`

`verdict_scope:` frozen-epoch local smooth teacher-residual allocator with randomized audit and
inverse-probability correction. Raw argmax reward, deterministic skip, current failed `#455`, and
uncertified `#454b` reuse are excluded.

Reactivation/admission gates:

1. local residual linearity/calibration and rank measured on exact `n600` receiver-realized rows;
2. `p_t` propensity custody, `p_min>0`, unbiasedness and variance receipt;
3. proof of the VR-GHAL fixed-point/oracle assumptions from `#462`;
4. exact-metric non-regression and complete wall-clock economics, not call-count projection alone.

## 7. Triality disposition

- **Equation leg:** two conditional laws are written in
  `.omx/research/tofupov_ranker_allocation_equation_feed_20260713.md`:
  `tofupov_costate_measurement_index_v1` and
  `vrghal_inverse_probability_exact_oracle_v1`.
- **DAG leg:** standalone intents are written in
  `.omx/research/tofupov_ranker_allocation_DAG_FEED_20260713.md`.
- **DSL leg:** deliberately deferred. The proposal names no trainer flag and makes no live controller
  edit. A future typed controller policy must compile the epoch freeze, masks/propensities, fallback,
  P8 governor, and checkpoint schema before any run.
- **Shared registration:** `DEFERRED_MAIN_REVIEW`. At derivation time,
  `.omx/state/canonical_equations_registry.jsonl`, the canonical `sub015_DAG`, and
  `src/tac/canonical_equations/__init__.py` are modified sibling-owned hot surfaces. Writing through
  them would violate the anti-collision instruction. The standalone files do not claim canonical
  registration.

## STORES CONSULTED

Full `CLAUDE.md`; full `AGENTS.md`; full `docs/operating_manual_craft_handoff.md`; top Claude memory;
latest Codex findings/session summary and latest council/design memo; official arXiv abstract records;
`.omx/state/lane_registry.json`; `.omx/state/subagent_progress.jsonl`;
`.omx/state/canonical_task_status.jsonl`; current activation/significance/curriculum stores through
their canonical readers; `activation_ledger.py`; `curriculum_candidate_pool.py`;
`producer_bridge.py`; `control_alphabet.py`; `shadow_controller.py`; `lambda_net.py` and its real-only
backtest contract; `#426` capabilities envelope; synthetic-costate real-only gate; current `#455`,
`#454b`, and `#456` memos/receipts. No live run, cloud/provider, GPU, protected trainer, archive,
evaluator, or pointer surface was actuated.

## Final pointer honesty

This memo derives a conditional controller advantage and a bias-corrected allocation law. It
measures no ranker win, no surrogate win, no `n600` endpoint, no archive score, and no 95%-kill.
Only a custody-complete measured comparison can move system status. **POINTER DELTA: UNMOVED.**
