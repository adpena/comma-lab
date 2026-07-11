# Closed-form GP costate posterior (Warnick 2025) — derivation + walk-forward backtest (2026-07-11)

**Author:** decision-theory / GP subagent. **Status:** MEANS. **Pointer 0.19108282 [contest-CPU]
UNMOVED — no score claim; every number `[macOS advisory] NON-PROMOTABLE, score_claim=false`.**
Paper: R. Warnick, *Bayesian Decision Theory for Gaussian Process Models* (Oct 12 2025, 15pp, read
in full incl. Appendix A/B/C proofs). This is a SOLVE-DON'T-TRAIN candidate for the #426 costate
organ's n=1 fragility: the seal killed the *trained* λ-surrogates (MLP/GRU/DeepONet lost
walk-forward); a closed-form *Bayesian solve* with calibrated covariance is a different object. **The
backtest is the arbiter; it was run; the verdict is honest and PARTIAL (below).**

---

## 1. The mapping — organ costate ⟶ Warnick's source-term recovery `Ly = Q`

The paper recovers a source term `Q` in a linear operator equation `Ly = Q` under a GP prior `y ~
GP(0,K)` and noisy linear observations `z = H[y] + ε`, `ε ~ N(0,R)`. Under squared-error loss the
Bayes action is the posterior mean of `Ly`, with **closed-form mean AND covariance** via
linear-Gaussian conditioning of *differentiated* kernels (§3.3–3.4):

```
y*|z ~ N(μ*, Σ*),   μ* = K*Z (K_ZZ + R)^{-1} z,   Σ* = K** − K*Z (K_ZZ+R)^{-1} K_Z*
Q*|z ~ N(L* μ*, L* Σ* L*ᵀ)          # Q̂ = L*μ*  (Bayes estimator);  Cov(Q) = L*Σ*L*ᵀ
```

**Our instantiation (the organ's forecast leg).** The #426 backtest ranks arms by how well they
forecast the per-class state RATE `dx/dep` at the next verdict interval. Map:

| Warnick object | organ instantiation |
|---|---|
| latent `y(u)` | per-channel campaign state `x_c(ep)` (5 per-class d_seg + log-bytes) as a function of epoch |
| operator `L` | `L = d/dep` — a **1st-order ODE** (`a_1=1, a_0=0`), so `Ly = y'` |
| source `Q = Ly` | `Q(ep) = dx_c/dep` — the exact quantity the harness forecasts and the costate chain-rule multiplies by `∂S/∂x` |
| observation op `H` | point-evaluation at verdict epochs (rows of `H` = δ-functionals) |
| data `z` | the verdict-row states `x_c(ep_t)` |
| noise `R` | `σ_n²·I` (verdict measurement/aliasing noise, fit per channel) |
| kernel `K` | RBF `σ_f² exp(−(u−u')²/2ℓ²)`, differentiable to all orders |

The closed-form derivative posterior at a query epoch `u*` (the paper's §3.3 with the RBF derivative
blocks):

```
K   = σ_f² K_rbf(EP,EP) + σ_n² I                                   (m×m)
k_d = ∂_{u*} k(u*, ep_j) = σ_f²·(−(u*−ep_j)/ℓ²)·exp(−(u*−ep_j)²/2ℓ²)   (m,)
μ'(u*)  = k_dᵀ K^{-1} (z − z̄)          # Q̂ = posterior mean of x'(u*)  — the FORECAST
σ'²(u*) = σ_f²/ℓ² − k_dᵀ K^{-1} k_d     # closed-form posterior variance (the UQ / BOED signal)
```

Hyperparameters `(ℓ, σ_n)` are fit by **type-II marginal likelihood on PAST-ONLY observations at each
walk-forward fold** (bounded deterministic grid; `σ_f² = training channel variance`). Data are centered
per channel (`z − z̄`); the derivative is invariant to the additive constant, so centering is exact.

Code: `tac.witness_control.lambda_net.GPCostatePosteriorAdjoint` (arm `T_gp_costate_posterior`), numpy-only,
$0, deterministic.

### 1.1 Exact vs approximate — the honest accounting

- **EXACT (verified).** The forecast leg IS a textbook instance of Warnick §3.3: `L=d/dep`, point-value
  `H`, differentiable RBF ⟹ closed-form mean+cov. The differentiated-kernel blocks are **numerically
  verified correct**: closed-form `μ'` vs finite-difference of the (independently computed) GP function
  posterior mean agrees to **rel 2.85e-9** (test `test_derivative_kernel_matches_finite_difference`).
  `σ'²(u*) ≥ 0` by construction. Bit-deterministic across fits.
- **APPROXIMATE / where the map is a SURROGATE (stated, not hidden).**
  1. The organ's TRUE adjoint is over the **discrete-argmax nonlinear** SegNet `d_seg`. We GP-smooth the
     *already-scalarized* per-class `d_seg` summary trajectory — a **linear-Gaussian surrogate of a
     nonlinear/discrete object**, not a solve of the true costate. The paper's §5 Fréchet extension
     (first-order linearization around `μ*`, high-probability decision-error bounds) is the correct
     bridge for the nonlinear operator, and Theorem 3 is the bridge for the discrete decision boundary —
     but **we did NOT build the Fréchet-linearized nonlinear posterior**; we forecast the scalar
     trajectory. Valid as a forecaster; NOT a solve of the true nonlinear λ.
  2. The GP recovers only the **total forcing** `Q = dx/dep`; it does **not** decompose `Q` into
     per-lever responses `Λ_i = ∂x/∂u_i` (that is the ridge/prototype arms' job). So `response()` is the
     honest **zero field** and `base()` carries the whole forecast. Consequence: the harness
     binding-AUROC gate (a lever-field consistency floor) is **N/A** for this arm → the harness
     `passed=False` flag is the WRONG lens; the discriminating test is the walk-forward forecast MAE.
  3. Constant prior mean ⟹ the derivative reverts to 0 far from data (conservative shrinkage). This is
     exactly what helps in the transient and could hurt under persistent drift — a bias, measured below.

---

## 2. Theorem 2 (loss equivalence) applied to the seal's loss debate

**Theorem 2 (verified in Appendix B).** For a Gaussian posterior `Q|z ~ N(μ,C)` on a Hilbert space, the
Bayes action under squared-error loss `‖q−a‖²` AND under the 0-1 small-ball loss `1{‖q−a‖>τ}` is the
SAME — both equal the posterior mean `μ`, for any `τ>0`. The loss choice is FREE on a Gaussian posterior.

**Application to the seal.** The #426 seal noted the zero-model-error metric-relaxation arm (a relaxed
0-1) beat the learned surrogate, and separately that the trained λ-nets lost walk-forward. Theorem 2
reframes this precisely:

- **DERIVED:** if the organ's λ posterior is Gaussian, the squared-error-vs-relaxed-0-1 distinction is
  *not a loss-function effect* — both give the same `μ`. So the seal's arm differences came from the
  **POSTERIOR** (what `Q|z` is — the model/kernel), **not the loss**. Corollary: *don't tune the loss,
  tune the posterior.* The GP arm makes this operational — its forecast IS the posterior mean, loss-invariant
  by construction (`test`/structural).
- **MEASURED:** the GP posterior mean (WF 0.001852) **dominates the trained surrogates** (envelope
  measured MLP/GRU/DeepONet WF 0.010–0.086) by >5×, consistent with Theorem 2's "posterior mean is the
  Bayes action" being the right object and the trained nets being sample-starved approximations to it at n=1.
- **HONEST CAVEAT on applicability.** Does the Gaussian-posterior assumption hold for the organ's λ? For
  the **GP arm's** posterior — YES, exactly, by construction (Theorem 2 applies). For the **TRUE** λ (the
  nonlinear discrete-argmax adjoint) — NO; that `Q|z` is non-Gaussian, so "loss is free" is exact for the
  surrogate and heuristic for the real object. The claim is scoped to the surrogate.

---

## 3. Theorem 3 (decision boundary) — the separatrix bridge (equations-leg candidate, not over-built)

**Theorem 3 (verified in Appendix C).** For two GP posteriors `Y~N(μ,C)`, `Y'~N(μ',C')`, the small-ball
decision boundary equals the squared-loss boundary `{a : ‖a−μ‖² − ‖a−μ'‖² = tr(C')−tr(C)}`; when
`C=C'` it is the **codim-1 equidistant bisector hyperplane** `{a : ⟨a, μ'−μ⟩ = ½(‖μ'‖²−‖μ‖²)}`, a
Hilbert manifold locally diffeomorphic to a GP.

**Bridge (recorded, deliberately not built out).** This IS the structure of our d_seg **separatrix** and
the per-class-λ decision boundary: two candidate regime posteriors (e.g. lane-erosion vs plateau, or two
per-class-λ hypotheses) with posterior means `μ, μ'` have a decision boundary that is exactly this affine
codim-1 bisector in the state/λ Hilbert space — the same codim-1 boundary annulus where `d_seg` physically
lives (#333: ~97% of d_seg in a ~5% boundary annulus). It also **formalizes the prototype router**: the
router assigns `x` to the nearest regime prototype; Theorem 3 says (equal covariances) that assignment
boundary IS the bisector hyperplane — i.e. the interpretable router is a Gaussian-posterior Bayes decision
rule. This is an equations-leg *candidate*; it is NOT registered (see §7).

---

## 4. THE BACKTEST — walk-forward vs persistence (the acid test)

Harness: `tools/lambda_net_backtest.py --run-dir levelset_v752_baseline_20260710T185913Z` (the sealed
#205 trajectory: **10 verdicts / 9 intervals**, plateau-dominated). Gate = the seal's:
**LOO ∧ WALK-FORWARD (deployment-faithful, past-only) vs PERSISTENCE**, class-weighted `d_seg` (the
score-relevant aggregate `S=100·d_seg`). No look-ahead, no synthetic-fit.

| arm | LOO d_seg MAE | **WALK-FORWARD** d_seg MAE | per-class MAE | vs persistence WF |
|---|---|---|---|---|
| **T_gp_costate_posterior** | **0.001638** | **0.001852** | 0.040533 | **BEATS (−34%)** |
| E_prototype_bregman | 0.002947 | 0.002839 | 0.011716 | loses (+1.7%) |
| F_bsf | 0.002974 | 0.002844 | 0.011845 | loses |
| E_prototype | 0.003082 | 0.002967 | 0.012323 | loses |
| A_ridge_solve / G_scorerprior | 0.003296 | 0.003902 | 0.023 | loses |
| MLP/GRU/DeepONet (envelope) | — | 0.010–0.086 | — | loses badly |
| **persistence heuristic** | 0.003698 | **0.002792** | 0.0117 | — (incumbent) |

**MEASURED headline:** the closed-form GP costate posterior is the **BEST arm on both LOO and
walk-forward mean class-weighted `d_seg`**, and the **FIRST arm to beat persistence walk-forward on this
plateau-dominated trajectory** (0.001852 < 0.002792, −34%), where **every** incumbent arm LOSES to
persistence. LOO 0.001638 vs best incumbent 0.002947 (−44%).

**But the honest refinement (adversarially sought, per-fold):**

```
ep50->75   gp -3.9e-5  persist -3.5e-4  MEAS -9.6e-5 | gpErr .00143 vs .00637  GP  (transient)
ep75->100  gp +6.1e-5  persist -9.6e-5  MEAS +2.6e-5 | gpErr .00088 vs .00306  GP  (transient)
ep100->125 gp +7.5e-5  persist +2.6e-5  MEAS -9.6e-5 | gpErr .00429 vs .00306  PERSIST
ep125->150 gp +4.1e-5  persist -9.6e-5  MEAS -2.4e-6 | gpErr .00109 vs .00235  GP
ep150->175 gp +7.9e-5  persist -2.4e-6  MEAS +2.6e-5 | gpErr .00132 vs .00072  PERSIST
ep175->200 gp +6.8e-5  persist +2.6e-5  MEAS +9.9e-5 | gpErr .00078 vs .00182  GP
ep200->225 gp +5.9e-5  persist +9.9e-5  MEAS +1.9e-4 | gpErr .00316 vs .00217  PERSIST
mean: GP 0.001852  persist 0.002792   GP wins 4/7 folds   sign-test p=0.50
```

- The mean win is **concentrated in the early TRANSIENT folds** (ep50–100), where persistence
  over-chases a large transient slope and the GP's **calibrated shrinkage** wins 2–4×. In the plateau
  tail it is a **coin-flip (4/7, sign-test p=0.50 — NOT significant)**. The GP predictions are
  non-degenerate (real smooth slopes −3.9e-5..+7.9e-5, not trivially zero).
- This **CONFIRMS and sharpens the envelope's thesis** (§3: "the organ's edge is TRANSIENT-regime
  forecasting; the plateau favors persistence") rather than overturning it — the GP is simply the *best
  transient forecaster measured*, and the meta-λ should still route to persistence in the pure plateau.
- **Per-class MAE is WORSE** (0.0405 vs 0.0117): the GP wins the score-relevant *aggregate* `d_seg` but
  not the per-channel decomposition — it is a total-forcing/base-drift forecaster, not a per-class arm.

**VERDICT on "does the Bayesian solve crack n=1?" — PARTIAL, NOT DECISIVE (provisional-until-accrual).**
It cracks the **mean walk-forward on the score-relevant aggregate** (first arm to beat persistence in the
plateau trajectory, and it dominates every trained surrogate — a clean Theorem-2 confirmation). It does
**not** crack **per-fold significance** (p=0.50), the **plateau tail**, or the **per-class decomposition**,
and it produces **no lever field** (binding gate N/A). It is a genuine, deterministic, $0, closed-form-UQ
improvement to the **FORECAST/base-drift leg**, complementary to the decomposition arms — best deployed as
the base-drift forecaster with the meta-λ still selecting persistence in the pure plateau. It is NOT an
n=1 miracle; it is the honest best-forecaster-so-far with a calibrated covariance the trained nets lack.

### 4.1 The closed-form covariance (the paper's `Cov(Q)`, demonstrated)

Derivative posterior `x'(ep225) [mean ±2σ]` per channel (the UQ the trained arms cannot give):
`Road +7.4e-6 ±2.5e-3 · Lane +1.4e-3 ±2.5e-2 · Undriv −8.1e-6 ±4.2e-4 · Movable +1.7e-3 ±4.6e-2 ·
MyCar +2.0e-6 ±5.8e-5 · logBytes −6.1e-4 ±1.3e-3`. The bands are wide (esp. Lane/Movable) — **honest UQ
at n=9**, and directly the BOED signal below.

---

## 5. BOED / #434 Transient-Forge composition

Warnick §"Bayesian Risk-Based Experimental Design" targets the **decision functional** (the costate `Q`),
not just posterior variance: choose the next measurement to minimize `tr(L* Σ* L*ᵀ)` = the costate
posterior covariance. The GP arm **produces this covariance in closed form** (§4.1). Concrete composition
with #434: the Transient-Forge / duty-to-measure queue can rank the next verdict-epoch or probe by
`∂ tr(Cov Q_binding)/∂(new obs)`, weighting the **score-binding channels** (Road/Undriv/MyCar carry the
class weight; Lane/Movable carry the largest variance). This is a cheap, closed-form acquisition rule the
organ can run today, and it aligns with #434's regret-based teacher: the epochs where the GP derivative
posterior is most uncertain ARE the transient windows where forecasting beats persistence (§4). The GP
covariance is thus the **analytic acquisition signal** the Forge's simulator was going to approximate.

---

## 6. Round-1 adversarial review (attacking the hardest links)

- **(a) Is the GP-source-term map valid for the discrete-argmax nonlinear adjoint?** PARTIAL, stated in
  §1.1. The forecast leg (`Q=dx/dep` from the scalarized per-class `d_seg`) is an EXACT §3.3 instance
  (verified rel 2.85e-9). But the TRUE adjoint is over the nonlinear discrete SegNet argmax; the GP
  smooths the collapsed `d_seg` summary — a linear-Gaussian **surrogate**, not a solve of the true
  nonlinear costate. §5-Fréchet / Theorem-3 are the correct bridges and are **NOT built here**. Not
  overclaimed.
- **(b) Does Theorem 2's Gaussian assumption hold?** For the GP arm's posterior — YES exactly (Theorem 2
  applies; loss is free). For the real λ — NO (non-Gaussian). The "loss is free" claim is scoped to the
  surrogate and is labeled DERIVED, not asserted for the true object.
- **(c) Is the backtest walk-forward with zero leakage?** VERIFIED. `fit(intervals[:hold])` sees only
  verdicts with epoch ≤ ep_hold; hyperparameters fit by MLL on past-only; the query epoch = ep_hold
  (last training obs), forecasting the slope FORWARD; `x1` (the target) is never used. Persistence
  heuristic is also past-only ⟹ apples-to-apples. Determinism bit-checked. The one residual subtlety
  (query epoch is a training obs) is a boundary extrapolation, not leakage — the forward slope is not
  informed by the unobserved x1.
- **(d) Self-attack on my own headline.** The mean win is early-fold-concentrated, p=0.50, per-class
  worse, no lever field. So the correct headline is "best mean-WF forecaster / dominates trained
  surrogates / first to beat persistence on the aggregate in the plateau trajectory — but NOT
  significant per-fold, forecast-only." I flag my result **PROVISIONAL-until-accrual (verdict_scope:
  instance)**, exactly like every other n=1 organ verdict.

---

## 7. Triality legs + STORES CONSULTED

- **DAG:** `FEED-426-gp-costate` appended to `.omx/research/sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md`.
- **DSL:** arm `T_gp_costate_posterior` registered in `lambda_net.ARCHITECTURES` + `make_model` (the
  adjoint-tournament SoT the `costate_agent_dsl` arbitrates over). NO new witness-DSL `Lever` — this is
  an organ/adjoint arm, not a witness training knob (correct leg).
- **equations:** **N/A-with-reason.** Consistent with the seal's own standing stance ("Equations leg =
  NO new law: n=1 trajectory, below the ≥5-run anchor bar"), and reinforced by the measured p=0.50
  (not significant per-fold) + no-lever-field. The arm + the derivative-kernel correctness test are
  **APPARATUS**, not a registered law. Reactivation: register a `gp_costate_posterior_v1` law only when
  (i) ≥5 trajectory records show the GP posterior mean beats persistence out-of-noise AND (ii) the
  per-fold sign test clears significance.
- **Tests:** `src/tac/witness_control/tests/test_gp_costate_posterior.py` (5 tests: registration, shapes,
  **derivative-kernel correctness**, determinism, harness plug-in). All pass; ruff-F clean.
- **STORES CONSULTED:** Warnick PDF (all 15pp + Appendix A/B/C) · `costate_organ_capabilities_limits_envelope_20260711.md`
  (the seal) · `costate_lambda_marginal_ds_20260705.py` (the exact analytic λ) · `lambda_net.py`
  arm-API + backtest gate · `costate_posterior.py` (the *cross-run* inverse-variance posterior — a
  DIFFERENT object; noted to avoid conflation) · #342 solve-don't-train framing · organ trajectory
  ledger. Live pids: none disturbed (no training running; the arm is $0 numpy).

**Pointer 0.19108282 [contest-CPU] UNMOVED — all MEANS.**
