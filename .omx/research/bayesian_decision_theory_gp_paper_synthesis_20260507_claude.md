# Bayesian Decision Theory For Gaussian Process Models — Synthesis

Date: 2026-05-07
Owner: claude (subagent)
Source paper: `Bayesian_Decision_Theory_For_Gaussian_Process_Models-32.pdf`
  — Ryan Warnick, Microsoft Security Research, Oct 12 2025 (15 pages, 3 theorems)
Evidence grade: research-only (no contest-CUDA)
Score claim: false
Dispatch attempted: false
GPU launched: false

## 1. Paper TL;DR

Warnick's paper is a Bayesian-decision-theoretic recipe for recovering an unknown
source/forcing term `Q = Ly` from noisy observations of `y`, where `L` is a known
linear differential operator (ODE order n, or PDE multi-index order m), under a
Gaussian-process prior on `y`. Three core results:

1. **Theorem 1 (Extension theorem, p.3):** Under Lipschitz + linear-growth bounds
   on the loss `Ψ(a, h)`, the Bayes action computed on a sequence of finite-dim
   projections `H_n ↑ H` converges (in mean-square under the GP measure) to the
   infinite-dim Bayes action. Practically: discretize and solve, you get the right
   answer asymptotically.
2. **Theorem 2 (L₂ vs 0–1 small-ball equivalence, p.4):** For any Gaussian
   posterior `Q | z ~ N(μ, C)` on a Hilbert space, the Bayes action under squared
   loss `‖q - a‖²` AND under small-ball 0–1 loss `1{‖q - a‖ > τ}` (any radius τ>0)
   is identically the **posterior mean μ**. Mean = MAP = small-ball mode for
   Gaussian posteriors.
3. **Theorem 3 (Decision boundary between two GP posteriors, p.4):** For
   `Y ~ N(μ, C)` vs `Y' ~ N(μ', C')` on the same Hilbert space, the small-ball
   risk-equality boundary coincides with the squared-loss boundary, an explicit
   affine hyperplane `‖a-μ‖² - ‖a-μ'‖² = tr(C') - tr(C)`.

Sec. 3–4 give closed-form linear-Gaussian conditioning machinery: build kernel
blocks `K_ZZ, K_*Z, K_**` by differentiating `K` once per derivative order;
`Q_*|z ~ N(L_* μ_*, L_* Σ_* L_*ᵀ)` where `L_*` is the row-stack of operator
coefficients at test points. Boundary conditions encode as additional rows of
`H` with tiny noise `R`. Sec. 5 extends to nonlinear operators via Fréchet
linearization with a high-probability bound:
`‖Q - Q̂‖ ≲ ‖DN[μ_*]‖_op · √λ_max(Σ_*) · √χ²_{d,α} + sup ‖R₂(e)‖` (eq. 8).

The "Bayesian decision theory" wrapper buys: (a) a principled posterior-mean
estimator for the operator output, (b) closed-form posterior covariance you can
plug into experimental design, and (c) the L₂↔0-1 equivalence which justifies
using the cheap mean even when the actual loss is "stay within a tolerance ball"
— the contest-faithful loss on PoseNet quantization basin parity.

## 2. Direct relevance to comma codec

### 2a. Pose pipeline (`src/tac/pose_gaussian_process.py`, `experiments/fit_pose_gp.py`) — **LOW**

Lane GP smooth-basis-fit was **KILLED 2026-04-30** (Council #271, see
`council_lane_gp_v4_design_20260430.md` and the kill-acknowledged docstring at
`pose_gaussian_process.py:7-19`). The Lane G v3 pose trajectory is approximately
**white noise in dims 1-5** with **uniformly distributed spectral support** —
structurally incompressible by any smooth basis at any K, RMSE floored ≈ 1.2
(near signal std). A GP regression with RBF/Matérn kernel is a smooth basis;
Theorem 2 doesn't change the trajectory's spectral content. Refusing to reopen
Lane GP is correct.

The paper's closed-form-conditioning machinery (Sec. 3) does not buy us anything
the killed lane couldn't have bought, because the limit is the *rate-side*
incompressibility of the residual, not estimator suboptimality. **This is a clean
PASS for pose-axis reactivation.** (Confidence: HIGH; conditional on the white-
noise diagnosis being right, which Lane GP v4 council audited at three rounds.)

### 2b. Meta-Lagrangian atom selection — **MEDIUM**

We already implement a Bayesian-EI/EIG acquisition rule: see
`.omx/research/bayesian_experimental_design_20260506_codex.md`,
`src/tac/optimization/bayesian_experimental_design.py`, and
`tools/rank_exact_eval_information_gain.py`. The acquisition is
`α = w_EI · EI + w_EIG · EIG` with closed-form Gaussian EI and entropy reduction.

Warnick adds two conceptual upgrades to that surface:

- **Operator-output uncertainty (Sec. 5.2 high-probability certificate, p.11):**
  the credible-ellipsoid radius `r_α = ‖DN[μ_*]‖_op · √λ_max(Σ_*) · √χ²_{d,α}`
  is a *spectral* (operator-norm × eigenvalue) bound on decision error, not just
  posterior variance. Our atom ledger today reports `score_uncertainty` as a
  scalar; the paper would replace it with a Lipschitz/operator-norm × spectral
  radius product. That is sharper and is the calibration that
  `predictor.score_band` is *trying* to do empirically via 3-anchor power-law
  fit. (Confidence: MEDIUM; we'd need to choose a meaningful `DN` for the
  rate→score mapping, which is not naturally an ODE/PDE operator.)
- **Risk-driven design vs variance-driven design (Sec. 1.2 contribution 2,
  p.2):** "optimize posterior expected loss rather than posterior variance."
  Our EIG term reduces variance; a Bayes-risk-driven acquisition would prefer
  candidates whose evaluation reduces *expected score-loss for the next
  dispatch*, which is closer to the operator's true utility. This matches the
  CLAUDE.md mandate "prefer solvable math over arbitrary sweeps."

### 2c. δεζ joint-training callback — **LOW**

The δεζ callback (`src/tac/codec_pipeline_deltaepszeta_callback.py:1-57`)
reports per-op byte counts as a co-training signal and optionally adds a soft
archive-size penalty. The paper's loss machinery is decision-theoretic over a
random function `Q`; δεζ's loss is over deterministic encoder bytes plus a
discrete pixel/distillation loss. There's no GP-posterior to plug Theorem 1 into,
and no ODE/PDE operator we'd be inverting at training time. **PASS.**

### 2d. Joint-ADMM marginal oracle (`src/tac/joint_admm_coordinator.py`) — **LOW-MEDIUM**

The Boyd ADMM coordinator (file:14-39) iterates with a cached score-cost surface
`f_s(x_s)` per stream — it does NOT load the scorer at coordinator time. The
KKT condition equilibrates `dScore_s/dByte_s = λ*` across streams. Warnick's
posterior-mean Bayes action could replace whatever point estimate currently
populates `f_s` with a Bayes-optimal mean under a GP surrogate. But:

- our score-cost surfaces are constructed from finite empirical anchors, not GP
  regressions over a continuous rate dimension;
- Theorem 2 says mean = MAP for Gaussian posterior, so unless we already use a
  non-Gaussian surrogate the swap is a no-op;
- the ADMM proximal step minimizes a quadratic — already squared-loss — so
  Theorem 2 confirms what we do is correct, not improves it.

**This is "validation, not extension."** The paper's main reachable contribution
to ADMM is documenting that our chosen objective is the small-ball-loss-optimal
one too, which is useful reviewer ammunition for the writeup but not a code
change.

## 3. Concrete code-level connection points

If we wire anything (verdict: STUDY; see §6), the highest-leverage edits are:

1. **`src/tac/optimization/bayesian_experimental_design.py`** — replace the
   scalar `posterior_variance` field in EIG with the operator-norm × spectral
   radius product from eq. 8 (paper p.11). This requires only computing
   `λ_max(Σ_*)` over the score-band predictor's posterior covariance over the
   `(rel_err, archive_bytes)` design space. Does NOT require any new GP fit;
   reuses existing 3-anchor calibration.
2. **`src/tac/predictor/score_band.py:42-50`** — annotate that the predictor's
   refusal modes are doing exactly what Warnick's Theorem 1 requires (Lipschitz +
   linear-growth bounds on the loss; refuse outside calibration range). This is
   a docstring-only change for council citation, not a behavior change.
3. **`src/tac/joint_admm_coordinator.py:14-39`** — add a one-line comment block
   citing Theorem 2: ADMM's squared-loss objective is provably equivalent to a
   small-ball 0-1 loss for any τ>0, which is closer to the contest's basin-parity
   gate (the apogee_int6 sanity ladder is exactly a small-ball test).
4. **`tools/rank_exact_eval_information_gain.py`** — add an `acquisition_mode`
   flag with `"variance_reduction"` (current) vs `"bayes_risk_reduction"`
   (Warnick Sec. 1.2 prescription). The risk-reduction mode picks the candidate
   that maximizes E[score improvement | new eval] under the predictor posterior.
5. *(Optional, if §6 flips to WIRE)* **`src/tac/predictor/score_band.py`** —
   swap the empirical power-law fit `score(rel_err) = a · rel_err^b + c` for a
   GP posterior over the same anchors, using a Matérn-3/2 kernel
   (paper Sec. 3.6). Buys closed-form posterior variance everywhere, not just
   at fit anchors. **This is the only edit that adds new math, not just a
   citation; it is also the only edit that changes runtime behavior.**

## 4. Unique value vs what we already have

- **vs Boyd ADMM:** Theorem 2 confirms our squared-loss choice is small-ball
  optimal. No behavior delta.
- **vs Hessian/Fisher sensitivity (`src/tac/sensitivity_map.py`):** Sensitivity
  maps are first-order derivatives at one point; the paper's Sec. 5.3 second-
  order curvature-control bound (`‖R₂(e)‖ ≤ ½ M ‖e‖²`, eq. 9) gives a *certified*
  bound on linearization error. Currently we use Fisher heuristically; the paper
  would let us tag each sensitivity-derived prediction with a Fréchet-curvature
  certificate. This is "research-grade rigor," not a score lever.
- **vs Pareto frontiers:** Pareto operates on observed rows; Warnick's machinery
  operates on a GP posterior over the *unobserved* design space. They are
  complementary — Pareto is the *retrospective* selector, the GP posterior is the
  *prospective* selector. We already have Bayesian EI/EIG (codex memo above);
  Warnick sharpens but doesn't replace it.
- **vs MacKay's MDL framing:** The paper sits exactly where MacKay's
  Information-Theory + Bayesian-Inference + Learning-Algorithms framework
  predicted: posterior mean = arithmetic-coding optimal point estimate under
  squared loss; Theorem 2's 0-1 ball loss is the analog of MDL's "code length
  within tolerance." This is a citable bridge, not a new lever.
- **vs Ballé hyperprior:** Ballé learns rate prediction `bits = -log₂ p_y(y)`
  end-to-end; Warnick's framework is closed-form-conditional, not end-to-end-
  trainable. They cannot be composed without giving up tractability.

## 5. Risks / why this might NOT help

1. **Operator structure absent.** Sec. 3-4 require a known linear differential
   operator `L`. The score function `S = 100·d_seg + sqrt(10·d_pose) + 25·B/N`
   is nonlinear in `(d_seg, d_pose, B)` but is NOT a derivative operator; it's
   a static linear combination. Sec. 5 (Fréchet) covers nonlinear operators but
   needs `‖DN[μ]‖_op` bounded — for SegNet/PoseNet this requires loading the
   scorer, violating strict-scorer-rule.
2. **Posterior-mean equivalence is decoy-attractive.** Theorem 2 says mean is
   optimal under squared AND 0-1 ball loss. We already use mean estimators
   (the predictor's point prediction is a mean). Confirming a current practice
   is not new value.
3. **GP regressions over our 3-5 anchor calibration sets.** GP needs O(n³)
   conditioning (Sec. 1.4 acknowledges this). We have ≤8 anchors per lane class.
   GP at n≤8 reduces to a pleasant-looking Gram matrix — the *advantage* over
   power-law fitting is mostly uncertainty quantification on the interpolant,
   not predictive accuracy. The current refusal modes (extrapolation, monotone-
   curve, ≥3 anchors) already capture the same epistemic guard.
4. **Code GitHub link is a placeholder.** Sec. 6 says "Code available at the
   github repo **INSERT LINK HERE**" (paper p.12) — this is a draft preprint;
   no reference implementation exists. We'd be reimplementing Sec. 3-5 from
   the formulas alone.
5. **Pose direct application is dead.** §2a above. The headline use case
   (replace pose with a GP posterior) is closed by Lane GP v4's white-noise
   diagnosis.

## 6. Verdict: **STUDY**

Medium relevance, no urgent wire. The mathematics is clean and the L₂↔0-1
equivalence is a beautiful citation for the writeup, but every operationally-
reachable application maps onto something we already have:

- Bayesian EI/EIG → already implemented (codex 2026-05-06)
- Posterior-mean Bayes action → equivalent to our existing point estimators
  under Gaussian posteriors
- Operator-norm × spectral certificates → useful rigor, but bounded by our
  ability to estimate `‖DN[μ]‖_op` without loading the scorer

The paper's biggest unique contribution to us is Sec. 5's *high-probability
decision-error certificate* (eq. 8), which would let us put rigorous
confidence bands on score-band predictions instead of empirical power-law +
refusal-mode heuristics. That is one paper-grade research week, not a
contest-window lever.

Queue-as-research-only background reading. Re-evaluate if (a) we ever
fit a continuous score-band surface over 8+ anchors per lane class, OR (b)
the writeup needs a formal Bayes-optimal-decision citation for the predictor.

## 7. If WIRE (deferred — not recommended now)

Not applicable per §6. If verdict flips:

- **File(s):** `src/tac/predictor/score_band.py` (replace power-law fit with
  GP-posterior fit; ≤80 LOC delta + new dependency on a small Cholesky helper)
- **Test delta:** add `test_score_band_gp_posterior.py` covering posterior-mean
  reduction to power-law fit at fit anchors, EIG monotone-shrinking with anchor
  count, and refusal-mode parity with current behavior.
- **Predicted score impact:** **`[predicted-band] 0.000 ± 0.005` directly** —
  this is a measurement-quality change, not a score-lever. Indirect impact via
  better dispatch ranking: `[predicted-band] 0.005-0.020` over a 4-dispatch
  budget at this operating point, conditional on the predictor's current
  `[heuristic:Q1-Hotz "above 1%, run local proxy"]` refusal threshold being a
  *binding* constraint (it currently is, for apogee_int<6 candidates).
- **Falsification:** if a 4-dispatch GP-acquisition wave at PR106's frontier
  picks the same 4 candidates as the current power-law-acquisition wave, the
  delta is null and we revert.

## 8. Cross-references

**Existing memory:**
- `bayesian_experimental_design_20260506_codex.md` — already-implemented EI/EIG
  acquisition, scope of overlap with this paper
- `council_lane_gp_v4_design_20260430.md` — kills smooth-basis pose fits
  (relevant to §2a)
- `council_lane_mdl_bayesian_design_20260430.md` — prior MDL+Bayesian council
  on the same pose lane
- `feedback_grand_council_predictor_calibration_no_arbitrariness_20260505.md`
  — predictor refusal-mode prescription (relates to Theorem 1 Lipschitz bounds)
- `apogee_int6_scorer_basin_parity_20260507_codex.md` — small-ball loss in the
  wild (the basin-parity gate is exactly Theorem 2's 0-1 loss with τ = score-
  delta threshold)

**Council members touched:**
- **MacKay (memorial seat):** the paper's Theorem 2 (0-1 small-ball ↔ L₂)
  is precisely the bridge between MDL code-length-within-tolerance and squared-
  error-loss decision theory. MacKay's *Information Theory, Inference, and
  Learning Algorithms* §3 (model comparison) and §28 (decision theory) are the
  canonical co-citations.
- **Ballé:** the paper does NOT subsume his end-to-end trainable hyperprior;
  Sec. 3 requires known operators, hyperprior learns them. Complementary, not
  competitive.
- **Boyd:** Theorem 2 retroactively justifies the squared-loss choice in
  `joint_admm_coordinator.py` for the basin-parity context. Citation-grade.
- **Tao:** the Hilbert-space extension theorem (Theorem 1) is the kind of
  pure-math clarity Tao would endorse — it formalizes "discretize, then solve;
  the limit gives the right answer" with mild regularity. No new tool, but a
  rigor citation.
- **Shannon (lead) + Dykstra (co-lead):** the framework gives Dykstra a clean
  Hilbert-manifold description of the decision boundary between two posteriors
  (Theorem 3) — useful if we ever need to formally distinguish two predictor
  models on the same anchor set.
- **Hotz / Carmack (engineering):** would dismiss the wire as ceremony-not-
  signal; the predictor already refuses outside its calibration range. No-op
  on the score axis.

## 9. Quick verifier — what would change my mind?

A WIRE flip is justified iff one of:

1. The score-band predictor is found to mis-rank dispatch candidates at the
   current operating point in a way the GP posterior would have caught (e.g.,
   apogee_int6 mis-prediction in retrospect).
2. We grow to 8+ anchors per lane class AND the power-law fit's residuals show
   structure (non-monotone, multi-modal) the GP would model better.
3. A reviewer of the writeup specifically asks for a Bayes-optimal-decision-
   theoretic justification of the predictor's mean estimator, beyond the
   existing power-law + refusal modes. Theorem 2 is a one-paragraph answer
   and no code change is needed.

Otherwise: STUDY, archived, no further action.
