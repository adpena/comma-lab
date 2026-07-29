# Papers-checked — arXiv 1703.09194 "Sticking the Landing: Simple, Lower-Variance Gradient Estimators for Variational Inference" (Roeder, Wu, Duvenaud, NeurIPS 2017)

UTC: 2026-07-29 · Harvested by: ddm_stl1 (Opus, $0 crosswalk arm) · NO launches, NO scorer jobs.
Evidence class: FROM-LITERATURE + retro-typing of on-disk `[macOS-CPU advisory]` receipts.
`research_only=true` · `score_claim=false`. Pointer 0.1910828242 [contest-CPU] UNMOVED — MEANS.

Full crosswalk (mechanism · P2b retro-typing · gc6 row-8 forward spec · N-A sweep):
**`.omx/research/ddm_stl1_sticking_the_landing_crosswalk_20260729.md`** (+ DAG_FEED sibling).

## What the paper shows
The reparameterized ELBO gradient splits into a PATH term (objective moves because the sample
moves with φ) and a SCORE term `∇_φ log q_φ(z)` (explicit φ-dependence of the variational
density). The score term has zero mean, so dropping it stays **unbiased**; and at the optimum
`q_φ=p(z|x)` the path-only estimator is **exactly 0 per-sample → zero variance** while the full
estimator keeps jittering. STL "sticks the landing": estimator variance → 0 as q → posterior.

## Recall diff (why this was worth a memo)
- The score-vs-pathwise DICHOTOMY and **ES = likelihood-ratio class** are ALREADY in corpus
  (`policy_gradient_variance_reduction_survey_20260712.md`, line 140/155; Mohamed et al. JMLR
  2020). NOT re-claimed.
- NEW axis STL adds: **variance-AT-OPTIMUM / proximity-to-optimum**, the operating-point our
  post-burn plateaus live in — the existing crossover was keyed on horizon/support, not |grad|.

## Crosswalk verdicts (3 rows + N-A)
| # | lesson | our surface | disposition |
|---|---|---|---|
| 1 | STL variance-at-optimum; DReG (1810.04152) fixes naive-STL-on-IWAE; Geffner-Domke (2007.14634) optimal coeff ∈ (0,1) far from optimum / misspecified | — | **ADOPT-AS-THEOREM** (eyes open: near-optimum only; DReG if K>1) |
| 2 | ES = score-function-class; variance doesn't vanish at plateau | P2b MC400 ES receipt: −0.0411 = 100% pose √-term (d_pose 78.196→77.965, 0.30% rel), d_seg +2.6e-6 WORSE, 2023 s | **LAW CANDIDATE** (score-class dominated by aimed pathwise at low-|grad|; INSTANCE; falsifier = ES beats aimed pathwise at matched evals) → E2 / P2c round-2 / canonical-eqs |
| 3 | STL path-only for parameterized entropy models | gc6 row-8 Ballé rate-in-loss; `−log₂ p((d+u·Δ)/Δ)` learned Δ | **PREMISE-REFINED**: uniform proxy self-cancels (`(d+uΔ)/Δ=d/Δ+u`) → NO stochastic score term; STL OFF derived-default (ON biases). Genuine score term only under Gaussian/scale-hyperprior entropy model. DSL stub `rate_stl_path_only` default OFF + DReG guard → gc6 row 8 / v10 SPEC |
| — | N-A | deterministic seg loss · terminal pose GN · costate organ (already pathwise) · gradient-free search (algo N/A, theorem applies) · combinatorial coding | **N/A**, banked so nobody re-derives |

## Verdict
`LESSONS_HARVESTED; ONE_LAW_CANDIDATE_QUEUED (falsifier-gated); ONE_DSL_STUB_SPEC'D
(derived default OFF); N-A_SWEPT; NO_ARM_SPAWNED; NO_LAUNCH.` Pointer UNMOVED — MEANS.
Consumer-routing note: P3 N1=NO (d_pose 38.06 photometric wall) routes pose-in-burn to the v10
re-burn SPEC, making the row-8 rate-in-loss window MORE likely to fire (native to the re-burn).

STORES CONSULTED: policy_gradient_variance_reduction_survey_20260712 · ddm_gc6_from_endpoint_
convocation_20260729 (row 8 / T4 Ballé seat / §5 E2 N4) · ddm_e2_pose_stream DAG_FEED ·
p2b_mc400_diagonal_receipt.json (SSD) · segnet_recursive_fractal_factorization_20260715 ·
default-off-orphan + verdict-scope-ladder MEMORY rows.
