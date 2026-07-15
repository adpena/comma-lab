# n=1 / extreme-low-data learning theory + OSS for the costate organ (#499)

**Date:** 2026-07-14 · **Lane:** `lane_n1_lowdata_learning_theory_organ_20260714` ·
**Verdict scope:** `INSTANCE(real #205 trajectory, 10 verdicts / 9 intervals / 7 walk-forward folds)
< FORMULATION < FAMILY < PARADIGM`.
**Pointer:** UNMOVED (`0.1910828242 [contest-CPU]` submittable; `0.18804` PR128 = borrowed bank).
This is **MEANS-only apparatus research** — no score / launch / adoption / promotion claim. All
organ numbers are `[macOS-CPU advisory / NumPy-fp32; NON-PROMOTABLE]`. Literature results are
`[MEASURED — in-literature model]` (evidence about *their* models/datasets, never about this organ).

Sister task **#434** owns synthetic-data generation (the starvation cure) — this memo is the
THEORY + OSS side and deliberately does NOT duplicate it. Sister codex arm
`codex_findings_n1_lowdata_learning_theory_oss_20260714_codex.md` is the long-form; this is the
ranked-lever synthesis **updated with the now-MEASURED U result** (which the codex arm had marked
`SPECULATIVE — backtest owed`).

---

## 0. The MEASURED board (the baselines every lever must beat)

Walk-forward = deployment-faithful (past-only training, 7 folds). Aggregate = class-weighted mean
marginal ΔS MAE; per-class = mean over the 5 SegNet classes. Source:
`warmstart_organ_n1_rl_backtest_20260714.json` (organ arms) + the codex findings (GP/dispatcher).

| estimator | WF MAE (agg) | WF per-class MAE | verdict |
|---|---:|---:|---|
| **Persistence** (carry-last) | **0.002792** | **0.010823** | the honest floor; UNBEATEN per-class |
| `Q_priormean_iso` (physics prior mean) | 0.003067 | — | beats ridge, loses to persistence |
| `P_priormean_aniso` (physics prior mean) | 0.003182 | — | Q beats P — anisotropy does NOT win here |
| `A_ridge_solve` (incumbent recommendation) | 0.003902 | 0.022777 | **LOSES to persistence**, not Pareto |
| **`U_hierarchical_physics_residual`** (top-1, NOW MEASURED) | **0.002496** | **0.055910** | **WINS aggregate (−10.6% vs persistence, −36% vs ridge); LOSES per-class 5.2×** |
| `T_gp_costate_posterior` (RBF-time GP) | 0.001852 | 0.040533 | best aggregate but 4/7 folds, p=0.50; no per-lever field |
| dispatcher (regime router) | 0.001596 | — | instance-scoped, provisional-until-accrual |
| trained MLP / GRU / DeepONet | 0.010–0.086 | — | OVERFIT — out |

**The crux this board exposes (DERIVED):** the aggregate-MAE race is *already won* by physics-prior
means (U, T) — but **no data-driven arm beats persistence's per-class 0.010823.** The aggregate win
is the prior mean doing the work; the per-class collapse (U 0.0559, T 0.0405) is exactly the
"nine intervals update a prior, they cannot identify a free per-lever field" ceiling. **The organ's
job is the per-lever λ field, so the per-class column — not the aggregate — is the real gate.**

---

## 1. What tiny-n theory actually licenses (formalizing the measured ceiling)

**(a) Shrinkage / James–Stein / empirical-Bayes.**
[MEASURED — in-lit] James–Stein strictly dominates the MLE for the Gaussian normal-means problem in
dim ≥ 3 under squared loss (Stein 1956; Efron–Morris empirical-Bayes form). Ridge is asymptotically
minimax only over ℓ2-balls under dense Gaussian design; risk also depends on the design spectrum
(Dicker 2012, arXiv:1203.4572). **Provable-dominance condition for us (DERIVED):** shrinkage toward
the physics-prior mean `m0` dominates unshrunk least-squares iff the true field lies within an
`O(σ√k / ‖β−m0‖)` ball of `m0` — i.e. iff the physics prior is *approximately right*. The measured
board is consistent with this: shrinking toward `Q_iso` (U aggregate 0.002496) helps, but the
per-lever residual `β−m0` is large enough that per-class error blows up. Shrinkage buys the mean,
not the field.

**(b) PAC-Bayes / MDL — the capacity ceiling, made quantitative.**
[MEASURED — in-lit] McAllester/Catoni PAC-Bayes pays `KL(Q‖P)/n` + confidence terms; optimized
bounds only beat test-set bounds when the posterior stays near a meaningful prior (Pérez-Ortiz 2021,
arXiv:2106.03542). MDL two-part codelength penalizes effective spectrum, not raw parameter count
(Dwivedi 2023, JMLR 24:21-1133). [DERIVED] At n=9, δ=0.05, a Gaussian-KL penalty
`√((KL+log(2√n/δ))/(2n))` ≈ **0.516 with KL=0** — larger than the entire signal (marginal ΔS ~1e-3).
So **any posterior that moves appreciably off the prior is uncertifiable at this n.** This is the
formal statement of "updates-prior-not-identifies-field": admissible learned capacity ≈
`KL_budget ≲ 2n·ε² ` which at n=9 and target ε in score units is a *handful of scalar DOF*, not a net.

**(c) Interpolation / double descent.** [MEASURED — in-lit] Benign overfitting needs specific
random-design spectrum conditions (Belkin 2019 arXiv:1812.11118; Hastie 2019 arXiv:1903.08560).
[DERIVED] With 7 WF decisions those conditions are unverifiable → the MLP/GRU/DeepONet losses
(0.010–0.086) are `INSTANCE×FORMULATION` evidence against the current trained arms, **not** a
FAMILY closure. Reformulation queue stays open (frozen features + tiny readout).

**(d) Meta / amortization — where it HONESTLY applies.** [MEASURED — in-lit] Meta-learning transfers
a prior over a *distribution of tasks*; biased-regularization matches transfer-risk lower bounds
when task covariance is known (Konobeev 2021, PMLR 139). [DERIVED] **One trajectory is not a
meta-set.** MAML/Reptile/hypernetworks are inapplicable *as learners* here. They apply only as
**frozen-prior producers**: task #211 amortized pre-seeding, #433 comma10k/openpilot features, or
#434 simulated trajectories → then a tiny conjugate/ridge readout is fit on the 9 real intervals.
`GRADUATION_MIN_RECORDS=3` independent real trajectories remains the gate before any meta-arm counts.

**(e) Physics-informed / operator learning.** [MEASURED — in-lit] PINNs embed a known operator as a
soft/hard constraint; DeepONet/neural-operators learn a solution map with the operator as inductive
bias (Lu 2021 DeepONet; Li 2020 FNO). [DERIVED — the load-bearing organ fact] The organ analytically
knows `∂S/∂x` (the level-set costate / score law, cached) but **not** the response `∂x/∂u` (how a
lever perturbs the state). So the honest physics-informed form is: **hard-code `∂S/∂x` in the mean,
learn only the small `∂x/∂u` residual** — which is exactly `U_hierarchical_physics_residual`. A full
DeepONet needs the `n≫10` response coverage we don't have → gated behind #434.

---

## 2. Ranked ORGAN levers (each: theorem · OSS · $0 probe · baseline it must beat)

Ranked by expected per-class-MAE reduction (the real gate), given the MEASURED board.

### Lever 1 — Per-lever-field disambiguation of U: block-precision + P/Q ensemble *(highest EV, U already wins aggregate)*
- **Why now:** U is MEASURED to win aggregate (0.002496) but LOSE per-class (0.0559). The aggregate
  win is banked; the **entire remaining value is closing the per-class gap.** The measured
  `data_identified_lever_count=6`, `sign_resolved_fraction_95=0.5` says the field is half-blind.
- **Theorem (DERIVED):** conjugate Gaussian posterior `β̂ = P_N⁻¹ Zᵀ Σ⁻¹(y−f_phys)`,
  `P_N = P + Zᵀ Σ⁻¹ Z`, guarantees a unique PD solve even for p>n; per-class error is minimized when
  the block precision `P = blockdiag(τ_g⁻²I)` partitions {shared-drift, class-response, fixed-regime}
  so the class-response block borrows strength across classes (partial pooling ⇒ lower per-class
  variance). Empirical-Bayes selection of the few `τ_g` by **prefix-only marginal likelihood**.
- **OSS:** NumPy/SciPy closed form (Cholesky `solve`); cross-check against
  `sklearn.linear_model.BayesianRidge` (BSD-3, transfers as an oracle for the τ grid); PyMC/NumPyro/
  BlackJAX as *offline posterior oracles only* (keep NumPy authority; MLX-portable via matmul+Cholesky).
- **$0 probe:** on the identical 7 WF folds, sweep a **preregistered** 3–5 point τ-block grid with
  per-class partial pooling, prefix-only evidence selection, ship BOTH fixed `Q_iso` and `P_aniso`
  prior modes as a disambiguator. **Must beat persistence per-class 0.010823** (not just aggregate).
  Report effective-df + posterior covariance. Adoption only on a per-class win, never mean-only.
- **Honest risk:** if per-class stays > 0.0108, U is an `INSTANCE×FORMULATION` negative on the field —
  the aggregate win is real but the organ still can't be trusted per-lever from 9 intervals → escalate
  to #434 data. **Allergic note:** this probe does NOT need n≫10; it re-slices the existing 9.

### Lever 2 — GP-with-physics-mean, per-lever coregionalized residual (the honest `U_hierarchical`, GP form)
- **Why:** T (RBF-time GP) has the best aggregate (0.001852) but `response()` is deliberately ZERO —
  it forecasts *total* forcing, not the per-lever field, and its per-class 0.0405 is bad. A GP whose
  **mean function IS the physics prior `∂S/∂x`** and whose kernel is additive
  `k_time + k_regime + k_class` over the ~8-dim manifold gives a nonzero, uncertainty-quantified
  per-lever residual at tiny n because the *kernel* (not data) supplies capacity.
- **Theorem (MEASURED — in-lit):** GP posterior mean = physics mean + `k(x,X)(K+σ²I)⁻¹(y−m_phys)`;
  finite and well-posed at any n; multi-output/coregionalization (ICM/LMC) shares the response across
  classes. Kernel-ridge equivalence gives the same minimax-over-RKHS-ball guarantee as ridge but with
  the correct smoothness prior for a curved 8-manifold.
- **OSS:** existing NumPy `T_gp_costate_posterior` (reformulate mean+kernel, don't duplicate RBF arm);
  derivative-observation kernels from **tinygp** (MIT, docs/tutorials/derivative — directly relevant
  since λ=∂S/∂x is a derivative observation); GPyTorch (MIT) / GPflow (Apache-2.0) as design oracles.
- **$0 probe:** implement additive-kernel + physics-mean + coregionalized multi-output GP; compare to
  **T and U on identical folds**. Must beat **U's per-class 0.0559 AND approach persistence 0.0108**.
  Past-only hyperparameters (no held-out target leakage).
- **Rank rationale:** below Lever 1 because it is a new implementation (Lever 1 re-parameterizes an
  already-winning arm); above everything else because it is the only form that gives a *nonzero
  per-lever field with calibrated uncertainty* at n=9.

### Lever 3 — PAC-Bayes / MDL capacity governor as an admission GATE (not a learner)
- **Why:** the measured overfit (MLP/GRU 0.010–0.086) and the per-class blow-ups happen because
  nothing *refuses* capacity the data can't pay for. A closed-form Gaussian `KL(Q‖P)` + effective-df
  + log-evidence, emitted per fold, turns the ceiling into an executable admission rule.
- **Theorem (MEASURED — in-lit):** any arm whose posterior movement `KL(Q‖P) > 2n·ε²` is
  uncertifiable at (n=9, target ε) — the 0.516 penalty above; MDL codelength gives the twin pressure.
- **OSS:** none needed — lift the closed-form Gaussian-KL/MDL equations into `backtest()` JSON as
  diagnostics; `sklearn` BayesianRidge exposes log-evidence for cross-check.
- **$0 probe:** add `KL(Q‖P)`, effective-df, log-evidence columns to the existing backtest for U/T/A;
  verify the ranking of these certificates matches the WF-MAE ranking. **Beats nothing directly** —
  it is a governor that prevents future capacity-reflex arms from being adopted on an aggregate-only
  win. Value = extincting the overfit-reflex, not lowering MAE.

### Lower-ranked (recorded, not next):
- **L4 Plain ridge / RidgeCV / James–Stein** — keep `A_ridge_solve` as the incumbent comparator ONLY;
  it is measured to lose to persistence. Not a next arm.
- **L5 Prototype/regime partial pooling** — good for transients/routing (dispatcher 0.001596) but rare
  regimes have ≤1 example; policy transfer unmeasured. Keep small + interpretable + abstaining.
- **L6 Dependence-aware conformal wrapper** (MAPIE Apache-2.0 / crepes BSD) — too few serial blocks for
  a useful 90/95% certificate; coarse abstention telemetry only, never a learner.
- **L7 Meta/transfer prior** (learn2learn MIT; `higher` archived) — inapplicable until ≥3 independent
  trajectories or #211/#434 provide a real meta-set; import frozen features only, then tiny readout.
- **L8 #434 simulator distillation** — potentially high, RESEARCH-ONLY; belongs to sister task #434;
  admission only through the unchanged real-only WF gate.
- **L9 GRU / DeepONet / Gated-DeltaNet trained on target** — LOWEST in current form; 7B Olmo-Hybrid
  (arXiv:2604.03444) evidence is large-corpus expressivity, NON-transferable to n=9. Family stays open
  ONLY as a FROZEN prior-trained state-tracker + tiny Bayesian/ridge readout.

---

## 3. Direct answers to the task's sub-questions

- **When does physics-prior-centered shrinkage provably dominate?** (DERIVED) iff the true field is
  within `O(σ√k/‖β−m0‖)` of the prior mean — i.e. the prior is approximately right. Measured:
  Q_iso-centered U dominates on aggregate ⇒ prior is right *in the mean*; per-class blow-up ⇒ prior is
  *not* right per-lever ⇒ shrinkage buys the mean, not the field. **Is anisotropy the win? MEASURED NO
  — Q_iso beats P_aniso.** Ship both as a disambiguator; do not assume anisotropy.
- **Is GP-with-physics-mean the honest `U_hierarchical_physics_residual`?** (DERIVED) The conjugate-
  ridge form (Lever 1) and the GP form (Lever 2) are the SAME object in two bases (kernel-ridge duality).
  The GP form is the honest one *for the per-lever field with uncertainty* on the curved 8-manifold;
  the ridge/block form is the honest one *for cheap deterministic MLX-portable adoption*. Build Lever 1
  first (re-uses a winning arm), Lever 2 second (adds the calibrated field).
- **PAC-Bayes/MDL admissible capacity at n≈1..10?** (DERIVED) `KL_budget ≲ 2n·ε²` — a handful of scalar
  DOF, not a net. Formalizes the measured ceiling.
- **Meta-learning at n=1?** Only as a frozen-prior producer (#211/#433/#434), never as a learner on
  the 9 intervals.
- **Allergic-to-n≫10 flag:** L7/L8/L9 ALL need n≫10 (or a simulator) — say so plainly; they are gated,
  not next. Levers 1–3 all run on the existing 9 intervals at $0.

---

## 4. Triality / stores / pointer

- **DSL:** no mutation. `U_hierarchical_physics_residual` is a research candidate; register in
  `ARCHITECTURES` only with an implemented, deterministic-tested producer.
- **DAG:** standalone (this memo + `n1_lowdata_learning_theory_oss_DAG_FEED_20260714.md`).
- **Equations:** the conjugate residual posterior (§Lever 1) is a DERIVED equation *candidate* — NOT
  registered until a producer/consumer exists (no orphan law).
- **STORES CONSULTED:** CLAUDE.md; the n1-organ-ceiling memory; the codex n1 findings + warmstart
  backtest JSON (measured U/A/persistence); GP/router/Olmo DAG feeds; #211/#433/#434 references;
  primary papers linked above.
- **Pointer / launch / adoption delta:** NONE. MEANS-only.
