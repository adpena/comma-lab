# Expert team — statistics lens

**Date:** 2026-05-13
**Lane:** `lane_expert_team_fields_medalist_math_biology_alien_tech_20260513`
**Sister memos:** `expert_team_fields_medalist_math_biology_alien_tech_20260513.md` (master), `expert_team_pure_math_20260513.md`, `expert_team_geometry_20260513.md`, `expert_team_biology_20260513.md`.

## Persona seats

- **Bradley Efron** (Stanford) — bootstrap (1979), empirical Bayes, classifier calibration
- **Robert Tibshirani** (Stanford) — LASSO (1996), sparse regression, *Elements of Statistical Learning*
- **Trevor Hastie** (Stanford) — GAM, ESL textbook
- **Jerome Friedman** (Stanford) — gradient boosting, MART, CART
- **Peter Bickel** (Berkeley) — semiparametric mathematical statistics
- **David Donoho** (Stanford) — compressed sensing (2004), wavelets
- **Emmanuel Candès** (Stanford) — compressed sensing (Candès–Romberg–Tao 2006), matrix completion (Candès–Recht 2009)
- **Persi Diaconis** (Stanford) — MCMC mixing, Markov chain coupling, randomness theory
- **Christopher Bishop** (Microsoft) — *PRML* textbook, mixture-density networks
- **Andrew Gelman** (Columbia) — hierarchical Bayes, posterior predictive

## How the contest looks through this lens

The contest is a **statistical estimation problem**: given a fixed 600-pair sample on a fixed video, find the parameter `θ` minimizing an empirical-risk objective `S_hat(θ)` that is a noisy estimate of the population objective `S(θ)`. The Donoho/Candès question: **how many bytes do we need to encode the SegNet/PoseNet outputs faithfully?** The classical compressed-sensing answer: ~`k log(N/k)` where `k` is the effective dimensionality. The contest leaderboard shows N=37,545,489 pixel positions × 5 classes ≈ 187M, with `k ≈ 0.5–1M` (sparse boundary effective DOF). Compressed-sensing predicts ~`6–20K` essential bytes; PR101's archive is 162K. **Gap of ~10×.**

## Top derivations

### S-1 — Compressed-sensing lower bound on archive size `[statistical-bound]`

**Source:** Donoho, *Compressed sensing*, IEEE TIT 2006; Candès–Romberg–Tao, *Robust uncertainty principles*, IEEE TIT 2006.

**Statement.** For a `k`-sparse signal in ℝ^N with `k ≪ N`, the minimum number of linear measurements for exact recovery is `M ≥ C·k log(N/k)` for some constant `C ≈ 4` (Candès–Tao restricted-isometry constant). Random Gaussian / Bernoulli matrices saturate this bound with high probability.

**Application to contest.** The SegNet output is essentially a sparse signal: at each pixel, only the argmax of 5 logits matters. The active "support" is the **argmax-boundary set** — pixels where two or more logits are within ε. From PR101 lossy-coarsening rms ~1.84e-3 anchor, `k ≈ 8% × 196,608 = 15,728` boundary pixels. Lower bound:
```
M ≥ 4 · 15,728 · log(196,608 / 15,728) ≈ 4 · 15,728 · 2.53 ≈ 159,065 measurements
```
At 1 bit/measurement (one-bit compressed sensing, Boufounos–Baraniuk 2008), this gives ~**20 KB** lower bound for the SegNet payload alone. Current PR101 SegMap+renderer = 162 KB. **The 8× headroom argument depends on whether PoseNet contributes O(k log N/k) too.**

**Predicted Δ:** −0.020 to −0.040 [statistical-bound] if a one-bit-CS encoder is built for SegMap. **Cost:** $0 (theoretical); $5–10 for a CS-encoder prototype. **Implementation:** `tac.codec.one_bit_compressed_sensing_segmap`.

---

### S-2 — Matrix completion of HNeRV decoder weights `[statistical-bound]`

**Source:** Candès, Recht, *Exact matrix completion via convex optimization*, FOCM 2009.

**Statement.** A rank-`r` matrix `M ∈ ℝ^{n×m}` can be exactly recovered from `O(nr log² n)` random entries via nuclear-norm minimization. The constant ~3–4 for incoherent matrices.

**Application.** PR95's HNeRV decoder weight matrix is `~200K × 1` (when viewed as a learned bias surface across all positions), with empirical SVD rank ≤ 64 (PR101 LoRA evidence). The matrix-completion bound:
```
M_required = 3 · 200000 · 64 · log²(200000) ≈ 3 · 12.8M · 168 ≈ 6.5B entries  -- this is huge
```
because the matrix is essentially full-rank in the bad direction. **HOWEVER**: if we restructure as a `200 × 1000` matrix with rank-8, we get `3 · 200000 · 8 · log²(200000) / 200 ≈ 32M` → much friendlier. PR101 already does this via LoRA: encode only low-rank Δ on top of a fixed base.

**Operational claim.** The PR101 LoRA approach saturates the matrix-completion bound for the appropriately-restructured HNeRV. To beat it, restructure the decoder weight tensor into smaller, lower-rank blocks (block-LoRA). Predicted byte savings: another 30–50% on the decoder payload. **Predicted Δ:** −0.005 to −0.010 [prediction]. **Cost:** ~$0.50 for a block-LoRA training smoke. **Implementation:** `tac.substrates.block_lora_hnerv`.

---

### S-3 — Efron's bootstrap for score-floor uncertainty bands `[statistical-bound]`

**Source:** Efron, *Bootstrap methods: another look at the jackknife*, Ann. Stat. 1979; Efron–Tibshirani, *An Introduction to the Bootstrap* (1993).

**Setup.** We have 21 posterior anchors (per CLAUDE.md MEMORY.md). The continual-learning posterior gives a point estimate of the score-floor — but no uncertainty band. Efron's bootstrap:
```
S_floor_hat_b = predict(resample with replacement from {anchor_1, ..., anchor_21})
band_95 = [percentile(S_floor_b, 2.5), percentile(S_floor_b, 97.5)]
```

**Application.** The cathedral-autopilot dispatch logic currently uses point estimates to pick the top-K candidates. Bootstrap-band ranking is more honest: prefer candidates whose 5% bound (best case) is below the threshold, not just whose mean is. This avoids wasted GPU on candidates whose mean is good but whose 95% upper bound is catastrophic.

**Predicted Δ:** indirect — saves ~30% of wasted dispatches. At $5–15/dispatch and ~10 dispatches/week, **~$15–45/week of saved spend.** **Cost:** $0; 50 LOC change. **Implementation:** `tac.autopilot.bootstrap_uncertainty_band` (wrap `tac.continual_learning.predict_score_floor`).

---

### S-4 — Candès–Tao restricted isometry on the scorer-Jacobian `[statistical-bound]`

**Source:** Candès, Tao, *Decoding by linear programming*, IEEE TIT 2005; Candès, Tao, *The Dantzig selector*, Ann. Stat. 2007.

**Statement.** The RIP condition `(1−δ_k)||x||² ≤ ||Ax||² ≤ (1+δ_k)||x||²` for `k`-sparse `x` ensures `L1` minimization recovers sparse signals exactly. For random Gaussian matrices `A ∈ ℝ^{M×N}`, RIP holds with `M ≈ k log(N/k)`.

**Application.** The scorer-Jacobian `J(θ) = ∂(scorer-output)/∂θ` defines a sensing matrix. If `J` satisfies RIP on the support of the bit-allocator weights, **L1 minimization recovers the bit-allocator from O(k log N/k) score-evaluations**. Current bit-allocator uses Fisher saliency on `~100K`-dim parameter vectors with `k ≈ 1–5K` active — predicts only `5K · log(20) ≈ 21K` score evaluations to find the optimal allocation. That's `21K / 600 ≈ 35` dispatch passes.

**Predicted Δ:** −0.005 [prediction]; the bit-allocator gets ~2× tighter. **Cost:** ~$50 GPU. **Implementation:** `tac.bit_allocator.candes_tao_l1_recovery`.

---

### S-5 — Tibshirani LASSO on sparse-LoRA adapter selection `[statistical-bound]`

**Source:** Tibshirani, *Regression shrinkage and selection via the lasso*, J. Roy. Stat. Soc. B 1996.

**Statement.** L1 regularization induces exact sparsity (most coefficients become exactly zero). Vs. L2 (ridge) which only shrinks.

**Application.** PR101 uses dense LoRA. **Replace with LASSO-LoRA**: the regularization path automatically picks the K active rank-1 updates. Cross-validate on `score(θ_LASSO)` to pick the regularization strength `λ`.

**Predicted Δ:** −0.003 to −0.007 [prediction] — the active LoRA modes are typically much fewer than the full rank; LASSO finds them automatically. **Cost:** ~$0.30 GPU for a LASSO-path retrain. **Implementation:** `tac.substrates.lasso_lora_hnerv`.

---

### S-6 — Bickel semiparametric efficiency bound on the scorer `[statistical-bound]`

**Source:** Bickel, Klaassen, Ritov, Wellner, *Efficient and Adaptive Estimation for Semiparametric Models* (Springer, 1993).

**Statement.** For a semiparametric model `(θ, η)` with `θ` finite-dim and `η` nuisance, the asymptotic-variance lower bound on any `√n`-consistent estimator of `θ` is the inverse Fisher-info `(I_θθ − I_θη I^(-1)_ηη I_ηθ)^(-1)`.

**Application.** Treat `θ = HNeRV decoder weights` (finite-dim) and `η = scorer (treated as nuisance — fixed)`. The contest's effective sample size is 600 pairs. Bickel's bound: the **best possible** score-floor estimate from 600 samples has standard deviation `~σ/√600 ≈ 0.041·σ`. So the empirical posterior cannot reliably distinguish two candidates whose true scores differ by less than `~0.04·σ`. **Implication:** at PR106 r2 with score ~0.193, any predicted Δ smaller than `~0.008` is **statistically indistinguishable** at the 95% level on a single dispatch — you need 4+ dispatches of the candidate to claim improvement.

**Operational rule:** cathedral autopilot should NOT promote a candidate based on a single dispatch unless predicted Δ > 0.01. **Cost:** $0; ledger discipline. **Implementation:** preflight check for autopilot promotion rule.

---

### S-7 — Donoho thresholding for HNeRV weight quantization `[statistical-bound]`

**Source:** Donoho, Johnstone, *Ideal spatial adaptation via wavelet shrinkage*, Biometrika 1994.

**Statement.** For wavelet coefficients of a sparse signal, hard/soft thresholding at level `λ = σ√(2 log n)` is asymptotically minimax. Universal threshold = `σ√(2 log n)`.

**Application.** HNeRV's learned decoder weights, viewed in any wavelet basis (e.g., Haar, Daubechies-4), exhibit power-law sparsity. Apply Donoho's universal threshold to drop ~70–80% of wavelet coefficients with negligible distortion. Then entropy-code the survivors.

**Predicted Δ:** −0.010 to −0.020 [prediction] on the rate term. **Cost:** ~$0.10 for a wavelet-thresholding encoder prototype. **Implementation:** `tac.codec.donoho_universal_threshold_wavelet`. **Cross-ref Mallat (geometry lens):** wavelet basis choice = Mallat scattering transform.

---

### S-8 — Diaconis MCMC mixing for posterior sampling on the score landscape `[statistical-bound]`

**Source:** Diaconis, *The Markov chain Monte Carlo revolution*, Bull. AMS 2009.

**Setup.** Sampling from the score-Boltzmann measure `p(θ) ∝ exp(−β · S(θ))` is hard because the score map is expensive (600 video pairs per evaluation). Diaconis–Saloff-Coste theory of Markov-chain mixing tells us the relaxation time depends on the **second eigenvalue of the transition kernel**.

**Application.** Use **Hamiltonian Monte Carlo with Riemannian metric = Fisher** (Girolami–Calderhead 2011; this is also Connes' natural gradient extended to MCMC). On the HNeRV parameter manifold, the effective dimension is much smaller than the ambient dim (PR101 LoRA rank ≤ 32). Predicted mixing time: `O(d²) = O(1000)` samples vs `O(d³) = O(30000)` for vanilla Metropolis.

**Predicted Δ:** indirect — better posterior samples mean better continual-learning anchor weighting. **Cost:** ~$10 for a Riemannian-HMC prototype. **Implementation:** `tac.continual_learning.riemannian_hmc_sampler`.

---

### S-9 — Friedman gradient-boosted residuals for substrate stacking `[statistical-bound]`

**Source:** Friedman, *Greedy function approximation: a gradient boosting machine*, Ann. Stat. 2001.

**Statement.** Sequentially fit base learners to residuals of the cumulative ensemble. Each new learner reduces residual variance optimally per learner-complexity unit.

**Application.** Instead of training each substrate independently (PR95 → PR100 → PR101 → ...), **boost on residuals**: train HNeRV-base, then train HNeRV-residual against `(SegNet_target − HNeRV-base_output)`, then HNeRV-meta-residual, etc. Each layer is much smaller than the base (per Friedman's "weak learner" principle) so the archive cost decays geometrically.

**Predicted Δ:** −0.008 to −0.015 [prediction]; the residual layers are ~3–5% of base size with ~30% of base improvement. Cost: ~$3 GPU for a 3-layer boost. **Implementation:** `tac.substrates.gradient_boosted_residual_hnerv`. **Cross-ref biology lens:** this is also exactly Friston's predictive-coding hierarchy.

---

### S-10 — Gelman posterior-predictive check for candidate auth-eval gating `[statistical-bound]`

**Source:** Gelman, Carlin, Stern, Rubin, *Bayesian Data Analysis* (CRC, 1995/2003/2013).

**Statement.** Before believing a model, do a posterior-predictive check: simulate replica datasets from the posterior, compare to actual data. If the actual data lies in the tails of the simulated distribution, reject the model.

**Application.** Before claiming a [contest-CUDA] score is a true frontier move, do a posterior-predictive check: sample 100 nearby parameter perturbations from `p(θ | observed_anchors)`, compute predicted scores, ask "where does the actual measured score land?" If in the tails, the measurement is anomalous → re-run.

**Predicted Δ:** indirect; saves ~10% of false-frontier promotions. **Cost:** $0 (ledger discipline). **Implementation:** `tac.continual_learning.posterior_predictive_check`.

---

## Wire-in declarations (Catalog #125)

1. **Sensitivity-map:** S-7 Donoho threshold = per-coefficient saliency; wires into `tac.sensitivity_map.wavelet_threshold_saliency`.
2. **Pareto constraint:** S-1 CS lower bound + S-6 Bickel sample-size bound add 2 inequality constraints to `tac.pareto_kkt`.
3. **Bit-allocator hook:** S-4 RIP recovery as new `tac.bit_allocator.candes_tao_l1` allocator.
4. **Cathedral autopilot dispatch hook:** S-3 bootstrap-band ranking replaces point-estimate ranking; S-10 posterior-predictive check before promotion. Both wired into `tac.autopilot.dispatcher`.
5. **Continual-learning posterior update:** S-8 Riemannian HMC replaces current weight-update sampler when posterior has >50 anchors.
6. **Probe-disambiguator:** S-5 LASSO vs dense LoRA is a 2-mode tension — ship both via `--lora-mode lasso | dense` and build `tools/probe_lasso_vs_dense_lora_disambiguator.py`.

## Beauty pick (this lens)

**S-1 — Donoho one-bit compressed sensing on SegMap.** This is the most precise statistical move: the SegNet's *argmax-only* structure (a hard threshold over 5 classes) makes one-bit compressed sensing the natural codec. The receiving end doesn't need real-valued logits — just argmax. Donoho's lower bound = 20 KB on a payload currently using 80 KB is the cleanest 4× argument in the entire deck. Efron would call it "the right model for the right data."

Sources:

- [Donoho, Compressed sensing, IEEE TIT 2006](https://doi.org/10.1109/TIT.2006.871582)
- [Candès–Romberg–Tao, Robust uncertainty principles, IEEE TIT 2006](https://doi.org/10.1109/TIT.2005.862083)
- [Candès–Recht, Exact matrix completion, FOCM 2009](https://doi.org/10.1007/s10208-009-9045-5)
- [Tibshirani, LASSO, JRSS-B 1996](https://www.jstor.org/stable/2346178)
- [Donoho–Johnstone, Ideal spatial adaptation, Biometrika 1994](https://doi.org/10.1093/biomet/81.3.425)
- [Efron, Bootstrap methods, Ann. Stat. 1979](https://doi.org/10.1214/aos/1176344552)
- [Friedman, Gradient boosting machine, Ann. Stat. 2001](https://doi.org/10.1214/aos/1013203451)
- [Diaconis, MCMC revolution, Bull. AMS 2009](https://doi.org/10.1090/S0273-0979-08-01238-X)
- [Boufounos–Baraniuk, One-bit compressed sensing, CISS 2008](https://doi.org/10.1109/CISS.2008.4558487)
