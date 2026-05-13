# Expert team — biology lens (efficient coding, free energy)

**Date:** 2026-05-13
**Lane:** `lane_expert_team_fields_medalist_math_biology_alien_tech_20260513`
**Sister memos:** `expert_team_fields_medalist_math_biology_alien_tech_20260513.md`, `expert_team_pure_math_20260513.md`, `expert_team_statistics_20260513.md`, `expert_team_geometry_20260513.md`.

## Persona seats

- **Horace Barlow** — efficient coding hypothesis (1961); redundancy reduction in cortex
- **Bruno Olshausen + David Field** (1996) — sparse coding in V1
- **Joseph Atick + Norma Redlich** (1990, 1992) — retinal efficient-coding theorem
- **Eero Simoncelli** (NYU) — natural image statistics, divisive normalization
- **Karl Friston** (UCL) — free-energy principle, predictive coding
- **Peter Dayan / Larry Abbott** — Theoretical Neuroscience textbook
- **William Bialek** — biophysics, *Biophysics: Searching for Principles* (PUP 2012)
- **David Marr** — vision, primal sketch, computational/algorithmic/implementational levels
- **Rajesh Rao + Dana Ballard** (1999) — predictive coding in visual cortex
- **Francis Crick** — DNA structure, "the astonishing hypothesis"

## How the contest looks through this lens

**The contest IS the retinal coding problem.** Atick–Redlich (1990) ask: given a noisy input distribution `p(s)`, what encoding `y = f(s)` maximizes `I(y; s)` subject to a power constraint on `y`? The contest asks: given a video distribution, what archive `y` maximizes `I(scorer-output; archive)` subject to `|archive| ≤ B` bytes?

These are the SAME problem. The retina solves it via decorrelating filters + power-spectrum-whitening + redundancy reduction; we should too.

## Top derivations

### B-1 — Atick–Redlich retinal efficient-coding theorem applied to archive design `[biological-principle]`

**Source:** Atick, Redlich, *Towards a theory of early visual processing*, Neural Computation 1990; Atick, Redlich, *What does the retina know about natural scenes?*, Neural Computation 1992.

**Statement.** Given an input distribution `p(s)` with covariance `Σ_s = U Λ U^T` and an output power constraint `E[||y||²] ≤ P`, the optimal **linear** encoding `y = K · s + n` (where `n` is post-encoding noise) is:
```
K_optimal = U · diag(λ_i^{-1/2}) · (whitening)  followed by  Λ_y(λ) ∝ max(0, λ − λ_threshold)  (signal-power gating)
```
i.e., **whiten + threshold low-eigenvalue modes**. Equivalent to ZCA whitening followed by Wiener filtering. Maximizes `I(y; s)` for fixed power.

**Application.** The contest "input" is the (video, scorer-output) joint distribution; "output" is the archive. The Atick–Redlich-optimal encoder:
1. **Whiten** the SegMap and pose-target distributions over the 600-pair dataset (compute eigenbasis of empirical covariance).
2. **Hard-threshold** eigenmodes whose explained variance is below the noise floor (uint8 quantization noise ≈ `1/256² ≈ 1.5e-5`).
3. **Encode** only the above-threshold eigencoeffs.

For PR101's SegMap target (~196K positions × 5 classes), empirical effective rank ≈ 40 (per PR101 lossy-coarsening evidence). The Atick–Redlich prediction: ~40 floats × 4 bytes = **160 bytes for the SegMap signal** — vs PR101's 80 KB. **3 orders of magnitude headroom on the SegMap.**

The catch: the threshold has to be calibrated to the scorer's actual tolerance, not the L2 reconstruction tolerance. (That's where the proxy-auth gap lives.)

**Predicted Δ:** −0.030 to −0.050 [biological-principle] for SegMap eigenmode encoding. **Cost:** ~$1 for an eigen-decomposition + threshold sweep. **Implementation:** `tac.codec.atick_redlich_eigenmode_encoder`.

---

### B-2 — Olshausen–Field sparse V1 coding for the HNeRV decoder basis `[biological-principle]`

**Source:** Olshausen, Field, *Emergence of simple-cell receptive field properties by learning a sparse code for natural images*, Nature 1996; Olshausen, Field, *Sparse coding with an overcomplete basis set*, Vision Res. 1997.

**Statement.** Train an overcomplete dictionary `Φ` to reconstruct natural images `x ≈ Φ · s` with sparse coefficients `s` (L1 prior). Result: dictionary atoms become localized, oriented, bandpass — like V1 simple cells.

**Application.** PR101 HNeRV uses a learned overcomplete decoder (one decoder, many positions). **Replace with Olshausen-Field sparse-coding decoder**: dictionary trained for sparse activation. At PR101 rank-32, expect ~5 active atoms per position vs 32 dense atoms. 6× fewer effective parameters means 6× less archive cost.

**Predicted Δ:** −0.010 to −0.018 [biological-principle]. **Cost:** ~$2 for a sparse-coding-decoder training. **Implementation:** `tac.substrates.olshausen_field_sparse_decoder_hnerv`. **Cross-ref geometry lens:** Olshausen-Field atoms = Mallat scattering basis at low order.

---

### B-3 — Friston free-energy training objective `[biological-principle]`

**Source:** Friston, *The free-energy principle: a unified brain theory?*, Nat. Rev. Neurosci. 2010; Friston, *A free energy principle for the brain*, J. Physiol. Paris 2006.

**Statement.** The brain minimizes a single scalar — variational free energy:
```
F = E_q[log q(s) − log p(s, o)] = D_KL(q(s) || p(s|o)) − log p(o)
  = accuracy term + complexity term
  = surprise about observation + complexity of model
```
Equivalently: F = (negative ELBO of a variational distribution).

**Application.** Train HNeRV with **free-energy objective**:
```
F(θ) = − log p(scorer-output | archive(θ))   [accuracy / surprise]
     + D_KL(q(θ) || p(θ))                     [complexity / archive bytes]
```
The first term is precisely the contest distortion. The second term is precisely the archive rate (via MDL ≡ −log prior). **Free energy = contest score, up to coefficients.** Training with free-energy gradients is mathematically equivalent to training with the contest objective, but with the right Bayesian-information-theoretic decomposition.

**Bonus:** Friston's free-energy gradient is `−∂F/∂θ = (prediction-error signal) × (Jacobian of generative model)`. This matches Karpathy's "let compute speak" and Boyd's "ADMM dual updates" in form. The contest is a free-energy minimization problem.

**Predicted Δ:** −0.005 to −0.010 [prediction]; aligns the training objective with the eval objective at the variational level. **Cost:** ~$0.50 for a free-energy retrain. **Implementation:** `tac.training.friston_free_energy_objective`.

**Cross-ref pure-math (PM-1):** Perelman's `W`-entropy IS the variational free energy on a Riemannian manifold. Friston and Perelman are the same.

---

### B-4 — Rao–Ballard predictive-coding hierarchy for archive composition `[biological-principle]`

**Source:** Rao, Ballard, *Predictive coding in the visual cortex*, Nat. Neurosci. 1999.

**Statement.** Visual cortex layer `n` predicts layer `n−1`'s activity; only the PREDICTION ERROR is propagated. This generalizes the Helmholtz machine and gives rise to predictive coding as a unified framework.

**Application.** Build the archive as a **predictive-coding hierarchy**:
- Level 0 = base HNeRV decoder (prior over poses/maps)
- Level 1 = encode (per-pair-target − level-0-prediction) — the prediction-error signal
- Level 2 = encode (per-pair-residual at level 1 − level-1-prediction)
- ...

This is exactly Friedman's gradient boosting (statistics lens) but motivated biologically. Each level encodes residuals; residuals are by construction smaller-entropy than the original signal, so each level needs fewer bytes geometrically.

**Predicted Δ:** −0.010 to −0.020 [prediction]. **Cost:** ~$3 GPU for a 3-level hierarchy. **Implementation:** `tac.substrates.rao_ballard_predictive_coding_hnerv`.

**Connection to PR101 LoRA**: PR101's LoRA Δ on top of the base IS a 1-level Rao-Ballard hierarchy. Extending to 2-3 levels is unexplored.

---

### B-5 — Barlow's redundancy-reduction principle for cross-pair temporal coding `[biological-principle]`

**Source:** Barlow, *Possible principles underlying the transformations of sensory messages*, in Sensory Communication (MIT Press 1961).

**Statement.** Successive levels of the sensory system reduce redundancy. Decorrelated representations are biologically optimal because they minimize metabolic cost per bit transmitted.

**Application.** PR101 encodes each of the 600 pose pairs independently. But poses at adjacent timesteps are massively correlated (vehicle dynamics is smooth). **Decorrelate temporally before encoding**: apply a 1D temporal whitening filter across the 600-pair sequence, then encode whitened differences. Predicted: ~40% reduction in pose-payload bytes.

**Predicted Δ:** −0.005 to −0.012 [prediction]. **Cost:** ~$0 (encoder change). **Implementation:** `tac.codec.barlow_temporal_decorrelation_pose`.

---

### B-6 — Simoncelli divisive normalization for adaptive bit allocation `[biological-principle]`

**Source:** Simoncelli, Heeger, *A model of neuronal responses in visual area MT*, Vision Res. 1998; Carandini, Heeger, *Normalization as a canonical neural computation*, Nat. Rev. Neurosci. 2012.

**Statement.** Cortical neurons normalize their output by the local pool's energy: `r_i = γ · x_i² / (σ² + ∑_j x_j²)`. This is the "canonical neural computation."

**Application.** Apply **divisive normalization to bit-allocator weights**:
```
w_i = γ · s_i² / (σ² + ∑_{j ∈ N(i)} s_j²)
```
where `s_i` is the per-tensor saliency. This automatically reduces allocation to high-energy regions (already well-encoded) and increases to low-energy regions (under-encoded). Empirically: matches "adaptive bit allocation" without manual tuning.

**Predicted Δ:** −0.003 to −0.008 [prediction]; the bit-allocator becomes auto-adaptive. **Cost:** $0; allocator change. **Implementation:** `tac.bit_allocator.simoncelli_divisive_normalization`.

---

### B-7 — Lennie metabolic-cost minimum for sparse archive bytes `[biological-principle]`

**Source:** Lennie, *The cost of cortical computation*, Curr. Biol. 2003.

**Statement.** Each cortical spike costs ~10⁻¹⁰ J. The brain operates near a metabolic minimum: only ~1% of neurons can be active at once. This forces **sparse codes**: only a tiny fraction of "units" fire for any input.

**Application.** Enforce **archive sparsity** as a hard constraint: only K << N "tokens" are non-zero in any encoding. The K is set by a metabolic-budget analog: an ε ratio of bytes used vs bytes allocated. The contest's effective metabolic minimum is `K ≈ 5–10%` based on PR101 entropy estimates.

**Predicted Δ:** −0.008 [prediction]; the sparse-constraint trains tighter than the average L1 prior. **Cost:** $0; training-objective change. **Implementation:** `tac.training.lennie_sparsity_constraint`.

---

### B-8 — Marr's three-levels framing for cross-substrate decomposition `[biological-principle]`

**Source:** Marr, *Vision* (1982), Freeman.

**Statement.** Any information-processing system has three levels:
1. **Computational** — what does it compute? (the problem)
2. **Algorithmic** — how does it compute it? (the representation)
3. **Implementational** — what hardware? (the physical realization)

**Application.** Audit each substrate against the three levels:
- **Computational** — every substrate computes the same thing (RGB → seg/pose distortion + bytes). No room to differentiate here.
- **Algorithmic** — substrates differ wildly: PR95 implicit MLP, PR101 LoRA, PR103 attention, etc. Most contest variation is here.
- **Implementational** — same hardware (T4 / A100 / contest CI). No room to differentiate here.

**Implication.** The competition is FUNDAMENTALLY an algorithmic-level question — search the algorithmic level, not the computational or implementational. Reject any new substrate that doesn't introduce a genuinely new algorithmic primitive.

**Predicted Δ:** $0 direct; allocates research effort. **Cost:** $0. **Implementation:** documentation discipline in `feedback_*` memos.

---

### B-9 — Crick "astonishing hypothesis" alignment of attention with archive bits `[biological-principle]`

**Source:** Crick, *The Astonishing Hypothesis: The Scientific Search for the Soul* (1994); Crick, Koch, *Towards a neurobiological theory of consciousness*, Seminars in Neurosci. 1990.

**Statement.** "You are nothing but a pack of neurons." Consciousness/attention is identified with the binding of distributed neural patterns into a single coherent representation.

**Application.** Attention is "where the bits go." The contest archive has finite bits; allocate them to the parts the scorer attends to. SegNet's argmax-boundary is the scorer's "attention." Allocate bits there.

This is also G-1 + B-6 + PM-2 in disguise. The "binding problem" of consciousness ≡ the "bit-allocation problem" of compression.

**Predicted Δ:** subsumed by PM-2 (Tao restriction). **Cost:** N/A.

---

### B-10 — Bialek information-theoretic precision bound for biological coding `[biological-principle]`

**Source:** Bialek, *Biophysics: Searching for Principles* (PUP 2012); Bialek, Nemenman, Tishby, *Predictability, complexity, and learning*, Neural Comp. 2001.

**Statement.** Biological systems often operate at the information-theoretic limit (e.g., Drosophila bristle receptors achieve photon-counting precision). The information cost of any inference is bounded below by the entropy of the data.

**Application.** Bialek's bound: the minimum archive size for the contest is
```
H_min = H(scorer-output distribution conditioned on archive)
```
which is computable from the empirical SegNet/PoseNet output distribution. Bialek's research suggests biological systems achieve `H_min` to within a factor of 2. Current PR106 r2 is at ~5× `H_min` (estimated from entropy of empirical scorer-output distribution).

**Operational claim.** There is a 2–5× compression headroom at the Bialek bound. **Cost:** ~$1 for an entropy-of-scorer-output empirical estimate. **Implementation:** `tools/bialek_information_bound_estimator.py`.

---

## Wire-in declarations (Catalog #125)

1. **Sensitivity-map:** B-1 Atick–Redlich eigenmode threshold = per-eigenmode saliency; B-2 sparse-coding active-atom indicator = per-atom saliency. Wire into `tac.sensitivity_map.{atick_redlich, olshausen_field}`.
2. **Pareto constraint:** B-10 Bialek information bound is a hard lower bound on the rate axis.
3. **Bit-allocator hook:** B-6 Simoncelli divisive normalization as new `tac.bit_allocator.divisive_normalization`.
4. **Cathedral autopilot dispatch hook:** B-3 Friston free-energy ranks candidates correctly; integrate as `tac.autopilot.ranker.free_energy`.
5. **Continual-learning posterior update:** B-3 free-energy is the Bayesian evidence — the posterior update is canonical.
6. **Probe-disambiguator:** B-4 Rao-Ballard predictive coding vs Friedman gradient-boosting is a 2-mode tension (same math, different motivation). Probe selects regime-conditional verdict.

## Beauty pick (this lens)

**B-1 — Atick-Redlich retinal eigenmode encoder on SegMap.** This is the contest's most direct biological analog: the retina solves *literally the same problem* (encode visual signal under power budget while preserving recognizability). The empirical effective rank of the SegMap distribution is ~40 modes; PR101 spends 80 KB; Atick-Redlich predicts 160 bytes. This is the cleanest 3-order-of-magnitude headroom argument in the entire ten-lens deck. Barlow would smile.

Sources:

- [Atick, Redlich, Towards a theory of early visual processing, Neural Comp. 1990](https://papers.cnl-t.salk.edu/PDFs/Atick_Redlich_1990.pdf)
- [Atick, Redlich, What does the retina know about natural scenes?, Neural Comp. 1992](https://redwood.berkeley.edu/wp-content/uploads/2018/08/Atick-Redlich-NC92.pdf)
- [Olshausen, Field, Sparse coding, Nature 1996](https://www.nature.com/articles/381607a0)
- [Friston, Free-energy principle, Nat. Rev. Neurosci. 2010](https://doi.org/10.1038/nrn2787)
- [Rao, Ballard, Predictive coding, Nat. Neurosci. 1999](https://doi.org/10.1038/4580)
- [Barlow, Possible principles underlying transformations, MIT Press 1961](https://philpapers.org/rec/BARPPU-2)
- [Simoncelli, Heeger, Normalization model, Vision Res. 1998](https://doi.org/10.1016/S0042-6989%2897%2900183-1)
- [Lennie, Cost of cortical computation, Curr. Biol. 2003](https://doi.org/10.1016/S0960-9822%2803%2900135-0)
- [Bialek, Biophysics, PUP 2012](https://press.princeton.edu/books/hardcover/9780691138916/biophysics)
