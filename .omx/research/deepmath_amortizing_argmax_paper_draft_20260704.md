# Amortizing the Argmax — the differentiable geometry of a frozen-scorer task-space witness

**Synthesis draft (task #284, 2026-07-04).** Consolidates the six-chapter deep-math pass
(`deepmath_lens_{tropical_ot_powerdiagram, infogeo_naturalgrad, singularity_duality,
phasefield_gmt_levelset, microlocal_se3_code, dynamics_transition_easing}_20260704.md`) into one
document + a measured/provable/conjecture ledger. **MEANS, not ends** — this paper does not move the
pointer (contest-CPU **0.19110**, UNMOVED); it *names the geometry* of the witness we are training
(#205) so the levers stop being heuristics. Every proven law here is registered as a canonical
equation (A2, `deepmath_amortizing_argmax_laws_20260704.py`); every conjecture is flagged and NOT
registered. Governing discipline: NO-FAKE (no fabricated citation or number), means/ends firewall.

---

## 0. The one-sentence thesis

> **A frozen classifier's decision boundary is one object — a Fisher-metric anisotropic Laguerre
> tessellation — and "training a witness to match its argmax" is a temperature-annealed mirror-descent
> continuation (`softmax_τ`, `τ→0`) that dequantizes the smooth boundary onto the sharp one; the
> distortion we minimize is the boundary's perimeter, its optimal code is the curvelet, and its
> temporal transport is a single se(3) screw.**

Everything below is a facet of that sentence, and the honest §4 says *why it is one object with several
distinct deformations rather than one theorem.*

---

## 1. The object — the separatrix as a five-fold limit

The scored quantity is `d_seg = ` Hamming (symmetric-difference) rate of two hard-argmax labelings of a
frozen SegNet (EfficientNet-B2, smooth). The boundary of that partition — the **separatrix** — is
simultaneously, and provably:

| view | statement | status | chapter |
|---|---|---|---|
| **power/Laguerre diagram** | argmax cells are a power diagram (exact for ReLU nets — Balestriero–Baraniuk 2019; locally-curved for our smooth net). Our simplex probe (top1−top2 = exact distance-to-flip, min-diff **0.0 over 118 M px**) is the first-order power-diagram fact. `d_seg` = **Hamming** mismatch of two Laguerre labelings, **not** a Wasserstein cost. | PROVEN (local) | 1 |
| **tropical variety / Maslov limit** | `softmax_τ → argmax` as `τ→0` is the `(+,×)→(max,+)` semiring (Maslov) dequantization, `ħ=τ`, exact error `∈ [0, τ·ln 5]`. The hard partition is a tropical-hypersurface complement. Our `τ:1.0→0.05` curriculum *is* this limit. | PROVEN | 3 |
| **optical caustic** | the Fisher information field `I_τ = τ⁻² σ(m/τ)(1−σ(m/τ))` is a rank-1 bright ridge on the margin-zero Maxwell set, collapsing to the tropical variety as `τ→0`. The **measured `curvature↔(−margin)` Pearson 0.978** is this caustic — an **exact identity** (see §2), near-tautological ⇒ trustworthy. | PROVEN | 2,3 |
| **cartoon edge / curvelet** | the partition is a *cartoon* (piecewise-C² with C² boundary) ⇒ the N-term-optimal sparse chart is the curvelet/shearlet, error `O(N⁻²(log N)³)` ≫ wavelet `N⁻¹` ≫ Fourier `N⁻¹ᐟ²`. **Our `--self-orient` basis IS a discrete shearlet** — which is *why* the D1 directional-basis lever measures **−48% all-class** (Candès–Donoho cartoon-optimal, not a heuristic). | PROVEN (upper bound) | 5 |
| **se(3) Lie-orbit** | Chasles: every rigid motion is a screw `ξ∈se(3)`; the exact planar homography `H=K(R−t nᵀ/d)K⁻¹` gives `Σ_t = H_t(Σ_0)` for the static-class boundary. **`ξ` is the temporal sufficient statistic — encode once** (measured #257: 13.9× rate cut, bit-exact). | PROVEN (geometry) | 5 |

---

## 2. The proven spine (registered as canonical equations)

Eight load-bearing laws, each measured or derived, none analogized:

1. **Maslov dequantization bound** — `‖softmax_τ(z) − argmax(z)‖` contributes at most `τ·ln K` (`K=5`)
   to the free energy; the curriculum's `τ→0` *is* the semiring limit. `maslov_dequantization_bound_v1`.
2. **Fisher = caustic = 0.978 (exact identity)** — "Fisher curvature" `1−Σp²` **is** `tr[diag(p)−ppᵀ]`,
   the categorical Fisher trace in logit coordinates; in the two-class annulus `tr F = ½ sech²(m/2)`, a
   deterministic monotone function of the margin `m`. That is *why* Pearson = 0.978 (band) / Spearman
   0.908 (global). The margin field is a byte-faithful Fisher surrogate.
   `fisher_curvature_equals_categorical_fisher_trace_caustic_v1`.
3. **CE + softmax = mirror descent = natural gradient** — CE is the Bregman divergence of the
   categorical entropy potential; Raskutti–Mukherjee 2015 proves MD ≡ natural-gradient in dual
   coordinates. The curriculum-as-mirror-descent is a theorem. `ce_softmax_mirror_descent_natural_gradient_v1`.
4. **Shearlet N-term upper-bounds the task rate** — `d_seg` is a boundary-displacement functional
   ⇒ the Fisher/margin-weighted shearlet N-term count is a **proven upper bound** on the task-space
   rate `R_X(D_Y)`. (Tightness conjectured — see §4.) `shearlet_nterm_upper_bounds_task_rate_v1`.
5. **se(3) screw = temporal sufficient statistic** — `Σ_t = H_t(Σ_0)`; encode `ξ` once, derive `H(ξ)`
   free at decode. `se3_screw_temporal_sufficiency_v1`.
6. **τ = ε = ħ (one dequantization at two scales)** — the single scalar `τ` is simultaneously the
   Maslov/tropical Planck constant (pointwise `logsumexp→max`), the Modica–Mortola diffuse-interface
   width (spatial energy), and the mirror-descent temperature (softmax = neg-entropy Bregman mirror
   map). `tau_eps_hbar_one_dequantization_two_scales_v1`.
7. **Multi-phase Modica–Mortola Γ-limit** — `softmax(φ/τ)` of the K SDFs, with the length term
   (`δ_ε·|∇φ|` = Chan–Vese coarea perimeter) and the entropy barrier `τH(p)` as the double-well,
   Γ-converges (Baldo/Sternberg multi-phase, K=5 wells, Herring triple-junction angles) to the
   **weighted perimeter of the argmax partition** as `τ→0`. `multiphase_modica_mortola_perimeter_gamma_limit_v1`.
8. **MCF minority-erasure inevitability** — the perimeter-gradient training flow is (sharp-limit)
   motion by mean curvature (Bronsard–Kohn), which annihilates high-curvature features first ⇒ thin
   Lanes / small Movables erase *inevitably*, not as an artifact. Measured receipt: the MBO probe
   attributes **95.7% of smoothing cost to Lane**. The principled cure is a per-class area/volume
   constraint (auction-MBO), the sibling of the shipped `lane_thin_weight_map`.
   `mcf_minority_erasure_inevitability_v1`.

---

## 3. τ = ε = ħ — the hard↔soft duality resolved

The pass's pivotal cross-chapter tension: is the *temporal* Maslov temperature `τ` the same as the
*spatial* Modica–Mortola interface width `ε`? Ch.2's Claim 2.★ (the `τ`-softmax free energy
Γ-converges to the perimeter functional with the entropy term as the double-well) and Ch.4's explicit
phase-field construction agree: **yes.** The pointwise semiring limit and the spatial Γ-limit *coincide*
— hard↔soft duality is **one dequantization at two scales**. Level-set training is mirror-descent
graduated-non-convexity (GNC) / numerical continuation *along* this single `τ`, and Γ-convergence is the
theorem that the soft continuation lands on the hard perimeter-minimizer. The witness IS the classical
level-set variational triple exactly (softmax = entropic multi-phase field; length = Chan–Vese coarea
perimeter = the Γ-limit functional; eikonal = Hamilton–Jacobi SDF), and choosing the eikonal on the
**margin** `m=φ₁−φ₂` makes the interface half-width exactly `τ/2`.

---

## 4. The honest boundary — one object, several deformations (NOT one theorem)

The most important discipline of the pass. The separatrix is **one limit object** (power diagram =
tropical variety = caustic = cartoon = Lie-orbit), but it is reached by **distinct small-parameter
deformations that do not collapse to a single equivalence**:

- `τ = ε = ħ` is **one** deformation (§3) — the tropical/Modica–Mortola/mirror-descent Planck constant.
- **step-size** (the Fisher/MD discretization) is a *distinct* deformation.
- **`l7` `p→∞`** (`Lᵖ→L^∞` sharpening) is a *distinct* deformation.
- **Muon** is a *preconditioner outside the continuum* — see §5.

So the paper's thesis is not "one grand unification theorem" but the sharper, honest claim: **the
separatrix is the common limit object of several distinct geometric deformations, and the CE→τ→l7→Muon
curriculum is a continuation that visits them in scale order.** The **conjecture** (explicitly NOT
registered as a law): the trajectory literally *is* natural-gradient-along-Maslov ending at the Γ-limit
— it *matches* the measured stage behavior (CE→tau `d_seg −21.6%`, junctions `−14%`) but is unproven.

---

## 5. NO-FAKE catches (these supersede prior notes — propagate)

The pass self-caught three fake-adjacent claims and one false friend:

1. **"198:1 annulus anisotropy" is DISPUTED.** Ch.2 `grep` of the entire cache finds no cached 198;
   re-measurement gives **9.56:1 (gradient projection) / 37.8:1 (structure-tensor eigenvalue)** vs
   Lens 1's on-the-fly 198:1. The *qualitative* strong-codim-1 anisotropy holds (⇒ directional basis);
   the *magnitude* needs reconciliation. **Do not quote 198:1 as settled.** (Registered as the
   anisotropy-correction anchor, not a headline number.)
2. **"Muon = natural gradient" is a FALSE FRIEND (literal).** Muon = spectral-norm steepest descent in
   **weight** space (Bernstein–Newhouse), ≈ 1-step Shampoo ≈ Gauss–Newton — *not* the categorical
   Fisher–Rao natural gradient in **output** space. The **−32% d_seg is real**, but its attribution to
   "natural-gradient-ness" is **conjectural** (equally explained by `κ≈19` boundary-Hessian busting;
   Ch.6 confirms Muon ≈ Stiefel-flow, 2605.13079). Registered with `empirical_verification_status`
   distinguishing the measured win from the conjectural mechanism.
3. **"margin = UNIWARD cost" stays RETIRED.** Pixelwise correlation ≈ +0.04 (near-zero); UNIWARD is a
   *texture* cost, the margin is a *logit* distance. Keep **margin = Fisher = caustic**; the unity is
   metric-level (Fisher/Jacobian), not scalar-map. (Sister:
   `uniward_margin_pixelwise_nearzero_unity_is_metric_level_not_scalarmap`.)
4. **Disarmed false friends (do NOT build):** deconvolve-R (R is near all-pass ⇒ ≤ +1.25 dB);
   output-space `F⁻¹` natural-gradient bolt-on (softmax+CE gradient `(p−y)` already *is* the NG-simplified
   update — redundant); sub-2px basis refinement (below the SegNet stem Nyquist — needs a *store*, not a
   finer chart); "MFLD `λ≈1` = our τ" (do not pin a stage boundary to `λ=1`).

---

## 6. The measured / provable / conjecture ledger

| claim | class | evidence |
|---|---|---|
| top1−top2 = exact distance-to-flip | MEASURED | min-diff 0.0 / 118 M px |
| `1−Σp²` = categorical Fisher trace; annulus `½ sech²(m/2)` | PROVABLE + MEASURED | derivation + Pearson 0.978 / Spearman 0.908 |
| `softmax_τ→argmax` error `≤ τ·ln 5` | PROVABLE | Maslov dequantization |
| CE+softmax = mirror descent = NG | PROVABLE | Raskutti–Mukherjee 2015 |
| self-orient basis = discrete shearlet ⇒ −48% | MEASURED + PROVABLE | D1 lever measurement + Candès–Donoho |
| shearlet N-term ≥ `R_X(D_Y)` upper bound | PROVABLE (tightness conjectured) | boundary-displacement functional |
| `Σ_t=H_t(Σ_0)`; `ξ` sufficient; 13.9× rate cut | PROVABLE + MEASURED | Chasles/homography + #257 |
| `τ=ε=ħ` one dequant, two scales | PROVABLE | Γ-convergence (Baldo/Sternberg) |
| MCF erases thin-Lane first (inevitable) | PROVABLE + MEASURED | Bronsard–Kohn + MBO 95.7% Lane |
| trajectory = NG-along-Maslov | **CONJECTURE** | matches CE→tau −21.6%; unproven — NOT registered |
| shearlet-tightness hits the task-rate floor | **CONJECTURE** | annulus 1.4%-cell concentration; unproven |
| Muon win = "natural-gradient-ness" | **CONJECTURE** | −32% real, attribution open |
| anisotropy magnitude (9.56 vs 37.8 vs 198) | **DISPUTED** | needs reconciliation |

---

## 7. The engineering payload — the rate half of sub-0.15, made concrete

**The RATE half:** the counted `archive.zip` payload = `|Fisher/margin-weighted shearlet coeffs of Σ
above τ| + |ξ B-spline| + |Movable store|`; the generic shearlet bank, the openpilot lane raster, and
`ξ→H` are all **rule-118 FREE** (generic algorithm in `inflate.py`). §2.4 makes this a *proven upper
bound* on the rate; §2.5 makes the temporal axis a single screw.

**The ranked cross-chapter-CONVERGED next-run config** (all $0 config-mostly, A/B-owed, net-S #205-gated;
this is task #285 / B):

1. **Ch.4 geometric-τ + raised eikonal (COUPLED, do first):** `--tau-anneal-shape geometric`
   (equal epochs per octave = scale-space/GNC-correct) + float `--softmax-temp-end 0.05→~1.0` (floor at
   resolution scale — 0.05 = 0.025 px interface = 40× sub-grid aliasing, wasted) + raise
   `--eikonal-weight 0.01→0.05` (required to make `τ` a real interface width). Convergent with the
   code's own note that geometric "slows late-τ d_seg volatility."
2. **Ch.5 M1 along-tangent + M2 NTK-whitening:** `--n-dir-freqs 2→4` with `--freq-across 8` (respect the
   Nyquist cap `freq_across·2^(n_dir_freqs−1) ≤ 64`) to resolve the dashed-lane wavefront; per-scale
   band-pass whitening (amplitude `∝ 1/√λ`) = the microlocal preconditioner (dominant SPEED lever
   ~3–10×, up to `−3e-4` d_seg).
3. **Ch.6 L1+L2 transition-easing:** `--lane-band-start-epoch 350` (deconflict from `tau@300`) +
   `--stage-transition-rewarmup-epochs 20` (BUILT, default-off) — attacks the MEASURED ep300 bump
   (`d_seg 0.0056→0.020`, 3.4×) which is a *numerical-continuation* failure, not a loss failure.
4. **Ch.1 dash-comb + AHA logit-offset:** the tropical/PL max-plus dash comb (phase = ego-`ξ`) closes the
   3.2× along-tangent deficit at rate-cheap counted bytes; the AHA mass-matched Laguerre logit-offset
   head sets each class-cell logit offset so cell mass = GT frequency (principled LDAM, byte-free,
   counters Lane/Movable collapse).

**DON'T build:** deconvolve-R, output-space `F⁻¹` NG, sub-2px basis refinement (§5).

---

## 8. Novel-contribution accounting (NO-FAKE #7 — originality is earned, not claimed)

What is *ours-original* vs known, honestly separated (the paper is a synthesis of known theorems applied
to a specific frozen-scorer geometry — the novelty is the *application + measurement*, not the theorems):
- **Known (cited):** Maslov dequantization; Balestriero–Baraniuk power diagrams; Candès–Donoho curvelets;
  Raskutti–Mukherjee MD≡NG; Baldo/Sternberg multi-phase Modica–Mortola; Bronsard–Kohn MCF; Chasles/
  homography; Hörmander wavefront transport; Bernstein–Newhouse Muon.
- **Ours-original (the contribution):** (a) the *measurement* that the frozen SegNet's separatrix is a
  cartoon and its self-orient basis IS a shearlet, with the −48% receipt; (b) the *exact identity*
  `1−Σp² = tr F` grounding the measured 0.978 caustic; (c) `τ=ε=ħ` as the operating principle of a
  *task-space* witness (not RGB reconstruction); (d) the shearlet-N-term-as-task-rate-upper-bound framing
  tying the differentiable geometry to the contest rate term; (e) the se(3)-screw-transports-the-code
  unification. **None of this is a score until a byte-closed exact row beats 0.19110** (means/ends).

---

## 9. Next

- **A2 (done with this draft):** register the 8 proven laws + the anisotropy correction as canonical
  equations (`deepmath_amortizing_argmax_laws_20260704.py`) so the equations leg of the triality AGREES.
- **Triality:** DAG FEED-03z (synthesis complete) + DSL gauge sync (#271) — the three legs consistent.
- **B (#285):** fold the §7 ranked config into #270 / the next fresh run (config plan only; operator GO
  gates any dispatch — CONTAINMENT).
- **The pointer moves only** when one of these levers, byte-closed, produces an `upstream/evaluate.py`
  n600 row < 0.19110. Everything above is MEANS.

*Sisters:* `deepmath_amortizing_argmax_maslov_caustic_tau_eps_hbar` (memory) · DAG FEED-03y/03z ·
`project_gr_unified_action_full_witness_architecture` · `project_unified_variational_levelset_flow_everything_is_facets`.
