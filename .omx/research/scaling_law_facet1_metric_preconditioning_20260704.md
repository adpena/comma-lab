# Scaling-law engineering — FACET 1: the TRAINING METRIC as the d_seg-vs-compute scaling-law lever

**Task: the 4-facet "geometry-optimal scaling-law engineering" pass (operator: "do like we did
with Muon — engineer to change scaling laws to be optimal based on differentiable geometry, bridged
with deep math and engineering"). This is FACET 1 — the training metric (the Muon generalization).**
$0 research, no heavy/paid/GPU, #205 read-only. **MEANS** — pointer contest-CPU **0.19110** UNMOVED;
nothing here is a score until a byte-closed `upstream/evaluate.py` n600 row beats it. NO-FAKE: every
number below is either cited from the context memos (not re-measured) or DERIVED from standard theory;
none is fabricated. Grepped the trainer before naming any flag.

Context read (not re-derived): `.omx/research/deepmath_amortizing_argmax_paper_draft_20260704.md` §5
(false friends) · memory `muon_deep_dive_keep_and_tune_finishing_stage_schedule_not_switch_20260703`
· memory `deepmath_amortizing_argmax_maslov_caustic_tau_eps_hbar_20260704` · sister
`project_sig_proc_filter_chain_measured_R_allpass_L3_ntk_20260701` ·
`lane_dash_residual_root_is_along_tangent_freq_deficit_R_allpass_20260703`. Built surface verified by
grep (see §5).

---

## 0. The one-sentence answer

> **The training metric splits the d_seg-vs-compute curve into TWO regimes with DIFFERENT levers: in
> the well-conditioned boundary basin (κ≈19, FINITE) every fixed metric gives GEOMETRIC convergence
> and the metric only sets the RATE CONSTANT — this is where Muon's spectral flattening lives, which
> is why the measured −32% is a constant (rate) win, NOT an asymptotic-exponent win; the EXPONENT of
> the loss-vs-compute power law can only be changed in the FLAT, spectrum-power-law regime — the
> fine-scale lane-dash tail governed by the NTK/feature-Gram spectral bias — where a per-scale
> feature-Gram WHITENING preconditioner (natural gradient in FEATURE space, distinct from the
> REDUNDANT categorical-OUTPUT-space Fisher) flattens the eigenvalue tail and is therefore the
> geometry-optimal exponent-lever.**

Metric-engineering with what we have BUILT (Muon) changes the **constant**. Changing the **exponent**
needs a feature-space whitening build, and only if the tail is spectrum/saddle-limited rather than
STE-flicker-floor-limited (a $0 pre-metric decides — §7 Lever C/A).

---

## 1. The convergence-rate theory — what the metric changes (exponent vs constant)

Preconditioned gradient flow on a Riemannian manifold with metric `M` (the preconditioner is `M⁻¹`):

    θ̇ = −M⁻¹ ∇L(θ)

Locally quadratic `L(θ) = ½ (θ−θ*)ᵀ H (θ−θ*)` (H = loss Hessian, SPD in the active subspace). The
dynamics are governed by the spectrum of the **preconditioned operator** `M⁻¹H`. Two regimes, and the
metric acts *differently* in each — this is the whole story:

### Regime A — finite condition number κ = λ_max/λ_min < ∞ (strongly-convex-ish basin) → GEOMETRIC decay; metric sets the CONSTANT
For μ-strongly-convex, L-smooth `L`, gradient descent contracts **linearly (geometrically)**:

    L(θ_t) − L* ≤ (1 − 1/κ_M)^t · (L(θ_0) − L*)      κ_M = cond(M⁻¹H)

Continuous flow: `L(t)−L* ≤ e^{−2t/κ_M}(L(0)−L*)`. **Key facts:**
- The FUNCTIONAL FORM is `e^{−ct}` / `ρ^t` for ANY fixed positive-definite metric M, INCLUDING the
  identity (plain GD), *provided* λ_min>0. There is **no polynomial exponent α** here.
- Preconditioning changes `κ_M` → changes the RATE CONSTANT `c = 2/κ_M` (equivalently `ln ρ`). It does
  **NOT** change a power-law exponent — because in this regime there is no power law.
- Exact Newton / exact natural gradient at a quadratic sets `M = H` → `M⁻¹H = I` → `κ_M = 1` → the
  fastest possible geometric rate (one-step in continuous time). Newton/NG is the κ-**optimum** of a
  regime that is already geometric.

**⇒ In Regime A, "engineering the metric" = engineering `κ_M` = changing the CONSTANT of a geometric
decay, never the exponent (there is no exponent to change).**

### Regime B — κ → ∞ / a CONTINUUM of scales (flat directions, power-law spectrum) → POWER-LAW decay; metric can change the EXPONENT
When the active spectrum is not bounded below — a genuinely flat direction, a saddle-to-saddle
crossing, or a **power-law eigenvalue spectrum** `{λ_i} ~ i^{−(1+2s)}` (the neural-scaling-law
mechanism) — the loss decays as a **power law**:

    L(t) ~ Σ_i (L_i) e^{−2λ_i t}   →   L(t) ~ C · t^{−α}    (α set by the spectral tail index s)

Here the metric acts on the exponent:
- A metric that **whitens** the spectrum (`λ_i → 1` in the active band) turns `t^{−α}` into `e^{−2t}`
  — the strongest possible exponent change (polynomial → geometric). This is exactly what a
  feature-space (NTK/Gram) preconditioner does to a power-law feature spectrum.
- A metric/dynamics that accelerates **saddle escape** (Stiefel/sphere orthogonalization) converts an
  **exponential** plateau-crossing time into a **polynomial** one — an exponent-regime win on the
  time-to-cross, not on the in-basin rate.

**⇒ In Regime B, "engineering the metric" CAN change the exponent — but only where the loss is
genuinely spectrum/saddle-limited, not noise-floor-limited.**

### What Muon actually changes (honest, per paper §5)
Muon = **momentum (Nesterov) + spectral-norm steepest descent** (Newton–Schulz orthogonalization →
the matrix-sign `UVᵀ`, equalizing the update's singular values to 1 = "spectral flattening",
Bernstein–Newhouse). Decompose the two parts against the two regimes:
- **The orthogonalization (the METRIC part):** it is steepest descent under the spectral norm in
  **weight** space, ≈ 1-step Shampoo ≈ Gauss–Newton (paper §5). It reduces the *effective weight-space
  condition number* of the update from κ≈19 toward ~1 → a **Regime-A constant (rate) win.**
- **The momentum (the ACCELERATION part):** momentum can lift `α` from 1→2 in a convex plateau — but
  **AdamW also has β₁ momentum**, so the *differential* −32% Muon-vs-AdamW is NOT the momentum, it is
  the orthogonalization. ⇒ **the measured −32% is a Regime-A CONSTANT (κ) win, attributable to the
  spectral-flattening metric, not an exponent change.** (Paper §5 registers exactly this: "−32% is
  real, attribution to natural-gradient-ness is conjectural / equally κ≈19 boundary-Hessian busting.")
- **The one exponent-adjacent Muon effect:** in the saddle-to-saddle tail, Muon≈Stiefel-flow
  (2605.13079) converts exponential leap-escape → polynomial (the #217 conjecture) — a Regime-B win on
  the plateau-crossing TIME. **Unmeasured for d_seg; conjectural.**

### What a TRUE natural gradient changes
`M = F` (Fisher). At a quadratic, `F⁻¹H → I` (for exp-family outputs `F ≈ H` = Gauss–Newton) → `κ→1`
(a Regime-A constant win, the κ-optimum) **plus** reparametrization invariance (the trajectory in
KL/output space is independent of the arbitrary weight parametrization). NG does **not** change the
power-law exponent of a genuinely flat KL direction — a flat direction in the Fisher metric is still
flat. So even a *true* NG is a CONSTANT lever in Regime A, and inert on the Regime-B floor.

---

## 2. The candidate preconditioners, ranked for OUR object

Object: MLX level-set witness (60–230K conv-INR), K=5 SDF head → argmax palette → `R`
(bicubic↑/uint8/bilinear↓) → **frozen** SegNet argmax = d_seg. Loss = τ-annealed CE/mirror-descent
through the frozen scorer. Boundary Hessian **κ≈19** (memos). Fisher = margin = caustic
(`1−Σp² = tr F`; annulus `½ sech²(m/2)`; Pearson 0.978 — paper §2).

| # | preconditioner | metric it imposes | changes EXPONENT or CONSTANT? | verdict for our regime |
|---|---|---|---|---|
| 1 | **Euclidean / AdamW** (diagonal 2nd-moment) | per-coordinate rescale; blind to cross-coordinate correlation | CONSTANT (modest) | baseline; diagonal preconditioner *collapses* on the CORRELATED boundary Hessian (memo) — leaves κ≈19 largely intact |
| 2 | **Muon spectral (weight-space, BUILT)** | steepest descent under spectral norm = matrix-sign; singular-values→1 | CONSTANT (busts κ≈19→~1 in weight-spectral sense); *possibly* EXPONENT in the saddle-tail (conjectural) | **the measured winner (−32%)**; the cheap consistent approximation to GN/Fisher-in-weight-space |
| 3 | **Fisher-Rao NG (output-space)** | `F⁻¹` in categorical output geometry | CONSTANT (κ→1) + reparam-invariance | **FALSE FRIEND — redundant**: softmax+CE gradient `(p−y)` already IS the NG-simplified update (paper §5). No new win. |
| 4 | **K-FAC / Shampoo (approx NG, weight-space)** | Kronecker/GGᵀ-root block curvature | CONSTANT (κ→1, more exact than Muon) | dominated: **Muon ≈ 1-step Shampoo ≈ GN** (paper §5); the extra accumulation is a SCALE-play a small conv-INR does not need (muon memo) |
| 5 | **Gauss–Newton** (`M = JᵀJ` through frozen SegNet∘R) | the pullback of the output Fisher to weight space | CONSTANT (κ→1); it IS "NG-in-weight-space" | the *conceptual* ideal metric, but a full JVP/VJP-through-scorer solve is expensive; **Muon is its 1-step approximation** at a fraction of the cost |

**Geometry-optimal choice for the frozen-scorer level-set flow: Muon spectral (weight-space).**
Justification, three legs:
1. **It targets the right anisotropy.** The κ≈19 conditioning is a *weight-space, cross-coordinate,
   correlated* Hessian (the boundary geometry couples the hidden-layer weights). AdamW's diagonal
   preconditioner cannot see it; Muon's spectral flattening busts it directly (Regime-A constant win)
   — the measured −32% confirms the mechanism operates.
2. **It is the cheap consistent approximation to the ideal metric.** Muon ≈ 1-step Shampoo ≈
   Gauss–Newton = the output-Fisher pulled back to weight space (paper §5). K-FAC/Shampoo/GN would be
   "more exact" but buy little at 60–230K params and add scale-motivated machinery we don't need.
3. **The output-space metric is already spent.** For softmax+CE the `(p−y)` gradient IS the
   NG-simplified update; a bolt-on output-space `F⁻¹` is redundant (paper §5 disarmed it). The only
   output-space "metric" that matters — the τ-caustic Fisher blow-up on the boundary (`I_τ =
   τ⁻²σ(1−σ)`, = the margin field) — is ALREADY consumed as the `margin_saliency` (#141) loss weight
   and the Laguerre/additive-margin head levers (`laguerre_logit_offset.py`). So there is **no
   additional output-space win to capture** — the whole remaining metric win lives in **weight/feature
   space**, and Muon is the built, measured, geometry-appropriate weight-space metric there.

**Honest refinement (the load-bearing caveat):** Muon's win is a Regime-A **constant** win (busting the
finite κ of the boundary basin). It does **not** change the d_seg-vs-compute **exponent**. The residual
exponent headroom lives in **Regime B** — the flat lane-dash tail — and needs a *different* mechanism
(feature-space whitening / saddle-escape), NOT a different global optimizer.

---

## 3. The false-friend honesty — where the REAL metric win is

Per paper §5 + the identity analysis, the metric win by space:
- **Output-space (categorical Fisher-Rao): NO new win.** `(p−y)` already = NG-simplified. A full
  `F⁻¹` bolt-on is a false friend (redundant). The τ-caustic on the boundary = the margin field is
  already a loss weight (#141) / head lever. **Do NOT build output-space NG** (paper §5, reconfirmed).
- **Weight-space (Muon / Shampoo / GN): the win is HERE, and Muon largely captures the CONSTANT part.**
  The −32% lives here. Successors (Shampoo/K-FAC/SOAP/Dion/…) are scale-plays — no exponent left to
  capture that Muon's spectral flattening didn't already take in Regime A.
- **Feature-space (NTK / Fourier-feature Gram): the UNCLAIMED win, and the ONLY place an EXPONENT lever
  exists for us.** The fine-scale erasure (lane dashes) is the **NTK spectral bias** — a power-law
  feature-Gram spectrum, high frequencies at small eigenvalues learned last (Regime B). Whitening the
  *resolvable* directional Fourier band per-scale is natural gradient in **feature space** — a distinct
  metric from the redundant output-space Fisher, and the geometry-optimal **exponent** lever. Sisters
  `project_sig_proc_filter_chain_measured_R_allpass_L3_ntk` ("L3-NTK-bandpass★ > L4-matched") and
  `lane_dash_residual_root_is_along_tangent_freq_deficit` ("along-tangent freq deficit 3.2×") are the
  measured signature that this tail IS spectrum-limited — a convergence, not a fabrication.

There is no separate "boundary-Hessian preconditioner" beyond what Muon (weight-space κ) + per-scale
whitening (feature-space spectrum) jointly give.

**Verdict: the real metric win is WEIGHT-space (Muon, constant, BUILT/measured) + FEATURE-space
(NTK whitening, exponent, unbuilt); OUTPUT-space is redundant.**

---

## 4. The BIGGEST honest caveat — the flicker floor caps the whole facet

`#205` converged to d_seg ≈ 0.004964 ≈ the popout/flicker floor ≈ 0.00520 (memory
`witness_converged_to_flicker_floor`). If the residual is the **STE/SDE irreducible-noise floor**
(Clarke-subgradient through the set-valued argmax — memo), then it is NOT curvature and **NO metric
(constant OR exponent) lowers it** — the lever moves to REPRESENTATION (a store / the analytic lane
band) or REGULARIZATION (flicker down-weight). So the scaling-law (rate/exponent) framing governs the
**descent phase before the floor**; metric-engineering (a) reaches the floor faster (constant, Muon)
and (b) lowers the floor **only if** the floor is set by unconverged flat directions (Regime-B
saddle/spectrum tail), not by STE noise. **The $0 pre-metric for every exponent lever below must FIRST
separate saddle/spectrum-limited (addressable) from flicker-floor-limited (not).**

---

## 5. Built surface (grepped — no invented flags)

`experiments/train_levelset_witness_realized_through_R_mlx.py` (launch path) + base
`experiments/train_witness_realized_through_R_mlx.py`:
- **AdamW** (`--lr`, `--lr-end`, `--weight-decay`, `--adam-beta2`, `--lr-schedule`) — Euclidean default.
- **Muon finisher** (`--muon-start-epoch` [None=AdamW throughout=bit-identical], `--muon-lr`,
  `--muon-adamw-lr`, `--muon-momentum`, `--muon-weight-decay`, `--muon-ns-steps`) via
  `tac.optimization.muon_finisher_mlx` (MLX `Muon`, aspect `max(1,rows/cols)**0.5`; head/code/biases →
  AdamW). Schedule levers BUILT default-off: **`--muon-lr-final-frac`** (cosine-decay the flat Muon LR
  — river-valley fix, GAP 1) + **`--muon-warm-start-momentum`** (seed Muon buffer from AdamW
  first-moment — transition-thrash fix, GAP 2).
- **MD-decoupling** (`--optimizer md`, `--md-base {adam,muon}`) — magnitude/direction reparam
  (`tac.optimization.md_decoupling`), BUILT.
- **Directional Fourier bank** (`--n-dir-freqs`, `--freq-across`; Nyquist cap
  `freq_across·2^(n_dir_freqs−1) ≤ 64`) + `--self-orient` (= discrete shearlet, paper §1).
- **Stage transitions** (`--stage-transition-rewarmup-epochs`, `--stage-transition-reset-moments`).
- **NOT wired to the witness trainer:** the torch `tac.optimization.iglt.IGLTOptimizer`
  (Fisher-diagonal / block / "kfac" + `inverse_sqrt|inverse` power, Martens–Grosse 2015) exists but is
  torch-only and lives elsewhere — an output/weight-space Fisher preconditioner is **not** in the MLX
  launch path (and per §3 an output-space one would be redundant anyway). There is **no** `--whiten` /
  feature-Gram-precondition flag — Lever A below is a BUILD, not an existing flag.

---

## 6. My ONE contributed claim to the 4-facet synthesis

> **CLAIM (metric → scaling-law regime split):** For the frozen-scorer level-set witness the training
> metric changes the **CONSTANT** of a GEOMETRIC decay in the finite-κ (κ≈19) boundary basin and can
> change the **EXPONENT** of a power-law decay ONLY in the flat NTK-spectral-bias tail; our BUILT lever
> **Muon** is a boundary-basin **constant (κ-busting)** win (which is why the −32% is a rate win, not
> an exponent win — the differential vs AdamW isolates the spectral-flattening metric from the shared
> momentum), and the only **exponent** lever available to us is a **feature-space (NTK/Fourier-Gram)
> whitening** preconditioner (natural gradient in FEATURE space) — distinct from the REDUNDANT
> output-space Fisher (`(p−y)` already = NG-simplified) — and it moves the exponent only if the tail is
> spectrum/saddle-limited rather than STE-flicker-floor-limited.

**Tag:** the regime split + κ→constant / spectrum→exponent theory is **DERIVED** (standard
convex-optimization + NTK/scaling-law theory). The −32% Muon constant win is **MEASURED** (cited from
`muon_vs_adamw_from_stage4_convergence_arm_20260622`, not re-run). The identification of the lane-dash
tail as spectrum-limited (vs flicker-floor) is **CONJECTURE** pending the $0 pre-metric (§7 Lever A/C).

---

## 7. Ranked levers, each with a $0 pre-metric (config-mostly; A/B-owed; net-S #205-gated)

Ranked for "steepen the d_seg-vs-compute scaling law." **CONTAINMENT: none auto-fires; operator GO
gates any dispatch; #205 untouched.**

**Lever A — Per-scale NTK / feature-Gram WHITENING of the directional Fourier bank [EXPONENT lever;
highest theoretical ceiling; needs a BUILD].** Rescale each directional-frequency channel's amplitude
by `∝ 1/√λ_scale` so the feature-Gram spectrum is flat across the *resolvable* band. This is natural
gradient in FEATURE space; it turns the power-law loss tail into geometric decay (Regime B → A) —
the strongest "change the exponent" move for our object, and it converges with the deep-math pass §7.2
"M2 NTK/multiscale band-pass whitening (~3–10× SPEED)" and the measured along-tangent 3.2× deficit.
Scope to the **resolvable** band (`--n-dir-freqs 2→4`, `--freq-across 8`, Nyquist cap) — NOT sub-2px
basis refinement (paper §5 disarmed that: below stem Nyquist needs a *store*, not a finer chart).
- **$0 pre-metric:** on a cached n96/n600 φ-field batch, compute per-scale gradient-energy (or the
  directional Fourier-feature Gram diagonal); (1) verify it is **power-law** across scales (log-log
  slope < 0 ⇒ Regime B ⇒ whitening changes the exponent; flat ⇒ already whitened ⇒ null); (2) confirm
  the low-eigenvalue scales coincide with the surviving lane-dash flip mass (margin-saliency #141
  overlap). Pure-numpy read, no training. Positive pre-metric ⇒ the exponent is addressable.
- Requires a new per-scale amplitude lever (name TBD *after* grepping — do NOT invent a flag);
  `--n-dir-freqs`/`--freq-across` are the adjacent basis controls it would sit beside.

**Lever B — Muon finishing-stage SCHEDULE [CONSTANT lever; BUILT; highest confidence].** Turn on
`--muon-lr-final-frac` (cosine-decay the flat Muon LR to a low floor — the Newton–Schulz normalization
cannot self-reduce the step near the minimum, river-valley 2606.21514) + `--muon-warm-start-momentum`
(seed the fresh Muon buffer from the outgoing AdamW first-moment — kills the cold-start unit-norm
boundary thrash / d_seg spike). Both settle the Regime-A geometric descent and remove the transition
transient; **does NOT change the exponent** (constant win) but is the safest, already-measured-adjacent
lever (muon memo EV #1+#2).
- **$0 pre-metric:** this IS #270's live warm@726 A/B — read the two forks' d_seg-vs-epoch curves at
  the checkpoints; a widening gap post-switch = the schedule lever pays. Already scheduled; no new run.

**Lever C — #217 post-Muon leap-residual reheat micro-stage [EXPONENT lever in the saddle-tail;
conditional].** Margin-reweight the surviving lowest-persistence lane-dash pixels with Muon's
Stiefel/sphere orthogonalization ON (+ optional log-decay SGLD). Theory (2605.13079, Muon≈Stiefel):
converts EXPONENTIAL saddle-escape → POLYNOMIAL on the plateau-crossing time — a genuine Regime-B
exponent effect on the "d_seg down but very slowly" tail (muon memo EV #3, single highest-EV finishing
lever).
- **$0 pre-metric (the decisive one — §4):** at the current #205 checkpoint, histogram the per-pixel
  persistence/margin of the surviving flips (margin-saliency #141 already produces this). If a
  substantial mass sits in a **low-persistence saddle tail** ⇒ exponent-addressable ⇒ build the reheat.
  If the residual is dominated by the **STE flicker floor** (temporal flip variance, near-zero
  persistence, spatially incoherent) ⇒ NO metric helps ⇒ route to representation/regularization, NOT
  this lever. This pre-metric gates Lever A too.

**Lever D — MD-decoupling base (`--optimizer md --md-base muon`) [CONSTANT lever; BUILT; speculative
at our scale].** The magnitude/direction product-manifold metric regulates the relative update from
step one (fixes early-collapse / no-LR-transfer-across-width / warmup dependence; 2606.25971),
composes with Muon. A different Regime-A constant lever.
- **$0 pre-metric:** on a short cached run, measure the step-1 relative-update norm `‖ΔW‖/‖W‖` per
  layer — a spike (collapse) ⇒ MD indicated; flat ⇒ MD null at our scale (muon memo: "SPECULATIVE at
  our scale"). Lowest rank.

**Ranking for the operator's "change the scaling law" frame:** A (exponent, principled, build) ≳
C (exponent, tail, gated by the flicker-vs-saddle pre-metric) > B (constant, built, highest-confidence,
DO FIRST as the safe win) > D (constant, built, speculative). **Do the Lever-A/C shared $0 pre-metric
(persistence-tail vs flicker-floor + per-scale Gram power-law) FIRST — it decides whether ANY exponent
lever is live before a single epoch is spent.**

---

## 8. Honest verdict — does metric-engineering change the exponent or the constant?

**BOTH are possible, but they are DIFFERENT levers on DIFFERENT parts of the curve, and everything we
have BUILT changes the CONSTANT, not the exponent.** Muon's −32% is a rate-constant win in the finite-κ
boundary basin (the differential-vs-AdamW isolates the spectral-flattening metric from shared momentum,
so it is unambiguously a κ/constant effect, not acceleration). To change the **exponent** you must
attack the spectrum-power-law regime — the NTK/feature-Gram spectral-bias tail — with a feature-space
whitening preconditioner that is **not yet built**, and it only helps if the tail is
spectrum/saddle-limited rather than STE-flicker-floor-limited (the $0 pre-metric decides). Output-space
Fisher-Rao natural gradient is a false friend (redundant with the `(p−y)` gradient) and must not be
built.

**Sisters:** `deepmath_amortizing_argmax_paper_draft_20260704` §5 · `muon_deep_dive_keep_and_tune_
finishing_stage_schedule_not_switch_20260703` · `project_sig_proc_filter_chain_measured_R_allpass_L3_
ntk_20260701` · `lane_dash_residual_root_is_along_tangent_freq_deficit_R_allpass_20260703` ·
`witness_converged_to_flicker_floor_leverD_is_path_below`. Facets 2–4 of this pass compose here: this
facet supplies the **metric** term of the geometry-optimal scaling law. ALL MEANS; pointer 0.19110
UNMOVED.
