# Mallat + Ballé — targeted second-pass review against the MEASURED deep math (2026-07-07)

**Agent:** deep-math research subagent (report-only; no launches, no trainer edits). Task: operator
directive *"review Mallat and Ballé work given our latest understanding and measurement of the deep
math and frozen contest information space."* This is NOT a survey (those exist: #150-152, #305, the
2026-05-08 Mallat-wavelet and 2026-05-07 Ballé-hyperprior empirical passes). It is a per-theorem
adjudication against the anchors measured since.

**Operator mid-task correction (binding, integrated throughout §B):** the scorer chain IS
differentiable end-to-end (bicubic↑ → uint8-STE → bilinear↓ → SegNet/PoseNet via
`tac.differentiable_eval_roundtrip` + `load_differentiable_scorers`; d_pose is plain MSE). The ONLY
non-differentiable point is the final argmax READOUT in d_seg, and we train through the softmax-τ
relaxation with a MEASURED relaxed↔exact theory (Maslov error ∈ [0, τ·ln5]; τ-crossover
homogenization law; margin=Fisher 0.978). So the Ballé adjudication below is
**relaxation-vs-relaxation** (his quantizer relaxation, our argmax-τ relaxation — the same
mathematical move on different discrete maps), never "his variational machinery breaks on
non-differentiability."

**Label discipline:** MEASURED (ours, anchor cited) · THEIRS (paper + result cited) · INFERRED ·
SPECULATIVE. Attribution hierarchy per `[[uniward_attribution_honest_lineage_vcm_at_heart]]`:
VCM/indirect-RD is the heart; our separatrix/task-space level-set math is the star; external results
are adjudicated for what they actually state.

**Authority:** $0, advisory, MEANS. Pointer 0.19110 UNMOVED (a review moves no pointer).

---

## The measured anchors reviewed against (verified in-tree before use)

| Anchor | Status | Source |
|---|---|---|
| d_seg lives on codim-1 argmax boundary; margin=Fisher Pearson 0.978 (exact identity: 1−Σp² = tr categorical Fisher) | MEASURED | `fisher_curvature_equals_categorical_fisher_trace_caustic_v1`; #284 Ch.2/3 |
| All-class directional (anisotropic) basis −48% d_seg, ~0 bytes | MEASURED | `curvelet_directional_basis_dseg_reduction_v1`; CLAUDE.md capstone |
| Along-tangent frequency deficit 3.2× (dashes ~25 cyc/unit; basis freq_along ≤8; freq_across → 32,64) | MEASURED | `[[lane_dash_residual_root_is_along_tangent_freq_deficit_R_allpass]]` |
| R operator ALL-PASS to ~2px (0.997 at dash scale) — deficit is representational, not filter-imposed | MEASURED | same memo (grating-sweep MTF) |
| Homogenization law: dashes unrecoverable below τ-crossover at ANY capacity; #287 max-plus comb = unique repair class; comb removes 86% of solid-band dash-gap FP at frozen ep650, but render-composite net-negative → corrector must be IN-TRAINING | MEASURED (law registered; corrector efficacy in-training still A/B-owed) | `dash_erasure_homogenization_v1`; DAG FEED-08c |
| Curriculum = coarse-to-fine = annealing, per-stage measured; τ=ε=ħ Maslov reading; MCF erases thin-lane | MEASURED / PROVEN-limit | `maslov_dequantization_bound_v1`, `tau_eps_hbar_one_dequantization_two_scales_v1`; #284 Ch.4/6 |
| Quadratic basin confirmed near θ* (LM ratios 0.847/0.868); subset-solve transfer law reproduces +5.1% | MEASURED | `quadratic_head_chart_subset_solve_gap_v1`; FEED-08d |
| Self-orient basis IS a discrete shearlet; Fisher-weighted shearlet N-term count is a PROVEN upper bound on task rate R_X(D_Y) | PROVEN (tightness conjectured) | `shearlet_nterm_upper_bounds_task_rate_v1`; #284 Ch.5 |
| Ballé-style weight-entropy penalty λ50: −1.55 bits/wt order-0 H, **−19.6% archive bytes through brotli** | MEASURED | `weight_entropy_penalty_balle_adversarial_review_byteclose_20260620.md` |
| Hyperprior CANNOT reconstruct near-iid quantized weight symbols (no 2D locality); 8 N/M configs, rel_err 0.98-0.99 plateau | MEASURED (falsified, implementation-class exhausted) | `hyperprior_architecture_cannot_reconstruct_near_iid_quantized_symbols_no_2d_locality_v1` |
| ξ residual coder: delta-vs-predictor 2714 B beats 3200 B table; spline predictor DEAD | MEASURED | DAG FEED-08b |
| Frozen info space: SegNet EffNet-B2 U-Net stride-2 stem argmax-only; PoseNet FastViT-T12 6-dim MSE; R; 600 pairs ONE clip; rate = archive.zip bytes only (rule 118); S_floor≈0.118 rate-dominated; intrinsic dim ~8 | MEASURED/FROZEN | CLAUDE.md; `[[project_contest_is_indirect_rate_distortion_task_space_coding]]` |

---

## A. MALLAT — the representation leg

### A1. Second-order scattering ⇒ the dash residual is a MODULATION invariant; the comb IS the second-order term

**THEIRS:** Andén & Mallat, *Deep Scattering Spectrum* (IEEE TSP 62(16):4114-4128, 2014;
arXiv:1304.6763): second-order scattering coefficients S₂(j₁,j₂) = ⟨||x⋆ψ_{j₁}|⋆ψ_{j₂}|⟩ capture
**amplitude-modulation spectra of the envelope** of the first-order response — information to which
first-order (mel/first-layer oriented filter) coefficients are provably blind after averaging.
Bruna & Mallat, *Invariant Scattering Convolution Networks* (PAMI 2013): same algebra in 2-D with
oriented wavelets — S₂ at (j₂,ℓ₂) applied ALONG the ridge of a (j₁,ℓ₁) edge response encodes
periodic on/off structure along that edge.

**Mapping to MEASURED:** the lane dash is exactly an amplitude modulation (~25 cyc/unit on/off
envelope) of an oriented edge carrier (the solid lane band). Our directional Fourier/self-orient
basis is a FIRST-order oriented representation: it carries the edge carrier, and the measured 3.2×
along-tangent deficit is precisely the first-order blindness the scattering algebra names.
**INFERRED (sharp):** the scattering fix is not "more first-order along-frequency" — it is ONE
second-order factor: carrier × envelope, i.e. |edge response| re-analyzed along-tangent at the dash
frequency. The **#287 max-plus dash-comb is structurally this second-order term** in tropical form
(max-plus gating of the solid band by a periodic envelope with phase = ego-ξ): O(1) parameters per
lane (period, phase, duty) instead of O(width × 25) first-order along-tangent modes. The scattering
algebra therefore RANKS the two live repair candidates: comb (second-order, parameter-cheap,
phase-predictable from ξ) ≻ freq-along ladder (first-order, parameter-expensive, and
homogenization-blocked below the τ-crossover per `dash_erasure_homogenization_v1`).

**Verdict: NOW** — this is a theory adjudication FOR the in-training comb A/B that FEED-08c already
owes, and AGAINST spending the next run's basis budget chasing along-frequency for the lane class.
No new probe needed beyond the owed one (in-training comb, n600 through-R).

### A2. Parabolic scaling explains the MEASURED 8 cyc/unit along-ceiling — the 3.2× deficit is a theorem consequence, not a tuning accident

**THEIRS:** Candès-Donoho curvelet parabolic scaling (width ∝ length²; equivalently along-ridge
bandwidth ∝ √(across-ridge bandwidth)), the scaling that makes curvelets/shearlets N-term optimal
O(N⁻²(log N)³) for **cartoon functions** — piecewise-C² with C² edges (Mallat, *A Wavelet Tour*,
3rd ed., ch. on geometric representations, curates this; our #284 Ch.5 already used it).

**INFERRED (arithmetically exact, identification needs the caveat below):** our basis runs
freq_across up to 64 cyc/unit; parabolic scaling gives along-bandwidth ≈ √64 = **8 cyc/unit — the
exactly-measured freq_along ceiling**. Dashes need ~25. 25/8 ≈ 3.1 ≈ the MEASURED 3.2× deficit.
So the deficit is what parabolic scaling PREDICTS for any curvelet/shearlet-class frame at our
across-Nyquist: a dashed edge is NOT a cartoon function (it is edge × periodic indicator — an
oriented-oscillation class; the natively-matched frames are wave atoms, Demanet-Ying 2007, whose
scaling law is along ≈ across, NOT parabolic). **Consequence: the "derived parabolic rebalance"
(along ∝ √across) in `[[feedback_basis_match_rule118_are_the_levers]]` is the correct chart for the
SOLID all-class edges (keep it — it is the −48% lever's home), and is theorem-limited for the lane
dashes** — within a parabolic frame no rebalance reaches 25 cyc/unit along without breaking the
scaling that makes the frame optimal for everything else. The lane class must exit the frame:
either wave-atom-scaled along-bandwidth for that class only, or (cheaper, per A1) the comb.
**Caveat:** the 8 = √64 coincidence is exact arithmetic on our config, but our freq grid was chosen
by config, not derived from parabolic scaling — treat the identification as INFERRED until the
ladder probe below runs.

**Verdict: NOW (interpretive law + $0 probe).** Probe: frozen-checkpoint render probe (FEED-08c
apparatus, ep650 ckpt, n600 through-R, $0 CPU): lane-class recall-err vs freq_along ∈ {8, 16, 25,
32} at fixed freq_across=64. Discriminates three hypotheses: (i) deficit closes ∝ freq_along
(first-order fix viable, parabolic reading wrong), (ii) closes only at ≥25 (parabolic reading
right, wave-atom class), (iii) does not close even at 32 (homogenization-blocked in-render →
in-training comb is the only repair, strongest form of the registered law).
**Canonical-equation CANDIDATE (FORMALIZATION_PENDING — do NOT register unmeasured):**
`parabolic_scaling_along_tangent_ceiling_v1`: for a parabolic-scaled directional frame with
across-Nyquist W, representable along-tangent modulation ≤ √W cyc/unit; class-specific residual for
features modulated above √W. Measuring probe: the freq_along ladder above.

### A3. Bandlets vs curvelets — our self-orient basis is BANDLET-class, and rule-118 makes the bandlet's one weakness free

**THEIRS:** Le Pennec & Mallat, *Sparse Geometric Image Representations with Bandelets* (IEEE TIP
2005) + Mallat & Peyré orthogonal bandlets (SIAM MMS 2008): bandlet bases ADAPT to an estimated
geometric flow and achieve O(M⁻^α) M-term error for C^α-geometrically-regular images — beating
curvelets for α > 2 — where **M counts approximation coefficients PLUS the bits describing the
adapted basis (the flow)**. The classical objection to bandlets vs fixed-frame curvelets is exactly
that flow-description cost + estimation fragility.

**Adjudication (task asked: bandlets vs curvelets for self-orient):** our `--self-orient` basis is
adaptive-orientation (oriented by the local boundary tangent field) → it is **bandlet-class, not
fixed-frame curvelet-class**; #284 Ch.5's "discrete shearlet" naming is right about the parabolic
scaling but the ADAPTIVITY is bandlet lineage. The decisive point for us: **rule 118 zeroes the
bandlet's flow cost.** The lane geometric flow comes from the openpilot lane polynomial — a
deterministic generator that ships as FREE inflate.py code — so the "M includes the flow bits" tax
in Le Pennec-Mallat's theorem does not bind our counted rate. Bandlet-with-free-flow is strictly
the right formal home for the witness basis, and the O(M⁻^α), α>2 rate (vs curvelet's α=2 cap) is
the correct citation for why adaptive orientation beats a fixed directional frame at equal counted
bytes. **HONEST BOUNDARY:** their theorems are for L² approximation of the image function; our
target is Hamming mismatch of an argmax labeling through a frozen detector — the rates transfer as
upper-bound INTUITION only (the Fisher-weighted N-term bound `shearlet_nterm_upper_bounds_task_rate_v1`
is OUR statement of the task-space version, and remains the load-bearing one).

**Verdict: WATCH (paper/framing + attribution).** No new lever — the lever (self-orient + free
analytic flow) is built and measured (−48%). This entry fixes the citation and the originality
accounting: adaptive-flow directional approximation = Le Pennec-Mallat; free-flow-under-rule-118 +
Fisher-weighted task-space N-term = OURS.

### A4. Wavelet Conditional Renormalization Group vs our τ-crossover — it RHYMES, it is not the same theorem

**THEIRS:** Marchand, Ozawa, Biroli, Mallat, *Wavelet Conditional Renormalization Group*
(arXiv:2207.04941; published as *Multiscale Data-Driven Energy Estimation and Generation*, PRX 13,
041038, 2023): factorize p(x) scale-by-scale as p(coarse)·Π_j p(fine_j | coarse_j) with conditional
energies in a wavelet basis; estimation and sampling proceed coarse-to-fine; RG language is about
the conditional-interaction structure across scales (verified for Gaussian and φ⁴ theories, i.e.
they DO analyze an actual RG fixed point neighborhood for φ⁴).

**Adjudication (task asked: is our τ-crossover a known RG fixed-point statement?):** **No — it only
rhymes, and I say so.** Our τ-crossover (`dash_erasure_homogenization_v1`) is a
Γ-convergence/homogenization statement: when min(τ-scale, viscous ε, R-Nyquist) ≳ δ_along the flow
lands on the homogenized solid band with pinned lane interface. That is periodic-microstructure
homogenization (Γ-limit), not an RG fixed-point theorem; WCRG contains no theorem of this shape.
The genuine rhyme: (a) coarse-to-fine conditional factorization = our measured curriculum order;
(b) WCRG's "fine DOF conditioned on coarse" is structurally the comb-as-corrector — model the dash
(fine, conditionally-simple given the band) ON TOP of the solid band (coarse) instead of asking one
chart to carry both. This is CONSISTENT with A1/A2 but adds no mechanism we don't already have.

**Verdict: NO-as-lever, with-reason (recorded negative).** Keep as a supporting citation for the
coarse-to-fine curriculum's factorization structure in the paper; do not cite it as the source of
the τ-crossover law (that is OURS, homogenization-shaped).

### A5. Group-invariant scattering's diffeomorphism-stability bound → a discriminating $0 probe on flip reachability

**THEIRS:** Mallat, *Group Invariant Scattering* (CPAM 65(10):1331-1398, 2012): the scattering
metric is Lipschitz-continuous to diffeomorphisms — for x deformed by τ, ‖S x_τ − S x‖ ≤
C·(2⁻ᴶ‖τ‖_∞ + ‖∇τ‖_∞ + ‖Hτ‖_∞)·‖x‖ (translation term vanishes as J→∞; the load-bearing term is
the deformation-gradient ‖∇τ‖_∞). Mallat's standing argument is that deep convnets inherit
approximately this stability — SegNet's stride-2 stem + deep conv cascade is in scope as an
approximate scattering architecture (INFERRED, his own heuristic, not a theorem about EffNet-B2).

**Mapping to MEASURED:** we have measured margin=Fisher as the flip geometry and built the
through-R flip-reachability surface S_R (#268, owed A/B per
`[[msal_uni_texture_proxy_inert_build_exact_sR_reachability_weight]]`). The scattering bound makes
a PREDICTION those measurements have not yet separated: at equal input-L², **smooth diffeomorphic
perturbations (boundary translation, ‖∇τ‖ small) should flip argmax near the separatrix far more
efficiently than high-frequency additive perturbations** (which the cascade attenuates — the same
stability that makes the detector robust makes its flips deformation-dominated). This speaks
directly to the flicker decomposition (predictable ego-jitter = diffeomorphic = ξ-replicable vs
irreducible sensor noise): deformation-stability predicts the PREDICTABLE component dominates flip
production per unit energy.

**Verdict: WATCH with a named $0 n600 probe:** equal-L² warp-vs-noise flip-rate through R+SegNet on
the cached n600 GT (apparatus = the FEED-08c render-probe path + the R MTF grating harness):
(i) sub-pixel diffeomorphic warp fields (amplitude-matched), (ii) additive band-limited noise.
Output: flips per unit L² by margin bin. If deformation dominates as predicted, it upgrades the
S_R weighting design (#268) — reachability should be parameterized by local deformation
sensitivity, not additive-perturbation sensitivity — and quantitatively grounds the
replicate-vs-downweight split in the flicker lever.

---

## B. BALLÉ — the rate leg

### B1. NTC ↔ rule-118: his encode/decode asymmetry is our compile discipline — and rule 118 INVERTS his model-size economics

**THEIRS:** Ballé, Chou, Minnen, Singh, Johnston, Agustsson, Hwang, Toderici, *Nonlinear Transform
Coding* (IEEE JSTSP 15(2):339-353, 2021; arXiv:2007.03034): learn analysis g_a and synthesis g_s;
transmit only the quantized latent ŷ under a learned entropy model; R-D optimize
E[−log p(ŷ)] + λ·D end-to-end. In deployment economics, g_s weights are amortized across a
DISTRIBUTION of sources (they ship once in the codec, not per image).

**Mapping (adjudicated):** rule 118 is the per-instance radicalization of exactly this split:
inflate.py = g_s = FREE code; archive payload = ŷ = the ONLY counted object; encode (g_a =
training) is unlimited. Our witness-compile discipline ("move maximal deterministic generic
structure into inflate.py; store only the irreducible video-derived statistic") IS NTC's
transmit-only-the-latent principle — with one inversion his framework doesn't natively contain:
for NTC the synthesis network's CAPACITY is a fixed sunk cost per codec, while for us **learned**
(video-derived) weights are counted but **generic/deterministic** synthesis structure is free. So
his R-D-optimal latent design advice transfers with a twist: the optimal move is not "small
decoder" (his regime) nor "big decoder" (naive rule-118 reading) but **maximally-generic decoder +
minimal video-derived sufficient statistic** — which for the ~8-dim lane orbit means the latent
should approach the intrinsic dimension with the generator carrying everything deterministic
(openpilot poly, comb phase from ξ, homography transport). This is already the campaign thesis
(THE FRONTIER section); Ballé's formalism is the correct external citation for it, and
"train-big-compress-small" (arm D) is his amortization economics applied to the counted-weights
residual.

**Verdict: NOW (framing + one config consequence).** Config consequence, already in the basis-match
memo's arm D and validated by the MEASURED −19.6%: carry the differentiable rate term on counted
weights (B2) in the next run. Originality accounting (NO-FAKE #7): the encode/decode asymmetry and
transmit-only-latent are THEIRS; the free-generic/counted-learned split under rule 118, the
per-instance (one clip, frozen detector) exact optimum, and the task-space latent (~8-dim orbit
statistic instead of a pixel latent) are OURS — see §C.

### B2. Rate-in-the-loss (differentiable entropy penalty) — ALREADY MEASURED WORKING at −19.6% archive bytes; keep it

**THEIRS:** Ballé et al. 2017/2018: replace the codelength with a differentiable density model of
the (noise-relaxed) latent and optimize R+λD jointly — rate must be in the training loss, not a
post-hoc quantizer.

**MEASURED (ours):** the Ballé-style weight-entropy penalty at λ50 lowered live-decoder order-0
entropy by −1.55 bits/wt and REAL archive bytes by **−19.6% through brotli** (byte-close proof,
2026-06-20, Catalog #304-compliant; note the honest correction in that memo — the deployed coder is
brotli-q11 on zigzag(int8), not the constriction range coder, so the order-0→bytes translation was
itself verified empirically, not assumed).

**Verdict: NOW (config lever, already built + measured).** Include `--weight-entropy-penalty-lambda`
in the next-run counted-weights arm (composes with train-big-compress-small: the penalty shapes
weights toward the coder DURING training; the post-pass compresses at Δd_seg=0 through R). The A/B
owed is net-S at n600 (does the −bytes hold at witness scale without d_seg cost) — that is the
standard duty-to-measure entry, not new research.

### B3. Entropy models at hundreds-of-bytes scale: the hyperprior overhead calculus INVERTS — factorized/static + predictive-context wins (and already has, measured)

**THEIRS:** Ballé, Minnen, Singh, Hwang, Johnston, *Variational Image Compression with a Scale
Hyperprior* (ICLR 2018; arXiv:1802.01436): transmit side-info z (a few % of total rate at IMAGE
scale, ~thousands of latent elements) so the entropy model adapts spatially; beats factorized
priors when the latent has spatial dependency structure and is large enough to amortize z.

**MEASURED (ours, two independent falsifications at our scale):** (i) full
CompressAI ScaleHyperprior/MeanScaleHyperprior across 8 N/M configs cannot even reconstruct PR101's
quantized weight symbols (rel_err plateau 0.98-0.99 — no 2D locality;
`hyperprior_architecture_cannot_reconstruct_near_iid_quantized_symbols_no_2d_locality_v1`);
(ii) our counted payloads are now HUNDREDS of bytes (ξ delta-residual 2714 B beating a 3200 B
table; lane coeffs similar scale). **INFERRED (quantified):** side-info + hyper-synthesis parameters
are video-derived → COUNTED under rule 118; even a minimal hyper-latent + its decoder costs
O(10²-10³) B fixed, i.e. ≥10-100% of the payload it would model — at image scale z is ~1-5% overhead,
at our scale it is order-100%: the calculus inverts, hyperprior is dominated at our byte scale
regardless of the locality failure. What DOES transfer from his entropy-model ladder is the
**backward-adaptive/context** end: decode-time context models cost ZERO side-info bytes and
decode-time compute is free (30-min budget) — and our measured ξ predictive-delta coder (FEED-08b)
is exactly a deterministic-predictor context model, already beating the static table.

**Verdict: NO for hyperprior (twice-measured + overhead-inverted); NOW-already-done for
predictive/context coding** (extend the FEED-08b pattern to the lane-coeff payload if it isn't
already delta-coded — a $0 byte-close check, not a run lever).

### B4. The two relaxations — his quantizer noise/STE vs our argmax-τ: same move, adjudicated both directions (per operator correction)

**THEIRS:** (i) Ballé et al. 2017 (arXiv:1611.01704): additive uniform noise ỹ = y + u as the
differentiable train-time surrogate for scalar quantization, giving a continuous density whose
differential entropy upper-bounds/approximates the discrete rate; (ii) NTC 2021 §quantization:
uniform-noise training is EXACTLY the rate of universal (dithered) quantization (Ziv 1985 /
Zamir-Feder) — with a shared dither at deploy, the train/test gap for the RATE term is ZERO;
(iii) Minnen & Ballé lineage "mixed" practice: noise-relax the rate path, hard-round + STE the
distortion path; (iv) Agustsson et al. 2017 soft-to-hard annealing: anneal a softmax-based
quantizer toward hard assignment during training.

**Ours (MEASURED):** softmax-τ relaxation of the argmax readout with exact deviation bound
[0, τ·ln5] (`maslov_dequantization_bound_v1`); measured per-stage τ-anneal effects (CE plateau
0.00498; tau asymptote 0.003348 ≈ measured best 0.003366); the τ-crossover/homogenization law
coupling the anneal floor to feature scale; uint8-STE inside R; PR95-lineage σ-noise schedule
(CLAUDE.md L17) — which IS his additive-noise relaxation of the uint8 channel, already in the stack.

**Adjudication, his → ours:**
- *Soft-to-hard annealing* = our τ-anneal, less the theory: he left the schedule empirical; we have
  the Maslov bound + the crossover law. **Nothing new transfers.**
- *Mixed relaxation* (hard where you can STE, soft where you need a density): we already run the
  mixture (uint8-STE hard-path; τ-soft argmax; σ-noise on the channel). The one untried permutation
  — hard-argmax + STE on the d_seg readout in a final stage — is structurally l7/L∞-sharpening
  territory, and l7 is a MEASURED DEFECT (CLAUDE.md capstone caveat). **NO-with-reason.**
- *Universal/dithered quantization* (the exact-relaxation trick, his sharpest result): does NOT
  transfer. UQ works because the encoder and decoder share the dither on the quantizer WE own. The
  argmax we must match is inside the FROZEN scorer — we cannot dither the detector, and dithering
  our render's logits does not commute through SegNet to a dither on ITS argmax. The relaxation gap
  at the argmax is therefore structural for us in a way it is not for him — which is precisely why
  we needed a measured crossover law instead of an exactness trick. **Recorded negative, with the
  reason.**
- The genuinely transferable check: his framework treats train-relaxation noise as part of the
  CHANNEL and asks the deploy channel to match. Our deploy channel (R's uint8) is matched by
  uint8-STE + σ-noise already; verify σ-schedule and STE are simultaneously active in the next-run
  config rather than one shadowing the other (a $0 config audit, DSL-side).

**Adjudication, ours → his (what our measurement sharpens that he left empirical):** the
τ-crossover/homogenization law is a statement his literature lacks: for structured discrete
readouts, annealing the relaxation below the scale δ of the finest task-relevant feature makes that
feature unrecoverable at ANY capacity unless a matched corrector (our comb) is active — i.e.
**anneal schedules have a feature-scale-coupled floor, not just a stability tradeoff**. In his
world the analogue would be: soft-to-hard annealing under a spatially-structured latent can
homogenize away fine latent structure the entropy model then never sees. To our knowledge
(INFERRED — I did not find this stated in the NTC/soft-to-hard line) this is unpublished; it is a
paper contribution of ours, not a run lever.

**Verdict: NO-new-lever; one $0 DSL config audit (σ-noise ∧ STE co-active); one paper-contribution
claim recorded.**

### B5. GDN — no reading survives scrutiny

**THEIRS:** Ballé, Laparra, Simoncelli, *Density Modeling of Images Using a Generalized
Normalization Transformation* (ICLR 2016; arXiv:1511.06281): GDN is derived to GAUSSIANIZE local
joint statistics of natural images — a density-factorization objective; its role in his codecs is
making the latent match the factorized entropy model at low per-channel capacity.

**Adjudication (task said: be skeptical):** the witness head serves a frozen detector, not a
density model; we have no factorized-prior latent whose marginal shape GDN would fix (our counted
payloads are coded by brotli/predictive-delta, shaped by the B2 entropy penalty); and the
modulation role GDN plays in synthesis transforms is covered by FiLM conditioning in our
architecture. The only conceivable reading — GDN as a divisive-normalization contrast conditioner
to stabilize SegNet responses — is SPECULATIVE, unsupported by any measured residual (the residual
is dash erasure + boundary jitter, neither contrast-shaped). **Verdict: NO-with-reason.**

### B6. Rate-distortion-perception / machine-perception line — the triangle collapses in our regime

**THEIRS:** the RDP tradeoff (Blau-Michaeli 2019; Ballé-adjacent through NTC's perceptual-metric
results and the Google sandwiched-codec line — honest attribution: "sandwiched" compression is
Guleryuz-Chou et al., with Chou the NTC co-author; it is not primarily Ballé's line) says R-D-P
form a three-way tradeoff when distortion and perceptual divergence are distinct constraints.

**Adjudication:** our D is ALREADY the machine metric (frozen-detector disagreement); there is no
separate perceptual constraint we owe (RDC-not-RDP, MEASURED framing in
`[[project_contest_is_indirect_rate_distortion_task_space_coding]]`: realism is a tax we don't
owe). The RDP triangle degenerates to a line; nothing binds. The sandwiched-codec architecture
(neural wrapper around a standard codec) is a vehicle-class we already dominate with the witness
(their standard-codec core would re-introduce the pixel-fidelity tax). **Verdict: NO/paper-framing
only.**

---

## C. INTERSECTION + ORIGINALITY ACCOUNTING (NO-FAKE #7)

**The task's intersection question — does Ballé-NTC-with-Mallat-scattering-prior (structured,
non-learned analysis) beat a learned latent in a single-clip, unlimited-encode, free-generator
regime?** Adjudicated: in OUR regime the question dissolves into the layered answer both reviews
converge on, and the convergence is the finding:

- The Mallat leg says: the analysis structure for the scored content is KNOWN (bandlet-class
  adaptive directional frame for solid edges; a single second-order modulation factor for dashes;
  flow and phase deterministic from openpilot/ξ) — so most of the "latent" need not be learned
  at all.
- The Ballé leg says: whatever IS transmitted should be the minimal statistic under a
  rate-in-the-loss objective with the synthesis structure free.
- Composed: **structured-analysis + tiny-learned-residual + free-generic-synthesis** — which is
  exactly the L0-L3 layered architecture already in the council draft (holographic section). The
  two literatures independently license the two halves; NEITHER is natively in the regime.

**Genuinely OURS (per-item):** (1) the task-space latent — the statistic sufficient for a FROZEN
detector's argmax partition, not for the pixels (indirect-RD per-instance, exact not ensemble);
(2) the Fisher/margin-weighted N-term bound on task rate (`shearlet_nterm_upper_bounds_task_rate_v1`)
— the task-space version of their L² approximation rates; (3) the rule-118 free-generic vs
counted-learned split and its inversion of NTC model-size economics (B1); (4) the
τ-crossover/homogenization law for discrete-readout relaxations (B4, A4 — not an RG theorem, not in
the soft-to-hard line); (5) the comb-as-tropical-second-order-term identification (A1) with phase
supplied by the pose statistic (ξ dual-use) — a cross-axis coupling neither literature has.
**Borrowed (credit precisely):** encode/decode asymmetry + transmit-only-latent + rate-in-loss
(Ballé); adaptive-flow directional approximation + modulation-capturing second-order algebra +
deformation stability (Mallat lineage); parabolic scaling optimality for cartoons (Candès-Donoho).

---

## D. RANKED DRAW-FROM TABLE

| # | Their result (exact) | Our measured anchor it touches | Concrete $0 n600 probe / config lever | Verdict |
|---|---|---|---|---|
| 1 | Andén-Mallat 2014 / Bruna-Mallat 2013: second-order scattering captures along-ridge amplitude modulation first-order features are blind to | 3.2× along-tangent deficit; dash_erasure_homogenization_v1; FEED-08c comb 86% mechanism | Theory-ranks the ALREADY-OWED in-training comb A/B (n600 through-R) ABOVE any freq-along spend for the lane class; comb = the second-order carrier×envelope term at O(1) params, phase from ξ | **NOW** (adjudicates existing owed A/B; no new build) |
| 2 | Candès-Donoho parabolic scaling (along-bandwidth ∝ √across; cartoon-optimal), curated in Mallat's Tour | freq_along ceiling 8 = √64 (measured config); 25/8 ≈ 3.1 ≈ measured 3.2× deficit | Frozen-ckpt render probe (FEED-08c apparatus, ep650, n600 through-R): lane recall-err vs freq_along ∈ {8,16,25,32} @ across=64 → discriminates first-order-fixable / wave-atom-class / homogenization-blocked. Candidate eq `parabolic_scaling_along_tangent_ceiling_v1` FORMALIZATION_PENDING on this probe | **NOW** ($0 probe named; sharpens a registered law) |
| 3 | Ballé et al. 2017/2018 rate-in-the-loss (differentiable entropy penalty on transmitted object) | MEASURED −19.6% archive bytes at λ50, byte-closed 2026-06-20 | Config lever: `--weight-entropy-penalty-lambda` in next-run counted-weights arm; composes with train-big-compress-small (arm D); duty-to-measure net-S n600 | **NOW** (built + byte-close-measured; needs the net-S A/B) |
| 4 | NTC 2021 (Ballé-Chou-Minnen et al.): transmit-only-latent, g_s free at decode, R+λD end-to-end | Rule-118 compile discipline; ~8-dim lane orbit; S_floor≈0.118 rate-dominated | No new run lever (it IS the campaign thesis); fixes the formal citation + the B1 inversion (generic-free vs learned-counted) for paper/originality | **NOW** (framing; originality accounting §C) |
| 5 | Mallat CPAM 2012 GIS: scattering metric Lipschitz to diffeomorphisms, ‖∇τ‖ term load-bearing | margin=Fisher 0.978; S_R #268 owed; flicker decomposition (predictable ego-jitter vs noise) | $0 n600 probe: equal-L² warp-vs-noise flip-rate through R+SegNet by margin bin (FEED-08c + R-MTF harness) → if deformation dominates, S_R weighting should be deformation-sensitivity-based; grounds replicate-vs-downweight split | **WATCH** (probe named, informs #268 design) |
| 6 | Le Pennec-Mallat bandlets: adaptive-flow bases achieve O(M⁻^α), α>2, where M includes flow bits | Self-orient −48%; analytic lane band 0.00087 (rule-118 free flow) | None new — adjudication: self-orient is bandlet-class with the flow-bits tax zeroed by rule 118; correct citation for the paper | **WATCH** (framing/attribution only) |
| 7 | Ballé-Minnen 2018 scale hyperprior: side-info entropy model beats factorized at image scale | Hyperprior falsified on our symbols (no 2D locality, 8 configs); payloads now O(10²-10³) B; ξ delta-coder beats table (FEED-08b) | $0 byte-close check: lane-coeff payload delta/context-coded like ξ (extend FEED-08b pattern); NEVER a hyperprior at this byte scale (side-info overhead order-100%, calculus inverted) | **NO** for hyperprior (twice-measured + derived inversion); trivial NOW for delta-coding check |
| 8 | Ballé 2017 uniform-noise relaxation; NTC universal/dithered-quantization exactness; Agustsson soft-to-hard anneal; Minnen mixed relaxation | τ=ε=ħ Maslov bound; measured τ-anneal law + crossover; uint8-STE; σ-noise L17 | No new lever: anneal=our τ (we have the theory he lacks); UQ does NOT transfer (cannot dither the frozen scorer's argmax — recorded negative with reason); mixed relaxation already instantiated. $0 DSL audit: σ-noise ∧ uint8-STE co-active in next-run config. Ours→his: crossover law = paper contribution | **NO-new-lever** (one $0 config audit; one paper claim) |
| 9 | Marchand-Ozawa-Biroli-Mallat WCRG (2207.04941 / PRX 2023): scale-by-scale conditional factorization p(fine\|coarse) | Coarse-to-fine curriculum measured per-stage; τ-crossover law | None — rhymes only: our crossover is Γ-convergence/homogenization, NOT an RG fixed-point theorem; WCRG's fine\|coarse = comb-as-corrector consistency (already have) | **NO-as-lever** (recorded; paper citation for curriculum factorization) |
| 10 | Ballé-Laparra-Simoncelli GDN (ICLR 2016): divisive normalization for natural-image density factorization | Witness head serves frozen detector; residual is dash-erasure + boundary jitter, not contrast-shaped | None survives scrutiny (FiLM covers modulation; no density objective; no factorized latent) | **NO-with-reason** |
| 11 | RDP (Blau-Michaeli; NTC perceptual results; sandwiched codecs = Guleryuz-Chou, honest attribution) | RDC-not-RDP measured framing; realism is a tax we don't owe | None — the triangle degenerates; sandwiched vehicle re-introduces the pixel tax | **NO**/paper-framing |

**Single highest-EV NOW: row 2** — the freq_along ladder render probe ($0, frozen ep650 ckpt, n600
through-R, existing FEED-08c apparatus). It is the one probe that simultaneously (a) tests the
parabolic-ceiling reading of the measured 3.2× deficit, (b) arbitrates comb-vs-basis spend for the
next run's lane arm (the basis-match memo's open ranking), and (c) supplies the measurement that
would promote `parabolic_scaling_along_tangent_ceiling_v1` from FORMALIZATION_PENDING.

## E. Canonical-equation candidates (FORMALIZATION_PENDING — none registered here)

1. `parabolic_scaling_along_tangent_ceiling_v1` — statement + probe in A2/row 2.
2. `payload_scale_entropy_model_inversion_v1` (SPECULATIVE→derivable): side-info entropy models
   (hyperprior class) are dominated when fixed side-info+decoder cost ≳ payload size; crossover
   payload ~O(10³-10⁴) B. Measuring probe: byte-close ladder of {static prior, predictive-delta,
   minimal hyper} on the actual counted payloads (ξ residual, lane coeffs) — partially measured
   already (FEED-08b + the 2026-05-07 falsification); registration needs the three-way A/B on ONE
   payload.

## F. What this review did NOT resolve (honest)

- Whether the in-training comb closes the dash residual at net-S — the owed FEED-08c A/B is still
  the arbiter; A1/A2 rank it, they don't prove it.
- The 8 = √64 identification is arithmetic on our config, not a derivation that our frame actually
  enforces parabolic scaling — the row-2 probe is what separates coincidence from law.
- Whether the σ-noise schedule and uint8-STE are both live in the current sealed config (flagged as
  a $0 DSL audit; not verified here — I did not open the trainer).
- Second-order scattering as a LOSS-side feature (scatter the render, penalize S₂ mismatch at dash
  scale) was considered and deliberately NOT tabled: it duplicates the comb's information at higher
  cost and adds a new loss surface mid-campaign; revisit only if the comb A/B fails (SPECULATIVE,
  recorded so it isn't re-derived).

Sources (papers verified this pass): arXiv:1304.6763 (Andén-Mallat DSS) · Bruna-Mallat PAMI 2013 ·
Mallat CPAM 2012 (di.ens.fr/~mallat/College/CPAM-Mallat-Scat.pdf) · Le Pennec-Mallat TIP 2005 +
Mallat-Peyré (di.ens.fr/~mallat/papiers/BandSiam.pdf) · arXiv:2207.04941 / PRX 13.041038 (WCRG) ·
arXiv:1611.01704 (Ballé 2017) · arXiv:1802.01436 (hyperprior) · arXiv:2007.03034 (NTC, JSTSP 2021,
DOI 10.1109/JSTSP.2020.3034501) · arXiv:1511.06281 (GDN).
