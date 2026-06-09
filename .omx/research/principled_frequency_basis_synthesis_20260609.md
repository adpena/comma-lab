# Principled frequency basis — replacing the hand-tuned SIREN w with derived/learned/scorer-matched spectra

UTC 2026-06-09 · claude · `[macOS-CPU advisory]` / design synthesis (mechanism, not score). Operator
question: "is there a more optimal and elegant and mathematical way to tune frequencies — Fourier, Maxwell,
Taylor, Euler, UNIWARD, and what else are we not considering?" Empirical trigger: F1 arm A (skip ON, w=30)
flat at 21.73 dB = the **w=30 alias trap** confirmed; arm B tests w=1. But w-as-a-single-scalar is the
crudest possible knob. This memo is the principled replacement + the wire-in to EXISTING primitives.

## The reframe (the one idea)

`sin(w·x)` with a single global `w` is a **0-parameter stand-in for a whole spectral geometry**. The
elegant, non-arbitrary replacement: **synthesize in a basis matched to the (scorer × source × motion)
spectral geometry, with the basis frequencies DERIVED or LEARNED — never hand-set.** Every framework below
is a different, rigorous way to specify "which frequencies, where, how many bits" — and the deepest one
unifies them: the optimal synthesis basis is the **right singular vectors of the frozen-scorer Jacobian**
(the task-optimal basis); Fourier/wavelet/Gabor are generic approximations, and UNIWARD is the
steganalysis-derived estimate of exactly that sensitivity.

## Organized by the math (each: mechanism · what it buys · do we have the primitive?)

### 0. Sampling theorem / Nyquist — the immediate, derived fix (what we were NOT considering)
`sin(30·feature)` at a block whose spatial sampling can't resolve 30 cycles is **literal aliasing** (Nyquist
violation) — the high-w carrier folds into incoherent noise, which gradient descent then suppresses back to
the DC/mean (the d_seg≈0.5 plateau). The principled ceiling: **each upsample block's max frequency ≤ the
Nyquist limit of that block's spatial resolution.** This gives a DERIVED per-scale frequency schedule
(coarse blocks → low w, fine blocks → high w) instead of a global constant. Cheapest principled win; kills
the alias trap by construction. We have NO primitive — this is a ~10-LOC per-scale schedule in the carrier.

### 1. Fourier features + NTK (Tancik 2020) — the spectrum, not the scalar
Replace the single `w` with a **distribution of frequencies** `γ(x)=[sin(2πBx),cos(2πBx)]`, `B∼N(0,σ²)`.
The bandwidth `σ` is the principled knob; the Neural Tangent Kernel analysis proves the FF mapping turns the
NTK into a **stationary kernel of tunable width** — i.e., you compose a kernel matched to the target's
frequency content, which is the rigorous spectral-bias cure. Set `σ` from the MEASURED power spectrum (next)
or learn it. We have adjacent surfaces (`contrib/calibrated_positional_encoding.py` = a fixed viewing-ray PE;
`ffnerv_as_renderer.py`; coin_pp INR) but NO learned-Fourier-feature carrier primitive.

### 2. Euler — sin/cos are Re/Im of e^{iωx}; make {ω_k} learnable
`e^{iωx}=cos(ωx)+i·sin(ωx)` unifies SIREN (sin), Fourier features, and the DCT (ff_nerv). The decoder is
synthesizing a Fourier series; "tuning frequency" = choosing the basis `{ω_k}` and amplitudes. The principled
form: **make `ω_k` LEARNABLE per-channel/per-block** (learned Fourier features, Li 2021), initialized from
the measured/scorer spectrum. Gradient descent finds the frequencies → strictly dominates the w=30/w=1
hand-tune. ~15-LOC change to the carrier (make `self.w` a learnable vector, not a float constant).

### 3. Heisenberg–Gabor uncertainty — boundaries are LOCAL, so the basis must be (what we were missing)
`Δx·Δω ≥ 1/2`: a global `sin(w·x)` is perfectly frequency-localized but spatially DELOCALIZED — yet SegNet
argmax disagreement lives at spatially-LOCAL class boundaries that need WIDE frequency support at SPECIFIC
locations. So the right basis is **jointly localized** (wavelet / Gabor / local feature grid), not global
sine. This is the rigorous reason SNeRV's DWT and HiNeRV's grid-PE are correct and global SIREN is not —
the literature already converged here; our carriers just didn't adopt it (grid-PE gated OFF; SNeRV's wins).
We HAVE: `wavelet_variance.py`, SNeRV's exact DWT, HiNeRV's `HierarchicalFeatureGrid` (gated off).

### 4. UNIWARD + scorer-Jacobian — the EVALUATOR-MATCHED allocation (deepest; we HAVE the primitives)
Where should the frequency BUDGET go? Where the **scorer** is sensitive — not where the eye is. UNIWARD
(Holub–Fridrich 2014; Yousfi is Fridrich's student → the contest IS inverse steganalysis) is a directional-
wavelet **relative-distortion cost**: low cost in textured/HF regions (undetectable), high cost in smooth
regions (detectable). Inverted for us: it maps **where in the wavelet domain a change moves a detector** —
exactly where to spend HF. Compose:
- `tac.uniward_delta.compute_uniward_cost_map` — S-UNIWARD per-pixel detectability on rendered frames.
- `tac.logit_margin_sensitivity_weighted.{normalize_sensitivity, sensitivity_weighted_logit_margin_loss}` —
  the per-pixel SegNet **argmax-margin** weight (thin margin = a frequency there flips the class).
- the scorer-Jacobian `∇_x (SegNet,PoseNet)` (Taylor §5) — the exact first-order sensitivity field.
→ a **scorer-matched bandwidth map**: allocate frequencies where (UNIWARD-detectable ∧ thin-margin ∧
high-Jacobian). This is the non-arbitrary "give each frequency precisely what the scorer needs."

### 5. Taylor — the scorer response surface + the operating-point marginal
`S(x+δ)=S(x)+∇S·δ+½δᵀHδ`. `∇S` (the response surface, task #35) = which pixels/frequencies move the score
LINEARLY. The Hessian `H` is large/ill-defined near argmax boundaries (the score is non-smooth there) →
those are exactly the locations needing careful frequency placement, not more amplitude. And the **sqrt in
the pose term** has a Taylor structure that already governs allocation: `d√(10·d_pose)/d(d_pose) =
5/√(10·d_pose) → ∞` as `d_pose→0` — so at the PR106-class operating point the pose-axis marginal dominates
(already in CLAUDE.md "SegNet vs PoseNet importance"). The frequency allocation must be **operating-point
aware**, weighted by these per-axis Taylor marginals.

### 6. Maxwell / wave dispersion — the motion-aligned spatiotemporal basis (the elegant PoseNet insight)
A wave/field has a **dispersion relation** `ω(k)`. The video analog: a texture moving with velocity `v`
places its spectral energy on the **plane `ω = −k·v`** in 3D (kx,ky,ωt) space. PoseNet keys on EGO-MOTION
(a coherent global flow). So the two-frame carrier should be represented in a **motion-compensated basis
aligned to the dispersion plane** — NOT as two independent per-frame images. This is the rigorous statement
of "PoseNet wants a temporal trajectory tube": represent frame1 as a flow-warp of frame0 plus a sparse
residual, so the pair's spectral energy concentrates on the motion plane. Ties to RAFT / ego-motion / the
LA-pose foveation lane. Almost no carrier does this; it's a high-ceiling pose-axis idea.

### 7. Shannon water-filling — the rate bridge (we have the primitive)
The rate-optimal allocation of archive BYTES across frequency bands is **reverse water-filling on the
scorer-weighted power spectrum**: spend bits where (scorer-weighted spectral energy) is high, starve bands
below the water level. This is the missing **spectral coordinate of the meta-Lagrangian** — it ties
frequency directly to the `25·bytes/N` term, so "which frequencies" and "how many bytes" are solved jointly,
not separately. We HAVE water-filling in `engineered_corrections.py` / `hidden_gems.py` (applied to tensors,
not yet to the frequency/PSD axis).

## What else we are NOT considering (beyond the named)
- **Scorer-Jacobian SVD = the optimal basis (the unifying idea).** The decoder solves an inverse problem
  through the frozen scorer; the task-optimal synthesis basis is the **right singular vectors of
  `∂(SegNet,PoseNet)/∂pixels`** (the directions the scorer is most sensitive to). Fourier/wavelet/Gabor are
  generic; this is data+task-specific. UNIWARD (§4) is the steg-derived estimate of it; the response surface
  (§5) is the first-order estimate. Synthesizing in the top-k right singular vectors is the rigorous "spend
  capacity exactly where the evaluator looks."
- **Steerable / oriented bases (curvelets, Simoncelli steerable pyramid).** Dashcam scenes are anisotropic:
  horizon + lane lines (horizontal/vertical) + radial ego-flow. An ORIENTED basis matched to the scene
  geometry beats isotropic Fourier — fewer coefficients for the same boundary fidelity.
- **Information-bottleneck frequency selection (Tishby, council).** Keep frequencies with high
  `I(frequency ; scorer-output)` — the IB-optimal spectral subset, the rate-distortion-on-frequency view.
- **MDL on the spectrum.** Because of the bytes term, the win is the SPARSEST frequency set that holds the
  scorer terms — minimum-description-length over `{ω_k}`, not maximum fidelity.

## Synthesis — the principled replacement (one framework)
Stop tuning one global `w`. The carrier should: (a) cap frequency per scale at Nyquist (§0); (b) carry a
LEARNABLE frequency set initialized from the measured spectrum (§1,§2); (c) in a SPATIALLY-LOCALIZED basis
(§3, wavelet/grid); (d) with the frequency BUDGET allocated by the scorer-matched map (§4 UNIWARD ∧ margin ∧
Jacobian) weighted by the operating-point Taylor marginals (§5); (e) for the two-frame term, in a
MOTION-ALIGNED basis on the dispersion plane (§6); (f) with archive bytes water-filled across bands (§7).
The unifying target is the **scorer-Jacobian-matched basis** (the optimal-basis idea), which all of the
above approximate.

## Actionability (per the Vehicle OS + maturity ladder)
This is an **L2 intrinsic-optimization upgrade**, NOT a new vehicle — it's how the carrier expresses the
boundary the scorer asks for, and it composes EXISTING primitives (no invention):
1. **Cheapest principled win (do after F1 arm B confirms the skip mechanism):** replace global `sin_frequency`
   with (a) a per-scale Nyquist schedule + (b) a learnable per-channel `ω` initialized at the schedule.
   ~25 LOC, gated, default-OFF; falsifiable: it should match-or-beat the best hand-tuned `w` with NO
   hand-tuning, and remove the w=30-vs-w=1 fork entirely.
2. **The evaluator-matched allocation (the real prize):** wire `compute_uniward_cost_map` +
   `sensitivity_weighted_logit_margin` + the scorer-Jacobian (response surface #35) into a per-pixel/per-band
   frequency-budget map consumed by the carrier (and by the bit-allocator via water-filling §7). This is the
   "non-arbitrary frequency, derived from the scorer" answer, and it slots into the meta-Lagrangian as the
   missing spectral coordinate.
3. **The motion-aligned pose basis (#6):** a separate high-ceiling lane (flow-warp frame0→frame1 + sparse
   residual) for the PoseNet axis — needs its own manifest + intrinsic proof.

Do NOT branch to these before F1 arm B proves the skip mechanism (don't stack a principled frequency basis
on an unproven residual path — the OS rule: intrinsic-before-contextual, no bolt-on before the base passes).
The immediate hand-tuned arm B is the cheap mechanism gate; this memo is the iteration that follows it.

## This is one symptom of the ARBITRARINESS CLASS (operator, 2026-06-09)

`w=30` is not a frequency bug — it is one instance of a **class**: magic constants set by convention /
cargo-cult inheritance rather than DERIVED from the problem, MEASURED from the data/scorer, or LEARNED by
gradient descent. The same class includes (non-exhaustive, from the carriers + the shared harness): the
learning rate, grad-clip norm, EMA decay 0.997, the σ-noise schedule 0.2→0.1, the 8-stage epoch counts
(3k/5.65k/1.5k/…), the latent dims (16/20/24), the channel taper (48,40,32,…), the mid/fine injection
block indices (2,4), the codebook size, brotli quality=11, the trust-region slacks, the distill-weight
defaults (the Mistake-B `0.0`!), and the recon-vs-scorer loss weights. Each is a scalar standing in for a
decision that SHOULD have a provenance.

**The class-level fix (the non-arbitrariness principle, operationalized):** every numeric constant in a
carrier/trainer carries a provenance tag — `DERIVED` (from a formula/theorem, e.g. per-scale Nyquist),
`MEASURED` (from the data/scorer, e.g. ω from the scorer transfer function below), `LEARNED` (a trainable
parameter), or `ARBITRARY` (convention → must be justified or replaced). This is the **HARD-EARNED vs
CARGO-CULTED** classification (CLAUDE.md cargo-cult audit / Catalog #303) applied at the constant level,
and it slots into the Vehicle OS manifest as a `constants_provenance` section: a vehicle cannot reach L2
(intrinsically optimized) while score-relevant constants are tagged `ARBITRARY`. The frequency synthesis
above is the worked example of converting ONE arbitrary constant (`w`) into DERIVED (Nyquist) + LEARNED
(Fourier features) + MEASURED (the scorer transfer function). The same recipe generalizes to the rest of
the class.

## Custom spectral / frequency analysis tools to build or extend (operator-authorized 2026-06-09)

We are not limited to existing primitives — build bespoke spectral tooling that MEASURES, so the constants
become derived, not guessed:

1. **Scorer spectral-sensitivity analyzer (the empirical scorer transfer function) — `$0`, build first.**
   Take source frame pairs; inject band-limited perturbations (2D-FFT radial band masks, or a wavelet
   sub-band sweep) into the render at each frequency band `k`; measure `Δd_seg` (frame1) and `Δd_pose`
   (both frames) through the EXACT frozen `DistortionNet` per band. The result `H_seg(k)`, `H_pose(k)` is
   the scorer's **spectral transfer function** — the empirical, non-arbitrary answer to "which frequencies
   matter," and the MEASURED initializer for the learnable `ω`, the FF bandwidth `σ`, and the water-fill
   levels. It also tells us, right now, whether arm B's hand-picked `w=1` even sits where the scorer is
   sensitive. (Frozen-scorer read-only measurement; touches no carrier — safe to build + run immediately.)
2. **Source + boundary PSD estimator.** The contest video's power spectrum AND the SegNet-edge spectrum
   (the gradient/argmax-boundary field's PSD) — the target the carrier must represent.
3. **Scorer-Jacobian SVD basis extractor.** `∂(SegNet,PoseNet)/∂pixels` via MLX `vjp`/`jvp` (the atlas
   engine, task #36); its top-k right singular vectors = the task-optimal synthesis basis (the unifying
   idea). The measured upper bound on "how few coefficients suffice."

These three are the measurement layer that DERIVES the frequency constants — extincting the arbitrariness
for this knob and templating the class-level fix. They are intrinsic-mechanism measurements (frozen scorer),
not carrier bolt-ons, so they are safe to build now without waiting on F1 arm B.

## Cross-refs
`b1_f1_bilinear_skip_canonical_primitive_landed_20260609.md` (the skip + the w=30 trap) ·
`docs/vehicle_operating_system.md` (the L2 intrinsic-optimization bar this serves) · `tac.uniward_delta` ·
`tac.logit_margin_sensitivity_weighted` · `tac.wavelet_variance` · `engineered_corrections.py` (water-fill)
· task #35 (evaluator response surface = the scorer-Jacobian input).
