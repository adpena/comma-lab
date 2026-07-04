# Deep-math lens — MICROLOCAL / CURVELET-SHEARLET / WAVEFRONT SET + se(3) TEMPORAL FLOW: the optimal CODE of the separatrix and its transport

> Chapter 5 of "Amortizing the Argmax." The separatrix's optimal sparse chart is the **curvelet/shearlet**;
> the **wavefront set** is the microlocal object our self-orient basis approximates; the along-tangent-freq
> deficit is a **wavefront undersampling**; and the whole temporal boundary evolution is **one se(3) screw
> orbit** — a canonical transformation that transports the code, so ξ encodes it once.

- **UTC** 2026-07-04 · **git** `bef5a95d4` · **axis** `[macOS-CPU deep-math derivation] NON-PROMOTABLE`
- **pointer UNMOVED 0.19110** · `score_claim=false` · `promotable=false` · `ready_for_exact_eval=false`
- **Scope** $0, CPU-only, NO GPU / NO training / NO paid dispatch. Live #205 run SACRED — read-only; this
  memo touched no state files, no trainer, no checkpoints. MEANS (a theory lens), decade-compounding; the
  pointer moves only on a byte-closed `upstream/evaluate.py` n600 row < 0.19110.
- **Grounded in our measured rows** (not asserted): the −48% self-orient directional basis (`CANONICAL_RESEARCH_INDEX`
  D1); R measured near-all-pass |H|≥0.842 to Nyquist (`signal_processing_filter_levers_derived_20260701`);
  the INR-NTK + SegNet-stem localization of the true low-pass (ibid §3); the SDF-vs-spectral 587× post-R
  (`levelset_curvelet_witness_feasibility_20260627`); the ~8-dim nonlinear lane manifold; W1 Chasles screw
  ~0-byte win / W4 through-R warp negative (`canonical_research_index_vehicle_warp_20260629`); #257 ξ→H
  free-derive, 52,135→3,762 B (`store_nothing_pose_optimal_coder_20260703`); T_floor S=0.11797 rate-dominated
  + the indirect-RD `R_X(D_Y) < R_Ỹ^{E&C}` dominance theorem (`CAPSTONE_witness_taskspace_..._20260621`).

---

## 0. HEADLINE — three claims, each labeled by evidence tier

1. **[PROVEN theorem, applied]** For a cartoon (piecewise-C² with a C² boundary), the **curvelet / shearlet
   N-term approximation error decays as O(N⁻²(log N)³)** — strictly better than wavelets O(N⁻¹) and Fourier
   O(N⁻¹ᐟ²) (Candès–Donoho 2004; Kutyniok–Labate 2011). The SegNet argmax partition IS a cartoon. Therefore
   the **minimal-rate description of the separatrix is a shearlet N-term code** — and our `--self-orient`
   directional basis is a *discrete shearlet-like system*, which is WHY the D1 lever measures **−48%** (it is
   the anisotropic parabolic-scaling basis, not a heuristic).

2. **[CONJECTURED, measurement-supported] THE CODE CLAIM:** the microlocal N-term code of the Fisher-metric
   separatrix is the **rate-optimal indirect-RD (task-space) code**. Proven part: N-term rate is an *upper
   bound* on the task-space rate (the boundary error controls the argmax-flip distortion). Conjectured part:
   it is *tight* (achieves the floor). Support: the annulus concentration (1.4% of cells carry the detector
   sensitivity), the 587× SDF-vs-spectral post-R gap, and the dominance theorem `R_X(D_Y) < R_Ỹ^{E&C}` all
   say the task-space code is strictly cheaper than any reconstruct-RGB code — the shearlet count is a
   concrete realization of that cheaper code.

3. **[PROVEN geometry + MEASURED residuals] THE se(3) TEMPORAL-SUFFICIENCY CLAIM:** for the STATIC classes
   (Road / Undrivable-sky / MyCar-hood) the separatrix's temporal evolution is EXACTLY the push-forward of
   the initial separatrix by the ego-**screw** ξ∈se(3) (Chasles' theorem + the planar-homography
   `H=K(R−t nᵀ/d)K⁻¹`, exact for a ground plane). So **ξ is the temporal sufficient statistic** — encode
   ONCE (#257: ξ→H free at decode, 13.9× rate cut). The push-forward is a **canonical transformation on the
   wavefront set** (Hörmander): the same ξ transports *position AND codirection*, hence it transports the
   shearlet ORIENTATIONS too → the temporal *code* is generated, not stored. Honest residuals (all MEASURED):
   the SegNet readout adds ~0.008 inter-frame jitter (W4), independent scene motion (Movable) is NOT the
   ego-screw (W2 stratified), and the dashed-lane along-tangent structure is a genuine extra singularity.

---

## 1. MICROLOCAL FORMALISM — the wavefront set of the argmax, and its optimal sparse chart

### 1.1 The object: a cotangent-bundle singularity, not a spatial one

Let `L: Ω⊂ℝ² → {0..4}`, `L(x)=argmaxₖ φₖ(x)`, be the SegNet argmax field (our `gt.lstars`). The separatrix
`Σ = {x : φ_top1(x)=φ_top2(x)}` is a codim-1 curve set (measured: a thin annulus — only **1.38%** of cells
at margin |m|<0.5, 4.83% at |m|<2.0; `signal_processing_..._20260701` §3). The right microlocal object is not
Σ ⊂ Ω but its **wavefront set** WF(L) ⊂ T*Ω∖0 — the set of (position x, *codirection* ξ̂) where L is
singular in direction ξ̂. For a smooth boundary curve, WF(L) = { (x, ν(x)) : x∈Σ } — a 1-D lift carrying at
each boundary point the *normal codirection* ν(x). **The boundary's DIRECTION is part of the data**, and it
is exactly the quantity our `--reorient-every 50` self-orientation estimates.

Why this matters: a distortion that measures "did the argmax flip?" is sensitive to Σ AND to ν (a flip is a
displacement of Σ *along its normal*). A basis that ignores ν (isotropic Fourier) wastes coefficients; a
basis matched to WF (anisotropic, oriented) is optimal. This is the microlocal restatement of the measured
**basis-BEFORE-capacity** law (isotropic capacity-alone HURTS +6%; oriented basis then capacity −70%,
`n205_joint_nexus` §3-3): capacity spent off the wavefront codirection is wasted.

### 1.2 The optimal chart: curvelets / shearlets and the N⁻² theorem

For `f` that is C² away from a C² edge (the cartoon model), the best M-term approximation in classical bases:
Fourier `‖f−f_M‖² ≍ M⁻¹ᐟ²`, wavelets `≍ M⁻¹`. **Curvelets** (Candès–Donoho 2004) and **shearlets**
(Kutyniok–Labate 2011, Guo–Labate) achieve `‖f−f_M‖² ≲ M⁻²(log M)³` — provably optimal for this class (no
basis does better than M⁻² up to logs). The mechanism is **parabolic scaling**: an element at scale `2⁻ʲ`
has support `length ≈ 2⁻ʲᐟ²` × `width ≈ 2⁻ʲ`, i.e. **width ≈ length²**. This anisotropy is exactly tuned so
that O(2^{j/2}) elements tile a smooth curve at scale j, each needing only O(1) coefficients — vs O(2^j)
isotropic wavelets. The curvelet/shearlet frame *resolves the wavefront set*: each element lives at a
(position, scale, ORIENTATION), so the frame coordinates ARE a discretization of T*Ω.

**Our basis is a discrete shearlet, not a curvelet.** Curvelets use *rotation* (breaks the integer lattice —
awkward on a pixel grid); shearlets use *shear* (`[[1,s],[0,1]]`, preserves the lattice → faithful digital
transform). Our `self_orientation_directional_feats` orients features by the ratio `freq_across / freq_along`
about the estimated tangent — a **shear**, not a rotation. So the honest identification is: **our D1 lever is
a discrete anisotropic shearlet system**, which is why it inherits the N⁻² optimality empirically
(measured −48% all-class vs −8% lane-only; `CANONICAL_RESEARCH_INDEX` D1 = DM2 Candès–Donoho cartoon-optimal).

### 1.3 The along-tangent-freq deficit IS a wavefront UNDERSAMPLING (the microlocal diagnosis of Lens 2)

The BASELINE config is `--freq-across 32 --n-dir-freqs 2 --freq-along 4` (`n205_joint_nexus` §5). Read
microlocally:
- **freq_across (=32)** samples the ACROSS-tangent (normal) direction — the sharp edge profile. This is the
  "width ≈ 2⁻ʲ" leg of parabolic scaling. It is well-resolved and Nyquist-bounded by the detector.
- **freq_along (=4), n_dir_freqs (=2)** sample the ALONG-tangent (boundary-parallel) direction — the
  structure *of the wavefront itself as you travel along it*.

A **solid** lane line is smooth along its length → its wavefront set is a smooth curve → low along-tangent
frequency suffices (parabolic scaling literally says the tangent leg is coarse). But a **dashed** lane line
is NOT smooth along the tangent: the on/off pattern is a periodic modulation *along* the wavefront, and each
dash end is a **corner** — a point where WF(L) carries a *fan* of codirections (a corner is microlocally
"omni-directional"). Dashes therefore demand higher along-tangent frequency than a smooth curve. Lens 2
measured the dash along-tangent content at **~25 cyc/unit** while freq_along resolves **≤8** — a **3.2×
undersampling** of the along-boundary singularity. This is not a tuning nit; it is the precise microlocal
statement that **our shearlet frame is band-limited below the dashed wavefront's tangent-frequency support.**

The fix is microlocal, not ad-hoc: **raise the along-tangent resolution** — `--n-dir-freqs 2→4` (equivalently
lift `--freq-along`), which is exactly re-balancing the parabolic anisotropy toward a *rougher* wavefront.
**But two hard constraints (both MEASURED) bound it:**
- **Nyquist-safety (across leg):** `freq_across · 2^(n_dir_freqs−1) ≤ stem_Nyquist (~64 cyc/unit)`
  (`adversarial_review_round1_config` M3). Naively raising n_dir_freqs at freq_across=32 blows the across leg
  to 1024 cyc/unit (16× over) — pure waste that the SegNet stem cannot see. The microlocally-correct move is
  to **trade across for along**: `--freq-across 8 --n-dir-freqs 4` (product ≤ 64), i.e. re-shape the
  parabolic tile to be *shorter and rougher*, the shearlet matched to a dashed edge.
- **Detector floor (physics, hard):** dashes <2px@384 (area<5px, ~3536 dashes, 98.5% erased) are below the
  SegNet stride-2 stem Nyquist — **unrecoverable by ANY basis at ANY contrast** (`signal_processing_...` §3).
  The along-tangent lever helps ONLY the 5–80px detector-RESOLVABLE dashes; sub-2px dashes need a *store*
  (Yousfi flip-sidecar) or a *deterministic openpilot raster*, not a finer wavefront chart. **Microlocal-
  optimal ≠ recoverable when the measurement operator band-limits below the singularity scale** — the single
  most important honesty in this chapter.

### 1.4 Why R is not the villain (measured), and what that means microlocally

The eval roundtrip R (bicubic↑384→874 → uint8 → bilinear↓384) was measured **near-all-pass**: |H|≥0.842 to
Nyquist, dashes ≥91% survival at 1px (`signal_processing_...` §1). Microlocally: **R does not move the
wavefront set** — it is a mild pseudodifferential smoothing of order ~0 with no in-band null, so WF(Rf)≈WF(f).
The singularities the code must carry survive R. The true band-limiter is (a) the INR/NTK generator
(spectral bias — it under-*produces* fine wavefront frequencies) and (b) the SegNet stem (it under-*reads*
them). The code (what to store) is therefore an intrinsic property of Σ; R only sets the *sampling* at which
the code must be exact (2px), and that sampling PRESERVES the code. This is the precise sense of the task
line "R all-pass to 2px = the sampling that preserves the code."

---

## 2. THE CODE — is the microlocal N-term the indirect-RD floor?

### 2.1 The task-space (indirect) rate-distortion problem

We do not code the video X; we code the machine readout Y = SegNet-argmax(X) under d_seg = argmax-flip-rate.
This is **indirect / remote source coding** (Dobrushin–Tsybakov 1962; Wolf–Ziv 1970; the CEO problem, Berger–
Zhang–Viswanathan 1996) — the encoder sees X but is scored on a *function* of it. Our own dominance theorem
(`CAPSTONE_...20260621`; `CANONICAL_RESEARCH_INDEX` R-floors) states `R_X(D_Y) < R_Ỹ^{E&C}`: coding directly
for the task is strictly cheaper than reconstruct-then-threshold. Every vehicle built so far sits on the
DOMINATED reconstruct-RGB rung; the task-space rep that realizes the strict inequality **has never been
built**. T_floor S=0.11797 is rate-dominated (61.7%) but LOOSE/refuted-as-realizable (it assumes d_seg→0
byte-cheaply); the true task-RD optimum S* is strictly inside (0.118, 0.191).

### 2.2 The bridge: N-term rate ↔ task-RD rate

Claim: **the shearlet N-term code of Σ (to detector-resolvable precision) is a concrete construction of the
task-space code `R_X(D_Y)`.** Argument in three steps, honestly tiered:

1. **[PROVEN] d_seg is a boundary-displacement functional.** A flip at pixel p ⇔ Σ crosses p, i.e. the
   normal displacement `δ(x)=|Σ̂−Σ|` exceeds the pixel half-width where the margin is small. So
   `d_seg ≲ (1/|Ω|)·∮_Σ 𝟙[δ(x) > τ(x)] ds`, with τ(x) the local margin-to-pixel scale. d_seg is controlled
   by the boundary error in the margin metric — NOT by any RGB error. (Consistent with the 587× SDF-vs-
   spectral post-R: an exact boundary rep beats a band-limited-indicator rep even though both "look" similar.)

2. **[PROVEN, as an upper bound] the shearlet code controls the boundary error.** Encoding Σ with N
   above-threshold shearlet coefficients gives boundary L²-error `≲ N⁻²` (the cartoon theorem, restricted to
   the edge). Feeding step 1, `d_seg ≲ (boundary error) ⇒ rate-to-reach-D_Y ≤ |{shearlet coeffs to reach
   boundary error √D_Y}|`. So the N-term count is an **upper bound** on `R_X(D_Y)`.

3. **[CONJECTURED] tightness.** Is the shearlet count also a *lower* bound (does it hit the floor)? The
   metric entropy of C²-boundary-curve classes matches the shearlet N-term count (Grohs–Kutyniok "optimal
   sparsity" results), so for the geometric distortion the shearlet code is order-optimal. The remaining gap
   is the *margin metric ≠ Euclidean metric*: d_seg weights boundary error by the Fisher/margin field
   (Fisher↔−margin Pearson 0.978, `signal_processing_...` §3), so the truly optimal code is a **weighted /
   anisotropic shearlet** whose thresholding respects τ(x). Whether the weighted-shearlet count equals
   `R_X(D_Y)` exactly is the open theorem. Measurement leans yes (annulus concentration ⇒ most Σ carries
   near-zero task-weight ⇒ few coefficients matter), but it is CONJECTURED, not proven.

### 2.3 The rate ledger this predicts (the counted archive payload)

The microlocal code says the COUNTED bytes are exactly:
`rate = |weighted-shearlet coeffs of Σ above the τ-threshold|  +  |ξ SE(3) B-spline|  +  |Movable-class store|`
and everything else is **rule-118 FREE** (the generic shearlet bank regenerated from ~5 scalars at decode
— `levelset_curvelet_witness_feasibility` proved the bank is byte-FREE/GT-free; the deterministic openpilot
lane raster #203; the ξ→H derivation #257). This is the RATE half of sub-0.15 made concrete: the frontier
rate is 0.1185 (177 KB reconstruct-RGB); the microlocal target is the *shearlet coefficient count of a thin
annulus* — a much smaller sufficient statistic, IF the tightness conjecture holds and the generator can be
trained to realize it (the binding UNVERIFIED item; the −48% is still circular-GT, `CANONICAL_RESEARCH_INDEX`
open-item 5).

---

## 3. se(3) TEMPORAL FLOW — the separatrix as one screw orbit, and code transport

### 3.1 Chasles + the planar homography (EXACT)

**Chasles' theorem (proven, 1830):** every rigid motion of ℝ³ is a **screw** — a rotation about an axis
composed with a translation along that same axis, i.e. an element `ξ∈se(3)` with `g=exp(ξ)∈SE(3)`. Ego-motion
between two frames is such a `g`. For a world plane with normal n at depth d, the induced image-plane map is
the **homography** `H = K(R − t nᵀ/d) K⁻¹` (pinhole projection; exact for planar structure). This is exactly
`homographies_from_xi` in #257 (`src/tac/se3.py` + `tac.boundary_math.xi_pose_coder`): `exp_se3(ξ)` → H,
derived FREE at decode from fixed calibration.

### 3.2 ξ is the temporal sufficient statistic for the static classes (PROVEN geometry)

For a static world class lying on (or approximated by) a plane, its separatrix at frame t is the push-forward
of the frame-0 separatrix by `H_t = homographies_from_xi(ξ_t)`:  `Σ_t = H_t(Σ_0)`. Measured confirmations:
- **W1 (MEASURED-WIN):** Road reproduced EXACTLY by the shared twist; hood→identity (#139 static core);
  sky→rotation-only `KRK⁻¹`. O(10) static params/clip vs ~6,600 per-pair per-class-homography params.
- **#256/#257 (MEASURED):** the shipped per-pair fp64 H block (43,200 B) is **redundant given ξ** — H carries
  no per-pair info beyond ξ (derivation exact). Dropping it: 52,135 B → 3,762 B coded ξ = **13.9× rate cut**,
  bit-exact-gate proven. This is the "encode ξ once" claim, *realized and measured*.

So: **for the static classes the temporal boundary evolution is EXACTLY an ego-screw push-forward, and ξ (a
6-DOF-per-frame twist, entropy-coded as a temporal delta, or a low-order SE(3) B-spline over the clip) is the
temporal sufficient statistic.** This is proven at the geometry/pose level.

### 3.3 The canonical transformation — code transport, not just pose transport (the unification)

Here microlocal analysis and se(3) meet. A homography H acts on Ω; its **cotangent lift** `H_*` acts on T*Ω,
`(x, ξ̂) ↦ (H(x), (dH)⁻ᵀ ξ̂)` — a **symplectomorphism** (canonical transformation). Hörmander's theorem on
propagation of singularities gives `WF(H·f) = H_*(WF(f))`: **H transports position AND codirection together.**
Therefore the shearlet ORIENTATIONS at frame t are the pushed-forward orientations from frame 0 — the *code*
is transported, not re-estimated. Consequence (the encode-once-at-the-CODE-level claim): store the weighted-
shearlet code of Σ at a keyframe + the ξ_t sequence; the decoder **generates** every intermediate frame's
oriented code by applying the canonical transform `H_{t*}`. The ego-screw is the generator of the temporal
code flow. This is strictly stronger than #257 (which transports the pose/H); it says the *microlocal code
itself* rides the same ξ — the deepest form of the "one object" grok (`CANONICAL_RESEARCH_INDEX` G9/O2:
d_seg & d_pose are two readouts of the SAME sufficient statistic).

### 3.4 Honest residuals (MEASURED — where exactness stops)

The temporal-sufficiency claim is EXACT for static-class *geometry*, not for the *d_seg readout*:
- **W4 (MEASURED-NEGATIVE, robust):** warping a neighbor THROUGH R inherits the inter-frame SegNet boundary-
  **jitter floor ~0.008**; bulk through-R d_seg = 0.0048/0.0051 ≈ **4× the 1.23e-3 budget**. The SegNet
  argmax is NOT a deterministic function of the geometric boundary — it has irreducible readout noise. So
  ξ-transport gives you the geometry for free but does **not** collapse d_seg to zero. **Warp-the-bulk for
  d_seg is refuted; ξ-transport is a RATE win and a d_pose win, not a d_seg solve.**
- **W2 (stratified):** Movable/other-cars move INDEPENDENTLY — not the ego-screw. They are a separate
  stratum (store, ~sparse). The single ξ is sufficient only for the static strata.
- **Non-planar parallax:** the ground-plane homography is exact only on the plane; curbs/poles/3D structure
  have parallax a single H misses (W1 handles it by *stratifying* per class; the residual is the fine 3D
  structure — small, and largely below the detector floor anyway).
- **W9 nuance:** the bulk partition is intrinsically STABLE for 47+ pairs *by partition stability*, and the
  d_seg-optimal warp is ≈ near-identity — i.e. over short windows the transport is nearly trivial; its value
  is RATE (don't re-store) and POSE, not d_seg reduction.

---

## 4. HONESTY LEDGER — proven / conjectured / false-friend

**PROVEN (real theorems, correctly applied):**
- Curvelet/shearlet N-term O(N⁻²(log N)³) optimality for C²-cartoons (Candès–Donoho 2004; Kutyniok–Labate).
  Our argmax field is a cartoon; the theorem applies.
- Chasles: every rigid motion is a screw. The planar homography `H=K(R−t nᵀ/d)K⁻¹` is exact for a plane.
  ξ→H is an exact derivation (#257 bit-exact gate).
- Hörmander propagation of singularities: `WF(Hf)=H_*(WF f)` — the canonical transform transports position
  and codirection jointly. (Pure math; applies to the smooth-diffeomorphism part of H.)
- N-term rate is an UPPER bound on the indirect-RD rate `R_X(D_Y)` (via the boundary-displacement bound on
  d_seg).

**CONJECTURED (measurement-supported, not proven):**
- **The CODE claim's tightness:** weighted-shearlet N-term = `R_X(D_Y)` exactly (order-optimal is proven; the
  margin-metric weighting making it the *floor* is open). Support: annulus concentration, 587× SDF gap,
  dominance theorem — all consistent, none a proof.
- **Code-transport realizability:** that a trained generator can actually *carry* the shearlet code and apply
  `H_{t*}` to hit d_seg through R (§3.3). This is the untested strong form; #205 tests the weaker per-frame
  self-orient basis, not the transported-code form.
- **α≈2.34 RD exponent** (borrowed from 2 points, `n205_joint_nexus`): tempting to read as "the N⁻² curvelet
  exponent," but our exponent is a d_seg-vs-bytes fit, not an L²-vs-N curvelet fit. Coincidence-until-proven.

**FALSE-FRIENDS (traps this chapter explicitly disarms):**
- **"R is the low-pass, deconvolve it."** MEASURED-FALSE: R is near-all-pass (≤+1.25 dB Wiener headroom).
  Microlocally R doesn't move WF; deconvolving it is worthless. The band-limiter is the INR (produce-side)
  and the stem (read-side).
- **"Curvelet N⁻² ⇒ we can code the sub-2px dashes cheaply."** FALSE: the DETECTOR (stem Nyquist 2px@384)
  band-limits below those singularities. No basis recovers a singularity the measurement operator erases;
  sub-2px dashes need a STORE, not a finer chart. Microlocal-optimal ≠ recoverable.
- **"ξ-transport is a d_seg solve (one object ⇒ warp the bulk free)."** MEASURED-FALSE (W4, 4× budget): the
  readout jitter floor ~0.008 is irreducible by transport. ξ is a rate/pose sufficient statistic, not a
  d_seg eraser. (This is the refined-DOWN version of the G9 grok — honest, per FEED-jq/iz.)
- **"Our basis is literally curvelets."** Imprecise: it's a discrete *shearlet* (shear, lattice-faithful),
  not rotated curvelets. Same N⁻² class, different (correct-for-a-grid) implementation.

---

## 5. ENGINEERING NEXUS — ranked levers with honest EV

All ΔS are DERIVED ranges (not exact-eval rows). Witness baseline d_seg=0.006655; sub-0.19 needs ≤0.00118,
sub-0.15 ≤0.00077 (`signal_processing_...` §4). d_seg error split: shift 76.5% / erasure 23.5%.

| # | lever | microlocal reading | derived EV | $0-now? | composes | binding caveat |
|---|---|---|---|---|---|---|
| **M1** | **Along-tangent-freq / `--n-dir-freqs 2→4` (with `--freq-across 8`)** | resolve the dashed-wavefront's tangent frequency (3.2× deficit); re-shape the parabolic tile for a rougher WF | **moderate on the erasure/dash slice** (targets 5–80px detector-resolvable dashes, ~15% of erasure mass ⇒ up to ~−2e-4 d_seg) | flag-only to CONFIGURE ($0); training-side to REALIZE | M2 (NTK), M4 (routing) | MUST keep `freq_across·2^{n_dir_freqs−1} ≤ 64` (Nyquist); sub-2px dashes still need a store |
| **M2** | **NTK / multiscale band-pass whitening** (per-scale shearlet amplitude ∝ 1/√λ_scale, capped at 2px@384) | make every shearlet SCALE converge uniformly (the multiscale = curriculum = persistence facet); this IS the microlocal preconditioner | **dominant SPEED lever ~3–10× on finest band**; up to ~−3e-4 d_seg on the addressable erasure slice | training-side | M1, M5, freq-curriculum | conditioning/step-size stability caps the speedup; don't synthesize below detector floor |
| **M3** | **se(3) screw carrier (#257 ξ→H free) + code transport (§3.3)** | encode-once temporal sufficient statistic; canonical-transform the code | **RATE: MEASURED-WIN** (52,135→3,762 B, 13.9×). **d_seg-via-warp: MEASURED-NEG (W4)**. Code-transport form: UNMEASURED research lever | ξ→H is $0 decode-side (built, bit-exact); code-transport needs a run | M4 (pose), all rate | temporal exactness is geometry-only; readout jitter ~0.008 + independent motion are residual |
| **M4** | **weighted-shearlet threshold = margin-saliency / UNIWARD WHERE-prior** (Fisher-metric τ(x)) | threshold coefficients by the Fisher/margin weight, not L²; spend only on the annulus | **high on shift slice** (64% of flips at margin<0.5); MULTIPLIES M1/M2; prunes ~95% of frame | prior is $0 to compute (margin field); realize training-side | all | it routes, doesn't add flips; needs the #141 saliency (one backward) |
| **M5** | **subspace projection (~8 nonlinear DOF, Whitney 17–19)** | the code lives on a low-dim manifold — the shearlet frame's active atoms span it | neutral-to-+ d_seg; direct SPEED (fewer binding DOF) | training-side | M2 (band-pass × subspace = joint preconditioner) | manifold is NONLINEAR (linear store-the-flips NO-GO ×3) |
| ~~M6~~ | ~~R amplitude deconvolution / pre-emphasis~~ | R doesn't move WF | **~0 (measured ≤+1.25 dB)** | $0 but worthless | — | disarmed false-friend; do NOT build |

**Top recommendation (honest, means-level):**
1. **M1 + M2 together** are the microlocal upgrade of the current basis: re-balance the shearlet anisotropy
   toward the dashed wavefront (M1) AND whiten the scales so the fine band converges (M2). Both bake into an
   optimal-form run; M1 is a $0 config change (respecting the Nyquist cap), M2 is a per-scale amplitude
   schedule. These directly de-risk the still-CIRCULAR-GT −48% by making the *realized* self-orient basis
   match the wavefront it must code.
2. **M3-rate is BANKED** (#257 shipped, 13.9× — the pointer-relevant rate win is already real once byte-
   closed). **M3-code-transport** is the decade-compounding research bet: it is the only lever that makes the
   *temporal* code near-free, but it is UNMEASURED and gated behind the d_seg readout-jitter wall (W4) — so
   it is a *rate/pose* lever, honestly, not a d_seg solve.
3. **M4 (Fisher-weighted thresholding)** is the tightness half of the CODE claim — it is what turns the
   proven N-term *upper bound* into the *task-optimal* code. $0 to compute the prior; the realization is the
   sub-0.15 rate story.

**What NOT to do:** don't deconvolve R (M6, disarmed); don't chase sub-2px dashes with a finer basis (detector
floor); don't claim ξ-transport erases d_seg (W4); don't cite the −48% as byte-closed (still circular-GT).

---

## 6. means ≠ ends

This chapter is a THEORY LENS (a MEANS). It formalizes WHY the measured levers work — the separatrix's optimal
code is a weighted discrete shearlet (the microlocal chart of its wavefront set), the along-tangent deficit is
that chart under-resolving the dashed wavefront, and the temporal boundary is one ego-screw orbit whose
canonical transform carries the code so ξ encodes it once. It produced NO exact row; the pointer stays 0.19110.
It sharpens three binding cruxes for the byte-closed run: (1) realize the −48% self-orient basis with the M1/M2
wavefront re-balancing (de-risks the circular-GT); (2) the CODE claim says the counted payload is a Fisher-
weighted shearlet coefficient count — the rate half of sub-0.15; (3) ξ is the temporal sufficient statistic
(rate/pose banked; code-transport a research bet, d_seg-via-warp refuted). Feeds #205 (the run realizes M1/M2/
M4), #257 (M3-rate, banked), #203 (the deterministic raster for the sub-detector-floor dashes M1 cannot reach).
The pointer moves only on `upstream/evaluate.py` (CPU/CUDA, never MPS) < 0.19110.
