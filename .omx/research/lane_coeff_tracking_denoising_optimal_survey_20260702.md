# Wave-F Stage-2b — the OPTIMAL lane-coefficient tracking + denoising code (correspondence-first, R-D-optimal, deep-math + OSS survey)

**Status:** RESEARCH SYNTHESIS (2026-07-02). Advisory / build-only; **pointer contest-CPU 0.19110 UNMOVED**
(moves only via a byte-closed `upstream/evaluate.py` n600 exact row). Every rate number below is either a
MEASURED `[macOS-CPU advisory]` byte-count already on disk, or an explicitly-labelled DERIVED expectation.
NOT a score claim. Feeds the Wave-F Stage-2 follow-on (#229) + #205.

**Reads-on (the MEASURED ground truth this builds on — do NOT re-derive):**
`wave_f_lane_band_rd_code_LANDED_stage1_measured_20260702.md` (LBND2, n600 rate **0.02765**, delta-stream
Shannon floor **0.0174**) + `wave_f_unified_xi_build_measured_20260702.md` (ego-predictive LBND3 = clean
NEGATIVE 1.04–1.34×; source moving-average smoothing = POSITIVE **−42% → 0.01608** but LOSSY on geometry +
does NOT fix slot-swaps). R1 backbone: `wave_f_lane_band_rd_research_synthesis_20260702.md` (Wyner-Ziv /
task-RD Lagrangian). Design authority: `wave_f_optimal_lane_band_rd_code_design_20260702.md`.

---

## TL;DR verdict — the crux moved, and so must the code

The R1 synthesis put **ego-motion (L1)** as the dominant lever and DCT/KLT as the temporal transform. The n600
measurement **FALSIFIED the ego-predictor** (LBND3 every variant WORSE) and revealed the true crux the operator
now names: the frame-to-frame coeff delta is **(a) per-frame fit JITTER + (b) SLOT-SWAP discontinuities**
(44% of the temporal-delta L1 mass sits in the top-5% jumps = a labeling/association artifact, not real motion).
Crude temporal moving-average got −42% but is **doubly wrong**: it BLURS real geometry edges (lossy on d_seg) AND
it is *actively harmful at swap frames* (averaging across a swap renders a lane at a phantom in-between position).

**The optimal is not a better smoother — it is a different decomposition, applied in the right ORDER:**

> **CORRESPONDENCE-FIRST (fix the swaps, losslessly) → per-track BATCH edge-preserving denoise (kill jitter,
> keep edges, task-λ) → the existing LBND2 quantize+brotli backend.**

This is a strict, dominating improvement over moving-average on BOTH axes: correspondence removes the 44% swap
mass at **zero geometric cost** (it is a re-labeling, not a smoothing), and an edge-preserving R-D-optimal denoiser
removes the fit-jitter while **preserving** the real lane-geometry edges the moving-average destroyed. Expected
DERIVED rate: **below 0.016 with LESS d_seg loss than the moving-average** — but the d_seg-through-R leg is
UNMEASURED and IS the gate (per NO-FAKE). SOTA video-lane-detection independently validates the ordering: the
field moved from RVLD's **causal one-frame warp** (structurally identical to our failed LBND3) to LaneTCA's
**longer-window batch aggregation** — batch/aggregation beats causal-one-step, exactly our LBND3-vs-smoothing result.

---

## The deep-math backbone — why CORRESPONDENCE must come FIRST (one principle explains two measured failures)

**The single load-bearing theorem: an index-permutation discontinuity defeats every temporal model.** The LBND2
"lateral-sort into slots" step assigns slot *k* at frame *t* to the *k*-th lane by lateral position AT THAT FRAME.
When two lanes cross, or one is born/dies, the sort order flips → the coefficient time-series *of a slot* has a
**discontinuity that corresponds to no physical motion**. Every temporal operator built on the raw slot series
inherits that discontinuity:

- **A predictor** (LBND3 ego-DPCM, RVLD-style warp) predicts frame *t* from *t−1*; at a swap the prediction is
  catastrophically wrong → a huge innovation. This is *exactly* why LBND3 measured WORSE: it spent bytes coding
  swap-innovations that carry no information about the road. (MEASURED: 44% of delta L1 in the top-5% jumps.)
- **A linear smoother** (moving-average, LTI low-pass) averages across the swap → a phantom intermediate coeff
  that renders a lane between two real lanes. This is why moving-average is LOSSY *and can spill d_seg at swap
  frames*, not merely fail to help.
- **A temporal transform** (DCT/KLT/FPCA — the R1 recommendation) of a discontinuous series spreads energy across
  ALL frequencies (Gibbs) → poor compaction. DCT-first would have under-delivered for the same reason.

**Therefore the correct order is CORRESPONDENCE → (smooth / transform / predict), never the reverse.** Once each
slot tracks the *same physical lane* across frames, its coefficient series is **continuous** (a real lane's
polynomial evolves smoothly), and *then* — and only then — every temporal tool (batch smoother, DCT, low-rank,
even a predictor) works at its theoretical best. This is the deepest, cheapest, most certain lever, and it was
invisible until the ego-predict negative exposed it.

**The unifying state-space object (everything is one thing).** Model the lane manifold over time as a
**hidden-Markov / linear state-space system**:

| Symbol | Our object |
|---|---|
| hidden state `s_t` | true per-lane geometry (continuous) + the association (which slot ↔ which physical lane) |
| observation `z_t` | the per-frame noisy SegNet-mask polynomial fit (jitter + swaps) |
| MAP estimate | **association (MHT/JPDA/network-flow) + batch smoother (Kalman-RTS)** = MMSE reconstruction of the true trajectory |
| the optimal CODE | the **innovations** (whitened residual) of that estimator, entropy-coded |

Classical result (Kailath innovations / Shannon): the optimal estimator **whitens** the signal, so its residual is
the **minimum-entropy** representation → the fewest bits. The moving-average is a crude, mis-specified, non-causal
average that does NOT whiten (it leaves structure AND injects swap-phantoms). The batch smoother is the MMSE
estimator; its residual is minimal-entropy by construction. **This is the mathematically exact "less lossy than
moving-average" answer.** And it ships as a **compress-time source transform** (like the winning moving-average) —
the LBND2 decode path is unchanged, ZERO new inflate code, rule-118 clean.

**Level-set / Morse-Smale hook (the campaign's one object):** lanes are class-1 **separatrices** of the argmax
Morse-Smale complex; tracking a lane across frames = tracking a separatrix / its **birth-death persistence pair**
over time; a slot-swap is a **separatrix-reconnection (saddle-swap) event**. So "correspondence" is literally
"track the complex's critical structure across time" — it composes with the persistence component already in-tree
(the #122e59ba8 persistence landing). The temporal-consistency facet of the unified level-set flow.

---

## Ranked techniques (by expected S-reduction × d_seg-preservation × decode-legality × implementability)

### #1 — CORRESPONDENCE-FIRST: global track assignment (kills the 44% swap mass, LOSSLESS on geometry) — **THE decisive lever**

- **What.** Replace per-frame lateral-sort with a **temporally-consistent assignment**: slot *k* = the same
  physical lane for its whole lifetime. Births/deaths handled by the existing LBND2 presence bitmap (a track's
  active interval). This is a pure **compress-time re-ordering** — it changes WHICH slot holds a lane, never the
  lane's coefficients, so it is **exactly lossless on the rendered geometry** (the decoder renders each slot's
  polynomial identically regardless of index). It removes the discontinuities that no downstream tool can.
- **How (two tiers, both offline/free):**
  - **Tier A — per-frame Hungarian / LAP** (`scipy.optimize.linear_sum_assignment`, or `gagolews/lapsolver`,
    `cheind/py-motmetrics`): match frame *t*'s fits to frame *t−1*'s tracks by a coefficient-space cost
    (e.g. sampled-curve L2 in BEV or camera px). O(K³) per frame, K≤~6 lanes → trivial. Greedy-causal; can
    mis-associate through a long occlusion.
  - **Tier B — GLOBAL min-cost flow / MHT-lite** (Zhang-Li-Nevatia CVPR'08 network-flow; `motpy`, `norfair`,
    ByteTrack/SORT lineage for the birth/death lifecycle): map the whole 600-frame association to a **min-cost
    flow** with non-overlap constraints, solved globally → provably better than per-frame Hungarian (it uses all
    frames, so a momentary occlusion/cross doesn't create an irrecoverable swap). This is the RIGHT tier: the
    swap mass is a GLOBAL labeling problem and the global solver is the exact optimizer.
- **Why it dominates.** The 44% top-5%-jump mass is (measured) the swap/outlier signature. Correspondence removes
  it at **zero d_seg cost** — a strict Pareto improvement (rate ↓, distortion unchanged). It is also the ONLY lever
  that fixes the *d_seg-spill-at-swaps* failure mode of the moving-average. Ranked #1 because it is
  free-lunch, the largest single measured mass, and PRIOR to every temporal tool.
- **Rate impact (DERIVED):** removing ~44% of the delta-L1 mass plausibly takes the delta-stream entropy a large
  fraction toward the moving-average's 24 KB — but **without the blur**, so it stacks with #2/#3 rather than
  competing. MEASURE the tracked-only stream first (isolate the correspondence gain from the smoothing gain).

### #2 — JOINT / BATCH estimation on each track (the "fit ONCE"): Kalman-RTS fixed-interval smoother — **beats per-frame-then-denoise**

- **What.** On each tracked lane's now-continuous coefficient series, run a **fixed-interval Rauch-Tung-Striebel
  (RTS) smoother** (forward Kalman + backward pass) with a smooth process model (constant-velocity or
  road-curvature random-walk on the coeffs) and a measurement-noise covariance set by the fit uncertainty. The RTS
  smoother is the **MMSE** estimate given ALL frames — strictly lower variance than any causal filter (which is why
  the ego-DPCM/RVLD causal warp lost). It fuses all observations of a lane into its best trajectory → removes fit
  jitter optimally, and *interpolates cleanly across occlusion gaps* (a batch smoother fills the presence-bitmap
  holes better than carry-forward hold).
- **"Fit once, project" (the strongest form).** The design's L1 thesis realized correctly: instead of per-frame
  independent fits + then denoise, treat the tracked observations as measurements of ONE smoothly-evolving
  latent lane and estimate that latent jointly (RTS = the linear-Gaussian batch MAP). No ego estimate required —
  the smoothing is done per-track in coefficient space, so it dodges the ego-predict negative entirely.
- **SOTA validation (independent).** Video-lane-detection SOTA moved from **RVLD** (ICCV'23; recursive **one-frame**
  motion-warp — structurally = our failed LBND3) to **LaneTCA** (2024; **longer-window temporal-context
  aggregation**, higher F1). The field's own trajectory says: *batch/aggregation over multiple frames beats
  causal one-step warp.* That is our LBND3-vs-smoothing result, confirmed by the perception community.
- **Rate impact (DERIVED):** post-smoothing the per-track series is near-piecewise-smooth → its temporal-delta /
  DCT is sparse/low-order. Composes with the R1 DCT/KLT lever (which now WORKS because the series is continuous).

### #3 — R-D-OPTIMAL edge-preserving denoiser (the "less lossy than moving-average" answer): ℓ1-trend filter primary; TV/Potts for the piecewise-constant dims; task-λ

- **What.** Moving-average = LTI low-pass = optimal ONLY for stationary-Gaussian + MSE; lane coeffs are
  **non-stationary** (smooth trend + occasional REAL jumps at lane-changes/sharp curves) so a low-pass blurs the
  jumps. The optimal is a **variational edge-preserving smoother** that produces a **sparse-in-differences** signal
  (few knots) — which is *simultaneously* min-distortion (edges kept) AND min-rate (the entropy coder pays for
  differences, and sparse differences ⇒ few bits). One knob does both — this is the R-D optimum.
  - **ℓ1 trend filtering (Kim-Koh-Boyd, SIAM Review 2009)** — minimize `‖y−x‖² + λ‖D²x‖₁`; the 2nd-difference ℓ1
    penalty gives a **piecewise-LINEAR** fit with knots auto-placed at real slope changes (lane-change events).
    This is the right model for smoothly-curving coeffs. Convex, O(n) interior-point. Generalizes to
    Tibshirani's polynomial trend filtering (piecewise-quadratic if needed).
  - **1-D Total-Variation (ROF / fused-lasso)** — `‖y−x‖² + λ‖Dx‖₁` → piecewise-CONSTANT with sharp jumps
    preserved; **exact O(n) taut-string / Condat direct solver**. Right for the piecewise-constant dims
    (dash period/duty, presence).
  - **ℓ0 / Potts 1-D (`pottslab`)** — `‖y−x‖² + λ‖Dx‖₀` → the *true* jump-count optimum (exact 1-D DP,
    O(n²)/O(n log n)); the sharpest edge preservation when the coeff genuinely steps.
  - **Robust smoothing spline (Whittaker-Henderson / Garcia DCT-robust, iteratively-reweighted / ℓ1 fit)** —
    down-weights the fit-jitter OUTLIERS while fitting a smooth spline; **automatic smoothing parameter via GCV**
    (Garcia 2010, `smoothn`); DCT-fast. A drop-in, parameter-free robust denoiser — good default before tuning λ.
- **Task-λ (where the granted rate is spent OPTIMALLY).** The smoothing strength λ is NOT a scalar window — set it
  **per-coefficient by the margin-saliency** `∂d_seg/∂coeff` (#141), at the KKT operating point
  `∂d_seg/∂byte = 25/(100·37.5M)`. Coeffs whose perturbation never moves the SegNet argmax past the R-downsample
  tolerance (~1–2.27 px) get **large λ** (aggressively smoothed, ~0 bits); coeffs on the codim-1 boundary annulus
  get **small λ** (preserved). Denoise **in the task metric**, not the geometric metric — the coding-for-machines
  discipline (`R_X(D;T)=R_Y(D;T)`; task-RD strictly beats reconstruction-error proxies). The moving-average window
  is the degenerate edge-blind single-λ special case.
- **Rate impact (DERIVED):** matches or beats the moving-average's −42% because it removes the *same* jitter but
  concentrates the residual into sparse knots (better for brotli/zigzag) AND, crucially, is **less lossy** (edges
  kept), so it clears more d_seg headroom for the same or fewer bytes.

### #4 — [POTENTIAL BEAT / unifier of #2+#3] Robust low-rank + sparse (RPCA / Principal Component Pursuit) on the tracked coefficient matrix

- **What.** Stack the tracked coeffs into a matrix `M` (600 frames × coeff-dims, per lane or block-diagonal
  across lanes). Decompose `M = L + S` via **Principal Component Pursuit** (Candès-Li-Ma-Wright 2011):
  **L** = low-rank (the smooth, few-DOF lane trajectory — this IS the FPCA/KLT structure R1 wanted), **S** = sparse
  (the swap-spikes + jitter outliers). One convex solve **separates the coherent trajectory from the swap/outlier
  spikes automatically** — the swap-spikes are *exactly* the sparse-large-deviation structure PCP is designed to
  isolate. Ship **L** as a rank-`r` factorization (`r×(600+dims)` numbers, tiny) + an optional sparse **S** for
  genuine events.
- **Why it's a candidate BEAT.** It **unifies #2 (batch smoothing = low-rank trajectory) and #3 (outlier removal =
  sparse S)** in ONE principled convex program, and it partially absorbs #1 (a swap that survives association lands
  in S rather than corrupting L). Elegant, one-shot, no per-track dynamics model to specify.
- **The catch (rule-118 + rendering).** The rank-`r` temporal/spatial basis is **video-derived → COUNTED** (ship
  the factor matrices). At this scale (`r`≈3–5) that is a small counted payload — likely net-positive, but MEASURE
  it, never assert (per the closed-form-CDF / bit-spend NO-FAKE discipline). And PCP does NOT by itself guarantee
  the *rendered* lane is correct at a swap frame the way explicit association does — so run it **after** #1, on the
  tracked matrix, as the joint smoother+outlier stage, not as a replacement for correspondence.
- **Rank order:** high-value ALTERNATIVE to #2+#3; probe it against the explicit RTS+ℓ1-trend pipeline and keep
  whichever measures lower net S. (`facebookarchive/robust-pca`, `dganguli/robust-pca`, `dfm794/pyrpca` for OSS.)

### #5 — openpilot **supercombo** as a temporally-COHERENT PRIOR / initializer / regularizer (the shortcut — with a sharp caveat)

- **The shortcut idea.** supercombo has a **recurrent state** → its lane-line / road-edge / path outputs are
  temporally COHERENT **by construction** (mean + std over 33 BEV timestamps). Run it OFFLINE on the source video
  (external CODE = FREE, rule-118), get a smooth per-frame lane prior, and **fit/regularize the analytic band to
  it** instead of to jittery per-frame fits. This is the *lane analog of the openpilot-ξ pose prior* — a coherent
  source estimated offline, only the compact coeffs shipped (counted).
- **The sharp caveat (do NOT use it as the TARGET).** Our d_seg target is the **GT SegNet class-1 argmax mask**
  (through R), NOT the physical road. supercombo predicts the *driving* lanes/path (correlated but NOT identical to
  SegNet's lane-marking argmax; it hallucinates lanes through occlusions, ignores paint the net doesn't care about,
  etc.). If we fit supercombo's output we optimize the WRONG distortion → d_seg spill. **Verdict: supercombo is a
  high-value PRIOR / INITIALIZATION / temporal-consistency REGULARIZER for the tracker+smoother (Tier B
  association affinity; RTS process model; a Bayesian prior mean), but the MEASUREMENT (the thing we fit and
  entropy-code) must remain the SegNet-mask fit.** Its predicted **std** is a free, per-point measurement-noise
  covariance for the Kalman/RTS stage — a genuinely useful gift. OSS: `commaai/openpilot`, the supercombo ONNX
  (`MTammvee/openpilot-supercombo-model`), `mbalesni/openpilot-pipeline` for output parsing.
- **Rank:** adopt as REGULARIZER/PRIOR, not as source. Medium-value, low-risk in that role; high-risk as target.

### #6 — Quantize + entropy backend on the coherent signal (keep the LBND2 backend; ONE change to reconsider a range coder)

- **CONFIRM the measured verdict.** Stage-1 measured **PTC1/constriction range-coder DOMINATED** brotli
  (43,153 B > 41,526 B) because per-dim transmitted-PMF headers (~66 dims × ~150 B) swamped the coded stream.
  **Brotli-on-zigzag-int32 stays the backend.** Do NOT re-add a per-dim range coder as gold-plating.
- **The one reconsideration.** After tracking+smoothing the residual is far sparser and more *homogeneous* across
  dims. A range/rANS coder with **ONE pooled global PMF** (or a handful of shared context classes: "centerline"
  vs "halfwidth" vs "dash") — instead of a per-dim PMF — amortizes the header over the whole stream and could beat
  brotli on the now-low-entropy innovations. This is the only backend change worth a MEASURED probe; default stays
  brotli until it wins byte-closed. (`bamler-lab/constriction`, already #152.)
- **Quantizer.** Keep the geometric-tolerance step (Stage-1) but graduate to the **task-λ / margin-saliency**
  quantizer of #3 (Lloyd-Max companding read in the `∂d_seg/∂coeff` metric) — same object as the task-λ denoiser.

### KILL / DEPRIORITIZE (honest negatives, implementation-level, reactivation criteria noted)

- **Ego-motion-compensated predictive coding (LBND3).** MEASURED negative (1.04–1.34×). Root cause: causal
  one-step prediction on a swap-corrupted, mostly-untouched (only c0/c1 advected) signal. **Reactivation:** only if
  a JOINT (batch, not causal) world-BEV fit with a *measured-reliable* ego trajectory beats the per-track camera-
  frame smoother — a higher-risk stretch (needs the ego estimate the lane axis already declined). Prefer #2.
- **Learned VQ / deep task-quantizer / CompressAI entropy-bottleneck as the primary codec.** Dominated at ~30k-
  scalar scale (learned codebook/prior = counted + parse-back complexity, not amortized). **Reactivation:** only if
  a measured post-tracking residual shows structured, non-Laplacian, high-entropy tails a parametric prior + RPCA
  can't reach.
- **DCT/KLT/FPCA temporal transform BEFORE correspondence.** Deprioritized not killed: it is CORRECT but only
  AFTER #1 (a transform of a swap-discontinuous series Gibbs-spreads). It is the natural realization of #2's
  low-rank/#4's L — keep it, just downstream of tracking.

---

## Is openpilot supercombo the shortcut? (direct answer)

**Partially, and specifically as a PRIOR — not as the source.** Yes, it hands us a free, offline, temporally-
coherent lane estimate WITH per-point uncertainty (the std outputs) that plugs directly into the tracker's
association affinity, the RTS measurement-noise covariance, and a Bayesian prior mean. That is a real accelerant.
**No**, we must not fit/entropy-code supercombo's output as the target, because our authority is the SegNet class-1
argmax mask through R, and supercombo optimizes a different (driving-lane) objective → fitting it would spill
d_seg. Use it to make the SegNet-mask tracking+smoothing more robust; keep the measurement = the SegNet-mask fit.

---

## The recommended OPTIMAL pipeline (composition, in order — each stage on the output of the prior)

```
per-frame SegNet-mask polynomial fits  (jittery + slot-swaps; LBND2 raw source)
  │
  ├─(1) CORRESPONDENCE  — global min-cost-flow / MHT-lite track assignment (Tier B; Tier A Hungarian fallback)
  │        → each slot = one physical lane for its lifetime; presence bitmap = track active-interval
  │        [LOSSLESS on geometry; kills the 44% swap mass; PRIOR to every temporal tool]
  │        (optional: supercombo prior as association affinity + a per-point std → measurement covariance)
  │
  ├─(2) BATCH SMOOTH   — per-track Kalman-RTS fixed-interval smoother (MMSE; fills occlusion gaps)
  │        [ OR (#4) RPCA/PCP on the tracked matrix: L=low-rank trajectory, S=sparse residual events ]
  │
  ├─(3) EDGE-PRESERVING R-D DENOISE — ℓ1-trend filter (piecewise-linear coeffs) + 1-D TV/Potts (dash/presence),
  │        λ set PER-COEFF by margin-saliency ∂d_seg/∂coeff at the KKT point  [task-RD; less-lossy than MA]
  │
  ├─(4) inter-line: ego-lane + lateral offsets (near-constant; MapTR ordered-point-set parameterization)
  │
  └─(5) LBND2 BACKEND — geometric/task-λ quantize → temporal-delta → zigzag → brotli   → archive.zip
                (probe ONE-pooled-PMF range coder vs brotli; keep the winner byte-closed)
```

**Rate arithmetic (targets).** rate `= 25·bytes/37,545,489`: +0.005 → 7,509 B · +0.01 → 15,018 B · **+0.016
(current best) → 24,027 B** · +0.02 → 30,036 B. DERIVED expectation: (1) alone recovers a large fraction of the
moving-average gain LOSSLESSLY; (1)+(2)+(3) should land **below 24 KB with LESS d_seg loss** than the moving-
average — plausibly toward the ~11 KB ego-swept-only residual, i.e. rate ~0.007–0.012. **All DERIVED; MEASURE.**

---

## d_seg-preservation argument (why this is *less lossy*, not just smaller)

1. **Correspondence is lossless on geometry** — a re-labeling never changes a rendered lane; it only makes the
   series codable. It even REMOVES a d_seg-spill failure mode (moving-average's phantom-lane-at-swap).
2. **RTS/PCP is MMSE** — the min-variance reconstruction; it removes noise, not signal, and interpolates
   occlusion gaps better than carry-forward hold (fewer erased-lane frames → better lane recall = better d_seg).
3. **Edge-preserving denoise keeps the edges** — ℓ1-trend/TV/Potts by construction preserve real slope-jumps
   (lane-changes, curve onsets) that the moving-average blurred; the λ is set so sub-tolerance precision (invisible
   to SegNet through R) is deleted but boundary-flipping precision is KEPT (task-RD).
The net: the coherent signal is a **higher-fidelity** rendering of the true lane geometry than the moving-average's
blur, at fewer bytes. **This is the exact "lower rate than 0.016 AND less-lossy" the operator asked for — DERIVED,
gated on the through-R n600 measurement.**

## What would BEAT this framing (adversarial; flagged honestly)

- **RPCA/PCP (#4)** could *replace* (2)+(3) with one convex solve and is the strongest single alternative — probe
  it head-to-head; keep whichever measures lower net S. (Its counted low-rank basis is the only rule-118 cost.)
- **A joint direct-to-BEV world-lane fit** (fit the physical road ONCE, project via ego homography) is the
  theoretically strongest "fit once" but hits (a) the SegNet-vs-road target mismatch and (b) the noisy-ego-estimate
  problem the lane axis already declined — higher risk; deprioritized behind the per-track camera-frame smoother.
- Nothing surveyed BEATS **correspondence-first**; it is a free-lunch prerequisite, not a competing method.

## Honest risks (adversarial)

1. **The d_seg-through-R n600 win is UNMEASURED byte-closed.** Even a perfect coherent code NETS negative if the
   band's realized d_seg improvement < rate cost. The whole point of driving rate toward ~0.007–0.012 is to make
   this reduce to "does the coherent analytic band lower d_seg AT ALL through R?" — **run that #205 gate before
   over-investing in the codec** (measurement-first, per ANTI-SIGNAL-LOSS).
2. **Association errors are a NEW failure mode.** A wrong global track assignment could *create* a swap the sort
   didn't. Mitigate: gate the global solver by a swap-detector; verify the tracked stream's top-5%-jump mass
   actually DROPS (a direct, cheap, n600 diagnostic that isolates the correspondence gain).
3. **Task-λ needs a STABLE `∂d_seg/∂coeff` across 600 pairs** (per R1 risk #2). Estimate through R on real n600,
   never a proxy. A robust-spline (auto-GCV, #3) is the parameter-free fallback if the saliency map is noisy.
4. **RPCA basis is COUNTED** (rule-118) — MEASURE the byte cost of the rank-`r` factors; never assert-free.
5. **Wave-E gates PRESERVED** — decoded coherent render must equal the training render bit-exact through R;
   default-off byte-identical stays 7/7; ALL of (1)–(5) are **compress-time offline estimators** feeding the
   UNCHANGED LBND2 decode → ZERO new inflate code (the same rule-118 property that made moving-average shippable).

## Concrete build spec (extends LBND2; offline estimator; deterministic decode)

- **Where it lives:** a new compress-time module `src/tac/boundary_math/lane_track_and_smooth.py`
  (`track_lane_slots(...)` global-flow assignment; `batch_smooth_tracks(...)` RTS / `rpca_tracks(...)`;
  `edge_preserving_denoise(..., task_lambda=...)`), producing a **re-ordered + denoised** `LaneLine` set that the
  EXISTING `serialize_lane_band_rd` (LBND2) consumes unchanged. It is a SOURCE PRE-TRANSFORM, exactly like
  `temporal_smooth_pairs_lines` in the ego memo — same ships-as-LBND2-bytes, ZERO-new-inflate-code property.
- **rule-118 accounting (binding, MEASURE each):** the tracking permutation + smoothing are FREE (generic offline
  algorithm; the decoder renders whatever coeffs each slot holds). COUNTED = the (smaller, coherent) quantized
  temporal-delta coeff stream + presence bitmap — same KIND as LBND2. If RPCA is used, the rank-`r` basis is
  COUNTED (byte-close it, don't assert). NO GT mask, NO scorer weights, NO supercombo weights ship.
- **Determinism:** all offline; the decode is the unchanged numpy LBND2 path (`_lane_parse_rd`), bit-exact,
  O(1)/pixel, host-portable, within the 30-min budget. Global-flow solver + RTS + ℓ1-trend are all deterministic
  (seed the LAP tie-break); the byte-exact gate is numpy-inflate == numpy-oracle (`max_abs_uint8_diff == 0`).
- **Measurement plan (the gate order):** (i) MEASURE tracked-only stream bytes @ n600 (isolate correspondence
  gain, verify top-5%-jump mass drops); (ii) add batch-smooth + task-λ denoise, MEASURE @ n600; (iii) probe RPCA
  vs RTS+ℓ1-trend; (iv) THEN the #205 trained-in d_seg-through-R leg (the real gate); (v) probe pooled-PMF range
  coder vs brotli only if (i)–(iv) net-positive. Every row byte-closed `[macOS-CPU advisory]`, MPS-never.

## Wire-in (6-hook, research_only)
1. Sensitivity-map: the per-stage Δrate rows (tracked-only / +smooth / +RPCA) → `tac.sensitivity_map` (rate axis) +
   the per-coeff `∂d_seg/∂coeff` task-λ IS a sensitivity-map consumer. 2. Pareto: λ (and rank `r`) are rate↔d_seg
   Pareto knobs (#205 measures the d_seg leg). 3. Bit-allocator: the task-λ denoiser is a source-denoise pre-pass
   to the LBND2 allocator (#157 reverse-waterfill). 4. Cathedral autopilot: N/A (research_only; feeds the existing
   byte-close LBND2 path, no new archive artifact). 5. Continual-learning: this memo is the anchor; the tracked/
   smoothed n600 rows become anchors when measured. 6. Probe-disambiguator: the measurement plan IS the
   disambiguator (correspondence-only vs +smooth vs RPCA vs supercombo-prior, resolved by measured n600 bytes +
   the #205 d_seg leg).

**Council mission-contribution:** `frontier_breaking` (the coherent-source rate half of the lane band, less-lossy)
— all MEANS; the END is the #205 byte-closed exact row. Pointer 0.19110 UNMOVED until then.

## Sources
- [ℓ1 Trend Filtering — Kim, Koh, Boyd (SIAM Review 2009)](https://web.stanford.edu/~boyd/papers/l1_trend_filter.html) — piecewise-linear edge-preserving denoise (#3 primary)
- [Adaptive Piecewise Polynomial Estimation via Trend Filtering — Tibshirani (Annals of Stat. 2014)](https://www.stat.cmu.edu/~ryantibs/papers/trendfilter.pdf) — polynomial trend filtering generalization
- [A Direct Algorithm for 1-D Total Variation Denoising — Condat (SPL 2013)](https://lcondat.github.io/publis/Condat-fast_TV-SPL-2013.pdf) + [taut-string O(n) impl](https://github.com/bgailleton/TVD_Condat2013) — exact 1-D TV (#3 piecewise-constant dims)
- [Robust smoothing of gridded data (smoothn) — Garcia 2010 (PMC4008475)](https://europepmc.org/api/getPdf?pmcid=PMC4008475) + [Whittaker-Henderson modern framework (arXiv 2306.06932)](https://arxiv.org/pdf/2306.06932) — auto-GCV robust smoother (#3 default)
- [Robust PCA / Principal Component Pursuit — low-rank + sparse review (arXiv 1511.01245)](https://arxiv.org/pdf/1511.01245) — swap-spike / outlier separation (#4)
- [Global Data Association for Multi-Object Tracking Using Network Flows — Zhang, Li, Nevatia (CVPR 2008)](http://vision.cse.psu.edu/courses/Tracking/vlpr12/lzhang_cvpr08global.pdf) — global min-cost-flow association (#1 Tier B)
- [Online MOT via Min-Cost Flow on a Temporal Window (MDPI 2023)](https://www.mdpi.com/2032-6653/14/9/243) — windowed global association
- [Recursive Video Lane Detection (RVLD) — Jin et al. (ICCV 2023, arXiv 2308.11106)](https://arxiv.org/abs/2308.11106) — causal one-frame warp (= our failed LBND3; the anti-pattern)
- [LaneTCA: Video Lane Detection with Temporal Context Aggregation (arXiv 2408.13852)](https://www.arxiv.org/pdf/2408.13852) — longer-window aggregation BEATS RVLD (validates batch>causal)
- [openpilot supercombo model outputs (MTammvee ONNX)](https://github.com/MTammvee/openpilot-supercombo-model) + [openpilot in 2021 — comma.ai](https://blog.comma.ai/openpilot-in-2021/) + [output parsing (mbalesni/openpilot-pipeline)](https://github.com/mbalesni/openpilot-pipeline) — coherent lane PRIOR + per-point std (#5)
- [Rauch–Tung–Striebel smoother — fixed-interval MMSE (Kalman filter, Wikipedia §Smoothing)](https://en.wikipedia.org/wiki/Kalman_filter) + [RTS smoother optimal state estimation](https://www.emergentmind.com/topics/rauch-tung-striebel-smoother) — batch smoother (#2)
- [constriction — range + rANS entropy coders (bamler-lab, #152)](https://github.com/bamler-lab/constriction) — pooled-PMF backend probe (#6)
- [Rate-Distortion Theory in Coding for Machines (arXiv 2305.17295)](https://arxiv.org/html/2305.17295v2) — task-RD `R_X(D;T)=R_Y(D;T)`, task-λ discipline (#3/#6)
- [MapTRv2 (arXiv 2308.05736)](https://arxiv.org/abs/2308.05736) — ordered-point-set lane parameterization (#4 inter-line)

## Sisters
`wave_f_lane_band_rd_code_LANDED_stage1_measured` · `wave_f_unified_xi_build_measured` (the ego-predict NEGATIVE +
smoothing POSITIVE this supersedes-by-refinement) · `wave_f_lane_band_rd_research_synthesis` (R1 backbone) ·
`analytic_lane_band_primary_authority_decomposition` · `project_contest_is_indirect_rate_distortion_task_space_coding` ·
`project_unified_variational_levelset_flow` (lanes = separatrices; tracking = persistence-across-time).
