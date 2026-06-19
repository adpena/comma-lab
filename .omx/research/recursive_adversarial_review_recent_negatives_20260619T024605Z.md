---
title: Recursive adversarial review of the campaign's recent negatives (the PINCER REDs + the AMBER + the int5/capacity caps)
authority: "[review/advisory] — NON-PROMOTABLE; exact pointer UNMOVED at 0.19110"
score_claim: false
promotable: false
frontier_pointer_moved: false
mission_contribution: rigor_overhead
reviewer_vs_author: separated (adversarial, FALSIFY-first; reviewer is not the author of any finding below)
date: 2026-06-19
verdict_summary: "4 of 5 findings CONFIRMED-SOUND (with caveats); 1 (int5 Path-B) UNDER-POWERED-RE-TEST; the AMBER label is OVER-CLAIMED relative to the probe's reproducible verdict; 2 untested assumption-corners surfaced."
cross_refs:
  - .omx/research/curve_core_gate_RED_survival_wall_and_the_pincer_20260618.md
  - .omx/research/generative_axis_nca_dseg_core_gate_20260619T013000Z.md
  - .omx/research/generative_axis_continuous_texture_nca_AMBER_20260619T020000Z.md
  - .omx/research/eval_roundtrip_deep_math_pr95_handling_and_exploits_20260619.md
  - .omx/research/factored_lf_core_capacity_gate_20260618T233940Z.md
  - .omx/research/frontier_int5_score_aware_qat_finetune_20260618T211958Z.md
  - .omx/research/polynomial_fill_survival_gate_AMBER_boundary_band_wall_20260619.md
  - .omx/research/SESSION_SYNTHESIS_SoT_20260617_20260618.md
---

# Recursive adversarial review of the recent negatives

Operator directive 2026-06-19: review all math/algebra/geometry/calculus/engineering/implementations,
falsify the negatives, no surprises, down to the bare math and metal. $0, read/CPU only, NO MPS,
running daemons untouched. Pointer UNMOVED 0.19110.

**The job was to BREAK the REDs.** I could not break the four structural REDs (curve, flat-NCA,
factored-LF, eval-roundtrip math) on the d_seg-axis conclusion — they survive adversarial scrutiny,
though each carries a documentation or extrapolation caveat below. I DID find: (1) the int5 Path-B
"structural cap" is **under-powered** (per-tensor symmetric quant only; no per-channel/LSQ — the exact
canonical low-bit fixes CLAUDE.md names); (2) the continuous-texture **AMBER label over-claims** relative
to its own probe's reproducible verdict (RED collapse); (3) the curve probe's docstring **mis-describes its
mechanism** (geometry is frozen, not "fit through the scorer") though the verdict survives; (4) two
**untested assumption-corners** the pincer never tested.

## The per-finding table

| # | finding | NO-FAKE | best-shot | bare-math | over-claim | OVERALL + fix |
|---|---|---|---|---|---|---|
| 1 | **Curve-core RED** (survival wall, S≈0.74) | REAL (real frozen SegNet CPU-authority, real GT L*, exact uint8 roundtrip; tests would fail on `return constants`) | adequate FOR THE VERDICT, but the docstring claims the **geometry is optimized through the scorer — it is NOT** (only 30 colour+band params are; geometry is frozen Douglas-Peucker of L*). Moot: the wall is the flat-paint→SegNet step (geo_seg−geo_recon=0.00562), measured BEFORE the roundtrip, after colour-training — geometry motion can't fix texture-keying on a flat fill. | S=0.736 recomputed from components ✓; decomposition matches the eval-math memo (roundtrip +0.00005, texture-gap +0.00562) ✓ | mechanism over-claim in docstring (curves "fit through scorer"); SUPERSEDED by the eval-math §3 correction (texture-dependence, not resize) | **CONFIRMED-SOUND** (verdict robust). Fix: correct the probe docstring + the curve memo's "curves AND colours optimized through the roundtrip" line — only colours/offsets are; geometry is frozen. |
| 2 | **Flat-partition NCA RED** (~0.02, both walls) | REAL (reuses curve gate's exact machinery; 15 NO-FAKE tests; iteration genuinely changes logits) | adequate; the gate honestly WITHDREW its over-stated "all 4 families capped" framing per the sister SoT correction | best-frame 0.0162 / avg inflated by per-frame collapse; S=2.10 plausible | the original "terminal finding" framing was over-claimed and is now WITHDRAWN in-memo (good) | **CONFIRMED-SOUND** for the flat-partition representation. No fix; the WITHDRAW already landed. |
| 3 | **eval-roundtrip deep math** (operators, null-space, PR95, §3 texture correction) | REAL/source-grounded: every operator I re-derived against `upstream/{evaluate,modules,frame_utils}.py` checks out | n/a (analysis) | SegNet last-frame argmax-flip ✓ (modules.py:108-113); PoseNet yuv6@192×256, (x−127.5)/63.75, out[:6] MSE ✓ (modules.py:64-65,84); GT camera-native uint8 vs recon up→Q→down asymmetry ✓; D applied inside preprocess to BOTH ✓; B0=37,545,489 ✓; ker(D)~820k/ch order-of-magnitude OK | §1 says "decoder PR95 sigmoid·255" + "STE-round training" are RECONSTRUCTED not source-read — **honestly flagged** (PR95 train loop is opaque .pyc); the "R_lin near-identity" eigenvalue claim is asserted, not measured (low EV to verify) | **CONFIRMED-SOUND**. One nit: §1 calls the SegNet input size "(512,384)" then "@384×512" — both refer to (H=384,W=512); upstream `segnet_model_input_size=(W,H)=(512,384)` and interpolate uses `(size[1],size[0])=(384,512)`. Consistent, just notation-dense. |
| 4 | **factored-LF capacity wall** (d_seg∼29.3·params^−0.71) | REAL (narrow HNeRV cores, real SegNet, CE-through-roundtrip, EMA-warmup-shadow; degenerate-fit false-GREEN was CORRECTED to measurement-first RED — a NO-FAKE save) | adequate at the 2 measured points; **but the power-law is a 2-POINT fit** (bc8=20K, bc12=36.5K) extrapolated to **10.7M params (128× basin)**. A 2-point line through a glassy/stretched-exp process is a weak extrapolant over 500×. | k=0.71, A=29.3 reproduced exactly from the 2 cores ✓; S_bc12=1.79 ✓; the *verdict* (small core walls WORSE per-byte than dense) needs no extrapolation — bc8/bc12 both measured 6.6–10.1× above the bc20 wall | the precise "needs 10.7M params" is an over-precise 2-point extrapolation; the QUALITATIVE wall (small≠low) is measured, not extrapolated | **CONFIRMED-SOUND** on the qualitative wall; the 10.7M figure is an order-of-magnitude indication only (label it ±a decade). Cheap re-test: add ONE more core (bc16) to make it a 3-point fit and bound k. |
| 5 | **int5 Path-B QAT cap** (d_seg plateaus ~0.0035, S≈0.48) | REAL (full-600 byte-closed CPU-authority eval; CE-vs-margin-hinge disambiguation is a genuine, valuable sub-finding) | **UNDER-POWERED.** The quant is **per-tensor symmetric** (`scale=abs().max()/n_levels`), **no per-channel scaling, no LSQ learnable step, no outlier handling, no GPTQ/AWQ calibration** — the exact canonical low-bit-PTQ/QAT fixes CLAUDE.md's "Forbidden premature KILL" lists. abs-max per-tensor is outlier-dominated → crushes most channels; per-channel routinely recovers 2-4× on coarse grids. | S=0.475 recomputed ✓; the int5 rate win (177K→118.6K B) is real ✓ | "the residual d_seg cap (~0.0035) is STRUCTURAL to the int5 grid" — **OVER-CLAIMED**: structural-to-THIS-quantizer (per-tensor abs-max), not proven structural to int5. | **UNDER-POWERED-RE-TEST-NEEDED.** The route (own-vehicle) is reasonable regardless, but soften "STRUCTURAL" → "structural to per-tensor abs-max int5." $0 re-test: per-channel scales + LSQ step on the int5 coarse stages, same CE arm; if d_seg still plateaus ≥0.002, THEN it's structural. |

## The AMBER (continuous-texture NCA) — a separate, sharper verdict

The AMBER is the one finding the prompt's checklist most needs adversarial scrutiny, because it is the
only one the campaign treats as a live sub-0.15 path.

- **NO-FAKE: the measurements are REAL.** The 0.00337 (T015100Z JSON, recon_rmse 14.4) and the 0.00505
  (gate_state h64 frame2) genuinely happened on converged runs; the readout is C→3 continuous RGB (11
  tests verify >5 distinct values, not a partition); CPU-authority d_seg. Not fabricated.
- **OVER-CLAIM (the headline): the AMBER label exceeds the probe's reproducible verdict.** The probe's
  OWN latest finalized JSON (T022855Z, n_frames=1 repro) returns **`RED_TEXTURE_NCA_CAPS_ABOVE_SUB015`,
  best_realized 0.5486** — the collapse. The robust daemon sweep in gate_state: h128 frames = [0.507,
  0.550, 0.509] (ALL collapsed); h64 = [0.549, 0.550, 0.0051] (1 of 3). **~2 of 8 runs converge.** The
  AMBER memo is COMMENDABLY honest about this in §5/§6 (the repro-collapse table is right there, and §6
  flags the rate-amortization is untested at n=1) — so this is NOT a fake; it is an **over-claim of the
  title/label**: the reproducible/typical result is the RED collapse, and the AMBER rests on the lucky
  ~25% tail.
- **OVER-CLAIM (the mechanism it "overturns"): asymmetric robustness.** The AMBER claims it OVERTURNS the
  polynomial probe's "representation-independent boundary wall ~0.15" by cutting bnd_flip to 0.079. But
  the 0.079 came from the **lucky single converged frame**, while the polynomial ~0.15 is **robust**
  (closed-form LS, n=3, all k=0..10, sub-1-minute, no MPS, no convergence luck). A non-reproducible spark
  does not overturn a robust closed-form result. The honest statement: "ONE converged run hit bnd_flip
  0.079; whether that reproduces is the open question." The continuity THESIS (continuous interior solves
  to ~0.00005, confirmed independently by the polynomial probe) IS robust and IS the real signal.
- **BARE-MATH:** S=0.4144 from the 0.00337 row ✓; the sub-0.15 projection (dseg 0.0007, rate 0.013-0.02
  → S 0.141-0.148) is arithmetically correct ✓ **but rests on TWO untested assumptions** (boundary cut 5×
  AND shared-rule amortization over 600 frames) — the memo says exactly this. Measured S = 0.415, n_sub015
  rows = 0.

**AMBER verdict: OVER-CLAIMED (label), CONFIRMED-honest (body).** The body documents every gap; the
title/SoT-row should read **"AMBER spark — ~2/8 converge, typical result is RED collapse; continuity
thesis robust, amortization untested"** not "strongest sub-0.15 candidate." The running daemon (PID 8980,
n_frames=3, recon-w 2.0) is the right next test; do NOT promote the AMBER until it converges reliably.

## A generalization concern that touches findings 1, 2, 5-sister, and the AMBER (NEW)

The curve / flat-NCA / texture-NCA / polynomial gates all measure on the **first 3 frames of
`gt_targets_n16.pt`**, which are **3 consecutive frames of one dashcam segment** (class fractions
frame0/1/2 ≈ {road .49, class4 .26, class0 .23, lane .007} — near-identical). **n=3 here is effectively
n≈1**: not 3 independent samples. The contest averages 600 pairs across diverse scenes. This makes the
REDs MORE trustworthy (a wall on easy near-identical frames only worsens on diverse scenes) but makes the
AMBER's optimism TRIPLY suspect (lucky frame × similar frames × untested amortization). Cheap fix for any
future GREEN-seeking gate: sample non-consecutive frames spanning ≥2 videos.

## META — assumption-violations the pincer has NOT tested (the assumption-challenge axis)

The pincer frames sub-0.15-d_seg as: {flat-region: survival-walled} ∩ {continuous-texture one-shot:
capacity-walled} ⇒ only generative-iterated-continuous escapes. Two hidden shared assumptions, each a
potential corner the campaign never tested:

1. **"d_seg must be paid per-frame from a per-frame representation."** Every gate renders ONE frame's L*
   from that frame's representation. But d_seg is computed on the **last frame of each pair**, and
   consecutive last-frames are ~identical (the geometric-solve probe measured 0.33px contour drift). A
   **cross-frame / temporal-residual** representation — store ONE high-fidelity keyframe's boundary
   continuously + tiny per-frame warps — was NEVER gated for d_seg. The frontier decoder already exploits
   this (per-pair 28-d latents on a shared decoder). The d_seg-core gates all tested the HARDER
   per-frame-from-scratch problem. **Untested corner: does a shared-keyframe + warp representation reach
   GREEN d_seg at byte-cheap, sidestepping BOTH walls because the expensive boundary is paid ONCE?**
   (This is adjacent to the AMBER's shared-rule bet but more direct — a stored continuous keyframe, not a
   grown rule.)

2. **"The decoder must render an RGB frame through the survival-lossy roundtrip."** The eval loads the
   recon as **raw uint8 camera-res** and applies D (bilinear→384) inside the SegNet preprocess. Every
   gate renders at 384 then up→Q→down. But the recon is stored at **camera-res 874×1164** — a gate could
   render the d_seg-critical boundary band **directly at camera-res with sub-pixel-accurate boundary
   placement** so that D (the 2.28× downsample) lands the argmax on the correct side, instead of rendering
   at 384 and eating the up-then-down blur. The polynomial memo §next-move names this ("sub-pixel
   boundary placement at camera-res, untested") but NO gate has measured it. **Untested corner: does a
   camera-res sub-pixel boundary beat the bnd_flip ~0.15 wall the 384-grid gates all hit?** The boundary
   has 3× the pixels at camera-res — the argmax-flip is SET there, before D.

Both corners are inside the "render an RGB frame" frame but violate "per-frame, at 384." Neither is a
challenge to the frozen-frame assumption itself (the eval IS an RGB-frame scorer — there is no legal
non-frame path; the recon must be a raw uint8 frame per `TensorVideoDataset`), so I do NOT claim a
non-RGB escape exists. The two corners above are the real untested geometry.

## Prioritized re-test list (which negative most deserves a best-shot re-run)

1. **[HIGHEST, cheap, $0] int5 Path-B with per-channel + LSQ** — the ONLY finding that is genuinely
   under-powered vs CLAUDE.md's named canonical fixes. Per-tensor abs-max is the weakest possible
   quantizer; per-channel + LSQ on the int5 coarse stages, CE arm, full-600 byte-closed. If d_seg still
   plateaus ≥0.002, the "structural" claim earns its word; if it drops toward 0.0015, Path-B reopens as a
   rate lever. Decisive, fast, and it directly tests a load-bearing "STRUCTURAL" claim.
2. **[HIGH, cheap, $0] camera-res sub-pixel boundary gate** (META corner 2) — the polynomial probe
   localized the ENTIRE remaining d_seg deficit to the 1px boundary band at 384; measure whether placing
   the boundary at camera-res sub-pixel beats bnd_flip 0.15. If yes, S→~0.10 with the already-solved
   interior. This is the sharpest unwalled corner and it does NOT need MPS/convergence luck (closed-form,
   like the polynomial gate).
3. **[HIGH, cheap, $0] shared-keyframe + warp d_seg gate** (META corner 1) — directly tests the
   amortization the AMBER only assumes, but with a stored continuous keyframe (no NCA convergence
   fragility). Cleaner than waiting for the NCA to converge reliably.
4. **[MED] the running AMBER daemon (PID 8980)** — already testing n_frames=3 robustness; let it finish,
   read the verdict, do NOT promote until ≥reliable convergence. This is the amortization/robustness test
   the AMBER named; no new launch needed.
5. **[LOW] factored-LF 3rd core (bc16)** — only to bound the 2-point power-law's k; the qualitative wall
   is already measured, so this is rigor polish, not a reopener.

## NO-FAKE bottom line
None of the five findings is a fake implementation (class 1-8). All use the real frozen scorers on real
GT through the exact roundtrip; the verdicts are measurement-first; two probes (factored-LF degenerate-fit,
lane-geometric JSON label) had auto-label bugs that were CAUGHT and corrected — evidence the NO-FAKE
discipline is working, not failing. The corrections needed are: one under-powered re-test (int5), two
documentation/label softenings (curve docstring "geometry fit through scorer"; AMBER title "strongest
candidate" → "fragile spark"), one extrapolation-precision caveat (factored 10.7M = ±a decade), and the
two untested assumption-corners above. The pincer's d_seg-axis logic holds for the representations
tested; it is NOT airtight across the two untested geometries (cross-frame, camera-res sub-pixel).
Exact pointer UNMOVED at 0.19110.
