# YOUSFI BLIND-SPOT PROBE B — verdict: "d_seg cheap by construction"

**Date:** 2026-06-17
**Authority:** `[contest-CPU advisory]` / `[macOS-CPU advisory]` — **NON-PROMOTABLE.**
The exact frontier pointer is UNMOVED. This is a DIAGNOSTIC of where the d_seg
lever lives, not a score claim. Only `upstream/evaluate.py` on a byte-closed
archive is authority.
**Probe:** `experiments/probe_yousfi_detector_cost_blindspot_b.py`
**Data:** `.omx/research/yousfi_detector_cost_blindspot_b_20260617T161257Z.json`
(24 real GT pairs, 4,718,592 SegNet-grid pixels, 11,649 real argmax-flips;
basin `torch_vehicle_full_mps_basin_bc20_n600/best/best_archive.bin`,
d_seg=0.00260, archive 89,136 B; real frozen SegNet on CPU; GT via
`frame_utils.yuv420_to_rgb`).

## The question

Yousfi inverse-steganalysis lens: d_seg is a DETECTOR (EfficientNet-B2 SegNet,
argmax-flip rate). Can we make d_seg cheap BY CONSTRUCTION — put the renderer's
error where the detector is BLIND (UNIWARD high-texture / high-margin) and keep
it inside the margin-polytope INTERIOR so the argmax never flips — instead of
grinding d_seg down with days of gradient descent?

## Headline verdict

**The "allocate error into the BLIND budget / stay in the interior" framing is
FALSIFIED for this basin — but for an instructive reason, and a DIFFERENT lever
(in-cell flip repair) IS real (~32% d_seg at $0 byte cost, advisory).**

The d_seg is NOT wasted in the interior waiting to be reclaimed. It is already
concentrated almost entirely on the SegNet class walls. There is no free
interior budget to "reallocate into" — the flips are detector-INTRINSIC, sitting
exactly where the detector is sharp. So the Yousfi *cost-map allocation* move
(push error toward boundary, away from texture) is the WRONG allocation and
makes d_seg worse. The move that works is the blunt, direct one: a per-pixel
in-cell repair (CE of the rendered SegNet logits toward the GT argmax) on the
flip pixels themselves.

## The four measurements

### M0 (emergent finding, the key to everything): the basin is a DETECTOR-MATCHER, not a reconstructor
Raw RGB error of the basin's rendered f1 vs GT f1 = **~105/255 mean-abs** while
d_seg = 0.0026. The frames look almost nothing like GT in pixels, yet their
SegNet argmax matches. The HNeRV renderer trained on the SegNet/Pose objective
memorizes the *detector's decision*, not the image. Consequence: "reconstruction
error vs cost map" is the wrong lever surface; the lever surface is WHERE THE
FLIPS LAND.

### M1 — detectability cost map
- SegNet margin: mean 5.57, median 5.79. Only **1.34%** of pixels have margin <
  0.5 (the sharp wall) and **2.59%** < 1.0. The detector is confident over
  ~97% of the frame.
- S-UNIWARD cost (texture capacity) and margin are essentially UNCORRELATED
  (Pearson −0.035). Texture-blindness and detector-confidence are different
  axes — a textured pixel is NOT reliably a high-margin pixel.

### M2 — error/flip allocation (the d_seg-relevant version)
- Raw recon error is ~FLAT vs margin (Pearson 0.062) — confirming M0 (renderer
  carries no recon signal to allocate).
- **Flip density is overwhelmingly at the boundary:** flip rate in margin<0.5
  pixels = **0.170** vs margin≥0.5 pixels = **0.00019** — an **882×** ratio.
  By margin quintile the flip rate is `[0.0123, 0, 0, 0, 0]` — ALL flips live
  in the lowest-margin quintile.
- Flip density vs UNIWARD cost is nearly flat (`[0.0020, 0.0024, 0.0025,
  0.0026, 0.0029]`) — texture does NOT predict flips. Down-weighting textured
  ("blind") pixels would do nothing useful.

### M3 — avoidable d_seg (cell free budget)
- 11,649 flips. **92.3%** are at hard-boundary pixels (GT margin < 0.5);
  flip-margin median = **0.137**.
- **0.0%** of flips are "avoidable by staying interior" (above median margin);
  0.0% in the deep-interior (top-quartile). `wasted_d_seg_fraction = 0.0`.
- **There is no wasted/avoidable d_seg.** Every flip is a genuinely hard,
  on-the-wall decision. The polytope-interior regularizer has nothing to
  reclaim — the renderer is already spending its capacity in the interior and
  paying d_seg only where the detector is intrinsically uncertain.

### M4 — $0 paired smoke (3 pairs, 50 steps, L∞=6, real argmax-flip d_seg)
| Regime | d_seg | Δ vs basin |
|---|---|---|
| basin baseline | 0.002572 | — |
| A. naive recon-to-GT | 0.003142 | **+0.00057 (HURTS)** |
| B. in-cell CE (uniform) | **0.001755** | **−0.00082 (−31.8%)** |
| C. in-cell CE, cost-weighted | 0.003045 | +0.00047 (HURTS) |

- **A hurting** confirms M0: pulling toward GT RGB breaks the detector match.
- **B (uniform in-cell CE) wins** — a direct per-pixel argmax repair on the
  flip pixels cuts d_seg ~32% at zero byte cost (advisory).
- **C losing to B (Δ +0.00129)** confirms M2/M3: cost-map allocation (toward
  boundary, away from texture) is the WRONG allocation. The flips are ALREADY
  all at the boundary, so re-weighting toward the boundary + suppressing the
  (irrelevant) texture term just dilutes the gradient on the pixels that matter.

## Honest answer to the brief

1. **Is "d_seg cheap by construction" (detector-cost allocation + in-cell error)
   a real lever beyond oomph?** The *detector-cost-allocation* half is **NO**
   (FALSIFIED on this basin): there is no blind/interior budget to exploit —
   100% of d_seg is detector-intrinsic at the class walls, and cost-weighting
   makes it worse. The *in-cell / stay-in-cell repair* half is **YES, real but
   modest and not free at scale**: a direct argmax-repair cuts ~32% d_seg
   (advisory), but only on the genuinely-hard boundary pixels.
2. **Cost-map stats:** margin mean 5.57 / median 5.79; only 1.3% of pixels at
   the sharp wall (margin<0.5); cost⊥margin (Pearson −0.035); renderer raw RGB
   err ~105/255 (detector-matcher).
3. **% of d_seg avoidable by staying interior:** **0.0%.** Flips are 92.3%
   hard-boundary (margin<0.5), flip-margin median 0.137. None are interior.
4. **Smoke Δd_seg:** best in-cell repair (B) **−31.8%** (0.002572 → 0.001755);
   naive recon (A) **+22%** (worse); cost-weighted (C) **+18%** (worse).

## What this means for the sub-0.15 path (means→ends)

- **The d_seg lever is NOT detector-cost allocation.** Confirms the standing
  memo finding that d_seg is capacity/detector-bound, not a placement artifact.
  Yousfi UNIWARD blind-spot exploitation does not apply: the renderer already
  hides all its (huge) RGB error in the detector's blind spots automatically —
  that is precisely why RGB err is 105 while d_seg is 0.0026. The remaining
  d_seg is the irreducible boundary residual, not exploitable slack.
- **The only real d_seg knob surfaced here is per-pixel in-cell argmax repair on
  the ~1.3% boundary-wall pixels.** A −32% advisory cut is meaningful (basin
  d_seg 0.00260 → ~0.00177 would move 100·d_seg from 0.260 to ~0.177, i.e.
  ~−0.083 raw S if byte-free). BUT it is NOT byte-free at scale: B here is a
  free per-pair compress-time delta, not byte-closed. To bank it the repair must
  be encoded — and the flips are sparse (~1.3% of pixels) AND boundary-located,
  which is exactly the regime a sparse boundary-residual sidecar
  (`tac.boundary_math` margin-conditional residual / contour codec) targets. The
  honest next step is: take the B-style in-cell repair, restrict it to the
  margin<0.5 flip set, and BYTE-CLOSE it through the existing sparse
  boundary-residual sidecar to measure ΔS-per-byte on the exact scorer.
- **It does NOT beat oomph by a different mechanism** — oomph's margin-weighted
  loss already concentrates gradient on the same boundary pixels B targets. B is
  the inflate-time/post-hoc version of what oomph does in training. The probe's
  contribution is the falsification of the *allocation/interior* framing + the
  quantification that 100% of d_seg is detector-intrinsic boundary residual.

## Reuse / NO-FAKE provenance
- Real frozen SegNet: `tac.score_aware_loop.targets.load_frozen_distortion_net(device='cpu')`.
- Real basin: parse-back of `best/best_archive.bin` via vendored `codec.parse_archive`
  + `model.HNeRVDecoder` (the contest-visible bytes; reproduced basin d_seg
  0.00242 on 5 pairs vs 0.00260 full — faithful).
- Real GT: `frame_utils.yuv420_to_rgb` (never PyAV rgb24).
- Real cost map: `tac.uniward_delta.compute_uniward_cost_map` (directional-Haar
  S-UNIWARD; no duplication).
- Exact d_seg pipeline: native(384×512) → bicubic camera(874×1164) → uint8 →
  SegNet preprocess (bilinear 384×512) → argmax-disagreement (vendored
  `score.evaluate_decoder` + `modules.SegNet.compute_distortion`, bit-faithful).
- CPU-only (MPS owns the live train; MPS would 2× corrupt SegNet d_seg).
