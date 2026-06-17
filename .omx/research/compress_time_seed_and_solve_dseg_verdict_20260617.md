# Compress-time SEED + SOLVE vs gradient descent for d_seg — DECISIVE PROBE VERDICT

**Date:** 2026-06-17
**Authority:** `[macOS-CPU advisory]` NON-PROMOTABLE — exact frozen contest SegNet on CPU (MPS
off; a live MPS train owned the GPU). Real basin ckpt READ-ONLY; GT via the vendored
`frame_utils.yuv420_to_rgb` path; exact d_seg = literal argmax-disagreement of the frozen SegNet
through the full eval roundtrip. No score claim; the exact pointer (0.19110) is UNMOVED.
**Scripts:** `experiments/probe_compress_time_seed_and_solve_dseg.py` (latent-solve) +
`experiments/probe_residual_pixel_solve_survival.py` (residual pixel-solve survival/byte).
**Artifacts:** `experiments/results/probe_seed_solve_dseg_{smoke,hard}_20260617.json` +
`experiments/results/probe_residual_pixel_solve_20260617.json`.

## The operator reframe under test

d_seg descends slowly because the CE/soft-cosine gradient vanishes at the residual boundary
flips. But d_seg=0 is a CONSTRAINT-SATISFACTION problem (per-pixel argmax-cell inequalities), and
we have full compress-time custody + known GT + a frozen locally-linear SegNet. So SOLVE it
(Jacobian / Gauss-Newton), don't descend. **Decisive question: does a compress-time SOLVE land
d_seg far lower, far FASTER than the 5-day gradient descent?**

The exact scored chain (verified 1:1 with `score.evaluate_decoder` + `modules.SegNet`):
`decoder(z) → f1(384×512) → bicubic↑(874×1164) → uint8 round → bilinear↓(384×512) → SegNet → argmax`.
The basin (`torch_vehicle_full_mps_basin_bc20_n600/best`, bc20, 28-d latents, 600 pairs) has a
faithfully-reproduced baseline exact d_seg **0.00239–0.00246** (≈ the trained-basin 0.00260).

## (a) Does the latent-SOLVE beat the trained latents' d_seg, and how fast?

**NO. The 28-dim latent-solve is essentially a no-op, and it is far SLOWER per unit d_seg.**

| solve | steps | lr | baseline d_seg | solved d_seg | rel. cut | cost |
|---|---|---|---|---|---|---|
| smoke (n=4) | 20 | 0.05 | 0.002387 | 0.002370 | **+0.7%** | 38 s/pair |
| hard (n=4) | 120 | 0.20 | 0.002387 | 0.002369 | **+0.75%** | 205 s/pair |

The hard solve is fully converged (boundary-weighted CE flat at 0.0069 from step 40 onward) — the
0.75% is the CEILING, not under-convergence. Only 1 of 4 pairs improved at all; 3 of 4 were
completely stuck. Scaled to the full basis: ~34 h CPU for the full 600 pairs to cut d_seg ~0.75%
(0.00260 → ~0.00258, negligible). The descent reached 0.00260 in 9.3 h and is **still descending**
on the live MPS run. **The solve is both worse in outcome and ~3.7× slower per unit d_seg.**

## (b) What is the latent-space expressiveness ceiling (fixed decoder)?

**~0.00237 — i.e. essentially the trained value.** The trained latents are already at the
argmax-cell optimum the fixed 28-dim decoder image can reach. The surrogate gradient does NOT
vanish (boundary-weighted CE has a real gradient), but moving the 28-dim latent to chase the few
hundred boundary-flip pixels per frame is geometrically impossible: a 28-number code cannot
re-paint the specific high-resolution argmax-boundary pixels where the flips live without
disturbing the (vastly more numerous) already-correct pixels. **The bottleneck is the decoder's
28-dim capacity, not the optimizer or the gradient.** This matches the standing finding that the
d_seg plateau at this basis is capacity/structure-bound, and that the score-optimal d_seg lever is
the byte-neutral *taper* reallocation + a richer basis, not more descent on the same 28-d code.

## (c) Does the residual pixel-SOLVE close the rest, and does it SURVIVE, at what byte cost?

**NO — it hits exactly the survival/byte wall the witness/store route hit, and worse: the
roundtrip makes the solved correction actively DESTRUCTIVE.**

The residual probe gives the per-pixel solve the FULL DOF (a free camera-res (3,874,1164)
perturbation — the upper bound on any cheap subspace), solved to flip the 1451 residual-flip
pixels (0.246% of pixels over 3 pairs) through the EXACT ↑/uint8/↓ chain:

| metric | value |
|---|---|
| baseline exact d_seg | 0.00246 |
| residual-solve **survived (roundtrip)** d_seg | **0.00246** (no improvement) |
| residual-solve ideal (no-roundtrip) d_seg | 0.04547 (**WORSE** than baseline) |
| smear loss (round − ideal) | −0.043 |
| camera int8 delta nonzeros | 6,742,611 |
| camera delta brotli bytes | **3,762,601 (3.76 MB)** |
| rate-term S contribution | **+2.5054** |

Two walls, both fatal:
1. **Survival/smear wall.** The perturbation is stored at one grid but the SegNet sees it only
   after bicubic↑ → uint8 round → bilinear↓. That chain decorrelates the stored delta from the
   scored argmax: the surrogate CE drops monotonically (1.06 → 0.078) while the exact d_seg
   *rises* every step (0.021 → 0.045). Optimizing the flip pixels through the smeared,
   quantized chain corrupts the (already-correct) majority pixels faster than it fixes the flips.
   The best the per-pair tracker ever found was the un-perturbed baseline.
2. **Byte wall.** Even the destructive solution costs 3.76 MB (rate term +2.5 S) — a self-evident
   non-starter. A sparse/low-rank cheap projection (the Dykstra `project_onto_cheap` set) would
   shrink bytes but cannot survive the smear that already defeats the *free* delta.

## (d) THE VERDICT

**"Seed + solve at compress time" is NOT a faster route to low d_seg than the descent. It is the
SAME wall the witness/store route hit, reached faster.** Concretely:

- The latent-SOLVE is a near-no-op (≤0.75% d_seg cut) and ~3.7× slower per unit d_seg than
  descent — the 28-dim fixed-decoder image is already at its argmax-cell optimum.
- The residual pixel-SOLVE does not survive the bicubic/uint8/bilinear roundtrip — the smear makes
  the solved correction actively raise exact d_seg, and the byte cost (+2.5 S) is prohibitive.
- This is the geometric reason: d_seg IS a constraint-satisfaction problem, but the feasible set
  is defined on the *scored argmax after the lossy roundtrip*, and the only cheap actuators
  (28-dim latent, storable residual) cannot place mass at the post-roundtrip boundary pixels
  without disturbing the correct majority. The frozen SegNet is locally linear *per pixel*, but
  the resample+quantize chain between the storable representation and the SegNet input is what
  breaks the preimage.

### What this de-risks / re-routes (system intelligence)

- **De-risks the long run:** do NOT pivot the long MPS descent to a compress-time latent-solve
  finisher — it would not help and would cost ~34 h CPU. The descent (which co-trains the decoder
  weights AND latents jointly) is escaping the very ceiling the fixed-decoder solve is trapped in:
  the only way to reach the post-roundtrip boundary pixels is to change the DECODER, not just the
  latent — which is exactly what descent does and the solve forbids.
- **Re-confirms the sub-0.15 d_seg lever** is structural, not solver-side: byte-neutral d_seg-aware
  taper reallocation (Catalog #121 waterfill) + a richer basis + FiLM-pose decoupling — per the
  standing "small-basis micro→macro audit" and "d_seg floor is loss-movable not capacity-bound"
  memos. The boundary-flip pixels are reachable only by the decoder's high-res capacity, which is
  what the taper feeds.
- **Seed angle (qualitative):** a frontier-class decoder (d_seg ~0.00056) would NOT make the solve
  trivial here — the wall is the roundtrip preimage on the *stored* representation, independent of
  the seed's quality. A better seed lowers d_seg by having a better decoder, i.e. by the same
  mechanism descent uses, not by making the fixed-decoder solve converge.

## 6-hook wire-in

1. Sensitivity-map: N/A (no new per-axis byte savings; the finding is a NEGATIVE on a solver route).
2. Pareto constraint: ACTIVE — adds the empirical constraint "fixed-decoder latent/residual solve
   cannot reduce d_seg below the trained value at acceptable bytes" (rules the route out of the
   d_seg actuator menu).
3. Bit-allocator: N/A.
4. Cathedral autopilot: N/A (research-only negative; non-promotable).
5. Continual-learning posterior: this memo + the probe JSONs are the durable anchor.
6. Probe-disambiguator: the two probe scripts ARE the disambiguator (latent-solve vs descent;
   residual-survival vs store). Verdict: descent (decoder+latent joint) dominates compress-time
   fixed-decoder solve.

**Mission:** `frontier_breaking_enabler` (negative that re-routes the d_seg attack to the
structural taper/basis lever and away from a dead solver path). Exact pointer UNMOVED (0.19110).
