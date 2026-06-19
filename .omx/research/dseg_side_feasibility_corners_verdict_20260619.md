---
title: "d_seg-SIDE feasibility — the two untested closed-form corners for #155 (sub-pixel camera-res placement + cross-frame keyframe-warp)"
authority: "[contest-CPU advisory] NON-PROMOTABLE — pointer UNMOVED 0.19110; $0; CPU authority; no PR"
score_claim: false
promotion_eligible: false
ready_for_exact_eval_dispatch: false
pointer_moved: false
date: 2026-06-19
verdict: RED_NEITHER_CORNER_BEATS_THE_FRONTIER_DSEG_CODING
producer: experiments/probe_dseg_side_feasibility_corners.py
result_json: .omx/research/dseg_side_feasibility_corners_20260619T175221Z.json
cross_refs:
  - .omx/research/recursive_adversarial_review_recent_negatives_20260619T024605Z.md   # the META corners 1+2 this gate tests
  - .omx/research/eval_roundtrip_deep_math_pr95_handling_and_exploits_20260619.md      # §3 texture wall, the exact operators
  - .omx/research/p_suff_task_ablation_verdict_20260619.md                              # the frontier-near-floor bar #155 must beat
  - .omx/research/comma_openpilot_domain_tricks_20260619T035417Z.md                     # the cross-frame drift + homography premise
  - experiments/probe_curve_core_dseg_feasibility_gate.py                               # the reused roundtrip/realized-d_seg/GT-L* harness
---

# d_seg-side feasibility — the two untested closed-form corners (#149, #148)

The recursive adversarial review (`recursive_adversarial_review_recent_negatives`) surfaced TWO
untested assumption-corners the d_seg pincer never measured. Both violate "per-frame, render at
384" but stay inside the legal RGB-frame eval. This $0 CPU probe measures both on the REAL frozen
SegNet through the EXACT eval chain, recomputes S from components, and is guarded against BOTH
false-GREEN (degenerate fit) AND false-RED (under-powered solve). **Pointer UNMOVED 0.19110.**

**The bar #155 must beat (p_suff RED):** the 0.19110 frontier sits near its task-RD floor. Its
breakdown is **rate 0.11797 + d_seg 0.056 (d_seg=0.00056) + d_pose 0.01715**. A from-scratch
d_seg-core must code the d_seg-critical boundary so that `100·d_seg + (d_seg-coding-rate)` beats
the frontier's d_seg contribution.

## What was measured (NO-FAKE, measurement-first)

- **Real frozen contest SegNet, CPU authority** (NEVER MPS for the score). The cached L* IS
  `D(GT_camera)→SegNet→argmax` — verified `match==1.0` (the harness is faithful). The MPS path,
  where used, is the sanctioned GRADIENT-only device; every d_seg number is the CPU-authority
  argmax-flip-rate of the HARD uint8 frame vs L*.
- **The exact eval geometry (source-confirmed):** the eval loads the recon as raw uint8
  **camera-res 874×1164** (`evaluate.py`, `TensorVideoDataset`) and SegNet's `preprocess_input`
  applies **D = bilinear-down to 384×512** (`modules.py:113`). There is **no bicubic-up in the
  eval** — the up is the inflate decoder's own choice. So a camera-res representation skips
  up-then-down entirely; SegNet sees `D(camera_uint8)` directly.
- n = 3 consecutive GT frames (honest: these are near-identical dashcam frames ~0.1 s apart;
  n≈1 effective per the review's generalization caveat — but a wall on easy near-identical frames
  only worsens on the contest's 600 diverse pairs, so this makes a RED MORE trustworthy).

## CORNER 1 (#149) — camera-res sub-pixel boundary placement → **RED** (but a real, sharp finding)

The hypothesis: at camera-res the boundary has ~3× the pixels and can be ANTI-ALIASED sub-pixel so
D's weighted average lands the argmax on the correct side BEFORE D averages — beating the 384-grid
flat-paint wall (bnd_flip ~0.18). Measured three paths through the real SegNet, reporting the BEST
(false-RED guard):

| path | d_seg (avg n=3) | bnd_flip | note |
|---|---|---|---|
| flat @ 384 → up→Q→down (old gate path) | 0.0191 | 0.176 | the wall the prior gates hit |
| flat @ camera-res (eval-native D, no up) | 0.0185 | 0.178 | camera-native barely helps flat |
| **sub-pixel SOLVED (grad through D+SegNet)** | **0.0015** (best 0.00097) | **0.04** | **12× boundary-flip win over the 384 wall** |
| sub-pixel best-shot (1200 iters, wide band) | **0.00094 (1.7× frontier)** | 0.022 | converges toward, never below, frontier |

**The sub-pixel lever WORKS** — it is far more powerful than flat-paint, dropping the boundary
flip 0.18 → 0.04 (12×) and realized d_seg ~0.019 → ~0.0015 (and to 0.00094 at 1200 iters). This
**confirms the camera-res sub-pixel corner is a genuine d_seg lever the prior 384-grid gates
missed.** BUT it is RED on the #155 question for two independent, false-RED-guarded reasons:

1. **It plateaus ABOVE the frontier d_seg.** Best-shot d_seg 0.00094 = **1.7× the frontier's
   0.00056**, and it converges toward (never below) it. The §3 texture-dependence wall persists:
   sub-pixel placement fixes the 1px boundary band, but the flat-color interior loses the texture
   evidence SegNet uses for argmax further from the line, so d_seg floors above frontier. The
   **d_seg TERM ALONE is 0.094–0.15** (at d_seg 0.00094–0.0015) vs the frontier's **0.056** —
   **#155-Corner-1 loses on the pure d_seg axis EVEN IF the boundary code were free.**
2. **The byte cost is catastrophic.** The sub-pixel win requires storing the anti-aliased
   boundary-band pixel values at **camera resolution: ~68.6 KB/frame** (76,234 band camera-px × 3
   × 8 b × 0.30 packing). Even the absolute floor (band residual FREE, contour 914 B quasi-static)
   gives rate 0.037; with the real residual the amortized rate is **2.78** → projected **S ≈ 3.0**.
   Corner 2 PROVES the band can't be warped frame-to-frame, so this residual is per-frame, not
   amortizable.

**Corner-1 verdict: RED_C1_SUBPIXEL_DOES_NOT_BEAT_FRONTIER_DSEG.** The mechanism finding is real
and reusable: sub-pixel camera-res placement is a legitimate boundary-band lever (12× over the 384
wall) but (a) it can't break below the frontier d_seg because the wall is interior texture-keying,
not the 1px band, and (b) it pays camera-res band-residual bytes that explode the rate.

## CORNER 2 (#148) — cross-frame keyframe + tiny warp → **RED** (robust geometric premise failure)

The hypothesis: code ONE keyframe boundary (914 B) + a tiny per-frame warp; pay the boundary ONCE
and amortize d_seg across 600 frames at near-zero per-frame bytes. Measured the warped keyframe
partition's realized d_seg through the real SegNet vs each target frame's L*:

| quantity | value (avg) | note |
|---|---|---|
| cross-frame combinatorial drift (no warp) | 0.0126–0.0134 | consecutive L* drift (non-overlapping pairs ~0.1 s) |
| affine warp combinatorial residual | 0.0112 | **warp closes only ~10–15% of the drift** |
| translate warp combinatorial residual | 0.0119 | even less |
| realized d_seg (keyframe + affine warp) | **0.0253 (45× frontier)** | WORSE than per-frame own-flat |
| realized d_seg (per-frame own-flat) | 0.0107 | amortization HURTS by −0.014 |

**The geometric premise fails robustly.** The per-frame change in the d_seg-critical boundary is
**NOT a low-DOF rigid warp** — it's scene-content change (new objects, road perspective
foreshortening as the car advances). A translate / affine / longer-baseline warp closes <40% of
the partition drift (false-RED-guarded: tested translate, affine, and frame0→frame8). And using
the warped keyframe's colors against a different frame's L* compounds the partition mismatch with
the §3 texture wall, so realized d_seg (0.025) is WORSE than coding each frame's flat partition
from scratch (0.011). The amortization is a net loss.

**Corner-2 verdict: RED_C2_KEYFRAME_WARP_DOES_NOT_AMORTIZE_DSEG_TO_FRONTIER.** Note: the frontier
DOES amortize across frames — but via a per-pair 28-d latent on a learned JOINT decoder that paints
continuous texture, NOT a geometric warp of a stored boundary. A stored-boundary-plus-warp cannot
reproduce that.

## OVERALL d_seg-side feasibility for #155: **RED_NEITHER_CORNER_BEATS_THE_FRONTIER_DSEG_CODING**

Neither untested corner gives #155 a d_seg-core that beats the frontier. Combined with the four
prior structural REDs (curve, flat-NCA, factored-LF, eval-roundtrip-math), this closes the two
geometries the pincer had NOT tested:

- The **frontier's d_seg coding is near-minimal on this axis too.** Its d_seg=0.00056 at term
  0.056 is below what a from-scratch flat/sub-pixel/warp d_seg-core can reach byte-cheaply. The
  binding wall is the same one §3 identified: SegNet's argmax depends on **interior texture
  evidence**, not just the boundary geometry — so a representation that paints flat-per-class
  colors (however cleverly placed or amortized) floors at d_seg ~0.001–0.02, 2–45× the frontier.
- The **only known way below the frontier d_seg is the frontier's own move:** a learned decoder
  that paints continuous texture keeping `s_recon` inside the per-pixel argmax polytope. That is
  capacity-walled (factored-LF RED) and is exactly the generative-continuous-texture axis the
  campaign already runs — not a closed-form sub-pixel or warp shortcut.

## Honest caveats (NO-FAKE)

- **n=3 near-identical frames** (review's generalization caveat). A RED on easy frames is a
  conservative (more-trustworthy) RED; a GREEN here would have needed the diverse-600 re-test.
  No GREEN was found, so this caveat strengthens the verdict.
- **Sub-pixel solve is well-powered** (false-RED guard): pushed to 1200 iters + wide band; it
  reaches 1.7× frontier and converges there, not below. The RED is not an under-powered solve.
- **Byte accounting is advisory** (0.30 entropy factor on the band residual; contour 914 B
  measured). The d_seg-TERM-alone argument (Corner 1 loses even with FREE bytes) is
  byte-accounting-independent, so the verdict does not hinge on the packing factor.
- **The warp is a real geometry solve** (coarse-to-fine translate + Nelder-Mead affine,
  converging) on the combinatorial label-diff — verified it recovers a synthetic 8px shift in the
  test suite. The <15%-closure is the real cross-frame structure, not a weak solver.
- No exact-eval row produced — this is a $0 diagnostic that REDIRECTS #155 (away from closed-form
  d_seg-core shortcuts, toward the learned continuous-texture axis), it does not move the score.

## What this redirects (the system-intelligence wire-in)

- **#155 (from-scratch task-space rep):** its d_seg-core cannot be a closed-form flat/sub-pixel/
  warp boundary code — those are 2–45× the frontier. If #155 proceeds it must carry a learned
  continuous-texture renderer (the generative axis), inheriting the factored-LF capacity wall — OR
  target the RATE/d_pose axes, not d_seg.
- **Reusable mechanism for the bit-allocator (#154):** sub-pixel camera-res placement IS a real
  boundary-band d_seg lever (12× over the 384 wall) — if a future vehicle already pays for a
  textured frame, anti-aliasing its boundary band at camera-res before the inflate-up is a cheap
  d_seg top-up (the §5.3 deconvolution sister, now measured to help the band specifically). It is
  a top-up, not a from-scratch d_seg-core.

Pointer UNMOVED 0.19110.
