---
name: Adaptive Formula Errors + Correct Understanding (CRITICAL UPDATE)
description: T² cancels in the formula — sw should be static. The 1.33 winner used sw=100 (default), not sw=30.
type: feedback
originSessionId: ecc348c1-2829-48d3-9d4c-10ed2aa188a8
---
## What Went Wrong

The formula w_s*(p,T) = 20*sqrt(p/0.1)/T² has T² in BOTH the formula and the KL loss.
They cancel: effective gradient ∝ sw * T² * (1/T²) = sw * const.
The formula is temperature-independent. The "invariant" w_s*T² is trivially constant.

## Key Factual Correction

The 1.33 authoritative winner used **sw=100** (CLI default in train_tac.py), NOT sw=30.
The modal_dilated_kl_hardframe_deploy.py does not pass --segnet-loss-weight, so it
defaulted to 100. The sw=30 in COUNCIL_V1 was never verified authoritatively.

## What Should Be Static vs Adaptive

**STATIC (correct):**
- segnet_loss_weight: 35-100 range works. sw=100 is empirically validated.
- Temperature schedule is already handled by T² inside the KL loss.

**ADAPTIVE (still worth pursuing):**
- boundary_weight: increase as T drops (boundary_anneal). The 1.33 run used bw=1.0 (default).
  Higher bw could improve SegNet further. This IS theoretically justified.
- hard_frame_ratio ramp: 0.1→target over training. Already implemented.
- error_replay_every: more frequent in later epochs.
- eval frequency: more frequent near convergence.

## Principled sw Derivation

At operating point p=0.002:
- PoseNet gradient magnitude: 5/sqrt(10*0.002) ≈ 35
- SegNet KL gradient magnitude (after T² scaling): ~1.0
- Equal gradient contribution: sw ≈ 35

sw=100 means SegNet is ~3x over-weighted relative to equal gradient contribution.
This works because SegNet contributes 4.4x more to the score than PoseNet at this
operating point (0.61 vs 0.14).

## What adaptive.py Should Become

Retire optimal_segnet_weight(). Keep score_sensitivity() and effective_amplification()
as analysis tools. The real adaptive value is in boundary weight scheduling and
hard-frame curriculum adaptation, not in sw.

## Lean Proofs

The w_s*T² invariant proof is correct but trivial (T² cancels by construction).
Restate honestly: "the loss function's internal T² normalization makes sw
temperature-independent, which we proved is equivalent to the invariant holding."
