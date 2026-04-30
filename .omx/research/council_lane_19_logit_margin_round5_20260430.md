# Council Round 5 — Lane 19 Logit-Margin Loss

**Date:** 2026-04-30 (~5min after Round 4)
**Round:** 5 (counter at 1/3)
**Status:** REVIEW IN PROGRESS

## Rotating perspective: MacKay (Bayesian + MDL)

**Concern:** From an MDL standpoint, the optimal loss for a sequence of bits is `-log p(bit)`. SegNet outputs categorical distributions; `-log p(c*)` IS the Shannon-optimal loss per pixel. Lane 19 reweights this by a learnable importance weight — but the weight depends on the model's CURRENT confidence, not on the prior probability of the pixel being a boundary. From a Bayesian view, this is using the LIKELIHOOD as a proxy for the PRIOR, which violates Bayesian decision theory.

**Counter:** This is not a Bayesian estimator; it's a training-time loss surrogate. The contest scorer is ARGMAX disagreement, not negative log-likelihood. Lane 19 trades a small NLL deviation for a better argmax-disagreement signal. Acceptable for the contest objective.

**Sub-concern:** The MDL question "what's the rate cost of this approximation?" — answer: zero, since Lane 19 doesn't change archive bytes. It's a training-time-only knob.

**MacKay: GREEN.**

## Rotating perspective: Filler (post-Round-3 acceptance)

**Concern:** Re-checking my Round 3 finding. The detach fix is the same convention as STC: STC uses syndrome-trellis coding with a parity-check matrix; the matrix is FIXED (detached) per pixel; only the message bits are gradient-updated. Lane 19's weights are now FIXED-per-step (detached); only the CE direction is gradient-updated. Same paradigm. Endorsed.

**Filler: GREEN.**

## Rotating perspective: Boyd (convex optimization revisit)

**Concern:** With `weights.detach()`, the loss is now `mean(detached_w * CE)` which IS convex in logits (convex combination of CE terms, where each CE is convex). So per-step the loss surface IS convex in logits! Before the fix, it was non-convex due to the w-gradient. The fix improved the optimization landscape from an unknown non-convex surface to a known convex one (per-step).

Of course, the CHAIN through the renderer's params is still non-convex (NN function is non-convex). But the LAST LAYER (logits → loss) is now convex. This is good engineering practice.

**Boyd: GREEN with positive note.**

## Rotating perspective: Tao (mathematical re-check)

**Concern:** Verify the detach() behavior in PyTorch once more:
- `weights.detach()` returns a NEW tensor that shares storage but does NOT track gradient.
- `ce * weights.detach()`: forward computes ce_i × w_i for each element. Backward: ∂(ce * w_detached)/∂ce = w_detached (a constant w.r.t. autograd). ∂/∂w = N/A (no graph edge).
- The downstream `weighted.mean()` then `weighted.sum()` aggregations propagate ∂L/∂ce which then propagates back through the CE computation to logits.

Edge case: what if `weights` is detached AT ZERO? Then `ce * 0 = 0`, gradient w.r.t. ce is zero, no backward. Confirmed: confident pixels get zero gradient (intended behavior).

**Tao: GREEN.**

## Rotating perspective: Hassabis (cross-domain strategic)

**Concern:** From a strategic standpoint, is Lane 19 worth the dispatch cost ($1.50)?

**Analysis:** Phase 2 ACCELERATE per Council E lists Lane 19 as one of 5 tier-2 lanes. If the predicted band [0.75, 1.05] holds, this is a meaningful sub-1.0 contender for the contest deadline. If it lands at the floor (0.75), it's a -0.30 improvement on Lane G v3 (1.05) — biggest single-lane improvement we've achieved this month.

Compared to Quantizr 0.33: still 0.42 above. Lane 19 alone cannot win. But STACKED with Lane PD-V2 (-7-11bp), Lane Ω-W-V2 (40.98% byte savings on renderer), Lane SC++ (q_faithful), the cumulative path to <0.50 becomes plausible.

**Hassabis: GREEN. Dispatch.**

## Rotating perspective: Karpathy (engineering practitioner)

**Concern:** The remote_lane_19 script invokes `train_renderer.py` with the profile flag. The training takes ~5h on Vast.ai 4090. If at any point the watchdog's heartbeat detects an issue, we lose the run. The script has provenance + heartbeat + NVDEC probe. Standard pipeline; no extra failure modes specific to Lane 19.

**Karpathy: GREEN.**

## Round 5 verdict

**0 issues found. Counter: 2/3.**

All 6 reviewers GREEN. Round 3 fix is stable.
