# Council Round 4 — Lane 19 Logit-Margin Loss

**Date:** 2026-04-30 (~5min after Round 3 CRITICAL fix)
**Round:** 4 (counter at 0/3 after Round 3 RED finding)
**Status:** REVIEW IN PROGRESS

## Round 3 fixes verified

**Fix 1:** `weights = fragility_weights(logits, threshold=threshold).detach()` — added detach + 11-line comment explaining the gradient-direction bug it prevents.

**Fix 2:** New regression test `test_fragility_weights_detached_in_loss_synthetic` — pins the correct gradient direction on a boundary-WRONG pixel (∂L/∂z[top1] > 0, ∂L/∂z[GT] < 0).

**18 tests passing.** No new behavior changes to argparse / profiles / scripts.

---

## Rotating perspective: Selfcomp (post-Round-3 sanity check)

**Concern:** The detach() fix changes the loss MAGNITUDE distribution slightly because `weights.detach()` means the backward pass treats `w` as a constant. Forward value unchanged. Backward value: only the `w × ∂CE/∂z` term remains. The `∂w/∂z × CE` term that Round 3 identified as buggy is now ZEROED.

Verifying my regression test still triggers post-fix: yes — test_fragility_weights_detached_in_loss_synthetic passes only after the detach. If someone removes detach, the test fails because grad_top1 < 0 (the bug returns).

**Selfcomp: GREEN.**

## Rotating perspective: Yousfi (re-check the contest scoring)

**Concern:** Now that Lane 19 is a "pure CE-reweighting" (gradient only flows through CE), is it still doing what we want? The math:
- L_lane19 = mean(w * CE) where w is detached.
- ∂L_lane19/∂z = w × ∂CE/∂z = w × (p - one_hot(c*))

This is exactly "CE gradient × per-pixel importance weight" which is the textbook focal-loss-style weighted CE. For boundary pixels (w > 0): standard CE gradient × w. For confident pixels (w = 0): zero gradient. For confident-WRONG pixels (margin >> threshold even though wrong): w = 0 → zero gradient → REGRESSION TO Round 1 Contrarian concern.

**Counter:** Round 1 already accepted this; the underlying scorer_loss × 100 catches confident-wrong pixels via standard CE. Lane 19 is purely additive on top. The detach fix doesn't change this story — both pre-fix and post-fix Lane 19 contributes 0 to confident-wrong; only the boundary-pixel direction is now correct.

**Yousfi: GREEN.**

## Rotating perspective: Hotz (post-fix optimization)

**Concern:** With weights detached, the AST/autograd graph for backward is simpler (no cross-graph edges from weights to logits). Slightly faster backward pass. No correctness concerns.

**Sub-concern:** Lane 19 still does the topk + gather + scatter for every step. On CUDA at 384×512×5 logits, this is probably 1-2ms per step — negligible compared to the ~50ms SegNet forward. Not worth optimizing.

**Hotz: GREEN.**

## Rotating perspective: Contrarian (re-attack with full force)

**Concern 1:** Now that detach is in, what happens at the THRESHOLD BOUNDARY? Pixels with margin ≈ threshold ε have weight ≈ 0+ε. Tiny weight × CE = tiny loss. As training progresses and the renderer learns boundaries, these pixels DRIFT past threshold (margin > threshold) → weight = 0 → no gradient. The pixel is then ABANDONED by Lane 19. It must rely on scorer_loss × 100 for further refinement.

Is this actually a problem? scorer_loss × 100 still applies. The pixel transitions from "boundary, full Lane 19 + scorer_loss" to "post-boundary, scorer_loss only". At the post-boundary stage, the renderer could DRIFT BACKWARD (margin shrinks back below threshold) if scorer_loss alone isn't enough to keep it confident. Then Lane 19 kicks back in. This is HYSTERESIS-like. Probably fine.

**Concern 2:** What if the threshold is set such that NO pixels ever have margin < threshold (e.g., threshold = 0.001)? Then all weights = 0 → loss = 0 always → Lane 19 contributes nothing → silent OFF.

**Action item:** Add a runtime warning if a training step has loss == 0 from Lane 19 for too many consecutive steps (suggests threshold too low). Defer to instrumentation phase; not a blocker.

**Concern 3:** Combined with the Round 1 fix (logit_margin_weight = 10.0), the effective loss contribution is 10 × mean(w × CE) where mean is over ALL pixels (even confident ones with w=0). So if 90% of pixels are confident → only 10% contribute to the mean → effective magnitude is 10 × 0.1 × ~CE = ~CE. That's roughly comparable to scorer_loss / 100 ≈ CE / 100. Wait that's WAY smaller than I thought.

Let me redo: mean(w × CE) where 10% of pixels have w × CE ≈ 0.5, 90% have 0:
- mean = 0.1 × 0.5 + 0.9 × 0 = 0.05.
- Lane 19 contribution = 10 × 0.05 = 0.5.
- scorer_loss contribution = 100 × CE_mean ≈ 100 × 0.5 (if CE_mean is dominated by all pixels) = 50. WAIT.

Looking at scorer_loss in losses.py:210 — it's NOT mean of CE; it's the actual contest scorer eval score (segmentation distortion + pose distortion + ...). The output is in score-units, not CE-units. Looking at the call: `scorer_loss(...) → returns segmentation_score + posenet_score`. Typical training value is in the range 0.005-0.5.

So scorer_loss × 100 contribution ≈ 100 × 0.05 = 5. Lane 19 × 10 contribution = 0.5. Lane 19 is 10% of scorer_loss — actually meaningful, not noise. **The Round 1 weight=10 fix was correct.**

**Contrarian: GREEN.** No new bugs.

## Rotating perspective: Carmack (engineering shortcut)

**Concern:** The whole Lane 19 code adds ~150 LOC (module + tests + integration) to chase a -1e-3 to -3e-3 SegNet improvement. Could a simpler hack achieve the same?

**Counter:** The Round 1+3 analysis showed that NAIVELY upweighting boundaries (e.g., a hard boundary mask) would re-introduce the gradient direction issue OR be no different from standard CE. Lane 19's specific formulation IS the right technique. The 150 LOC are mostly tests + preflight + careful argparse plumbing — not "wasted".

**Carmack: GREEN with grumble that 150 LOC for -10bp is high overhead. But the council DESIGN said 'predicted band 0.10-0.30 score points', not -10bp. If actual delta is -10bp, lane is borderline.**

## Round 4 verdict

**0 issues found. Counter: 1/3.**

All 5 reviewers GREEN. Round 3 CRITICAL fix held up under attack from 5 different perspectives.
