# Council Adversarial Review — Round 1 (Yousfi + Fridrich + Contrarian rotation)

**Date**: 2026-04-30
**Document under review**: `.omx/research/grand_council_paradigm_shift_to_shannon_floor_20260430.md`
**Reviewers**: Yousfi (challenge creator), Fridrich (UNIWARD/SRM), Contrarian (veto)

## Issues found (3)

### Issue 1 (Yousfi) — NeRV mask codec boundary smearing risk

**Concern**: NeRV's coordinate-MLP is well-suited for smooth interiors but mask classes have HIGH-FREQUENCY argmax boundaries (5 classes, sharp transitions). NeRV may smear boundaries → SegNet distortion regression on decoded masks. Lane 12's "94.4% byte savings" claim is byte-only; SegNet impact unmeasured.

**Severity**: MEDIUM (could regress score by 0.02-0.05 if boundaries smear).

**Fix**: Lane 12 design must include SegNet-distortion-on-decoded-masks as a kill criterion. When Lane 12 trains, measure both byte savings AND argmax accuracy on the decoded masks vs ground truth. Add to Lane 12 success criteria in Section 4.1.

**Status**: ACK — incorporated into Section 4.1 design notes.

### Issue 2 (Fridrich) — Sensitivity-map calibration overfit risk

**Concern**: Sensitivity-map module computes per-channel Hessian on calibration set (600 video pairs). If the calibration distribution doesn't match the contest test set's distribution, the sensitivity weights are overfit and Ω-W-V3's protection scheme misses real-world high-sensitivity channels → PoseNet regression returns.

**Severity**: MEDIUM (could nullify paradigm shift β).

**Fix**: Sensitivity-map design must include cross-validation. Split 600 pairs into 480 train / 120 holdout for sensitivity computation; verify sensitivity distribution on holdout matches train within 10% tolerance.

**Status**: ACK — added to Section 5.3 acceptance criteria + Section 9.1.

### Issue 3 (Contrarian) — Data-side lanes ignored in paradigm shift inventory

**Concern**: All three top paradigm shifts (α + β + γ) assume codec is the bottleneck. What if it's the DATA? Renderer trained on 17 videos × 1200 frames = ~20K pairs. Quantizr/Selfcomp's data augmentation may be the secret sauce, not their codecs. Lane MAE-V (mask augmentation pretraining) and Lane SAUG (Cosmos self-augmentation) are scaffolded but not at Level 3.

**Severity**: HIGH (could mean we're solving the wrong problem).

**Fix**: Add data-side lanes (Lane MAE-V, Lane SAUG) to Phase 2 ACCELERATE list. Run augmentation lanes parallel to codec lanes. Re-evaluate paradigm-shift priority after augmentation lanes land.

**Status**: ACK — added to Section 9.1 + roadmap consideration.

## Counter

**Round 1: 3 issues found. Counter resets to 0/3.**

## Next round

Round 2 with Shannon + MacKay + Hotz rotation.
