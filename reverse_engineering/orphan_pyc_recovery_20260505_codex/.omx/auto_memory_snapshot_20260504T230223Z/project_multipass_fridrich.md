---
name: Multi-pass Fridrich fine-tuning — post-QAT steganalytic refinement
description: After QAT, do a dedicated pass with ONLY Fridrich losses (UNIWARD+L-inf+Markov) at very low LR. Pushes errors further into scorer null space. Like QAT fine-tuning but for steganalytic quality.
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---

## The Idea

After the main training pipeline (Phase 1→2→3→QAT), add a final pass:
- Freeze scorer loss weights (SegNet/PoseNet objectives are satisfied)
- Apply ONLY Fridrich losses: texture_weighted_loss, linf_penalty, markov_chain_loss
- Very low LR (1e-5 or lower)
- ~100-200 epochs
- Goal: push pixel-level statistics further into the scorer's null space
  without disturbing the already-optimized scorer distortion

## Why It Should Work

The Fridrich losses operate in pixel space, independent of scorer loss.
They shape the pixel-level distribution to be statistically indistinguishable
from natural images, as measured by SRNet/UNIWARD-style features.

After QAT, the model has learned scorer-optimal weights but the FP4 quantization
introduces new pixel-level artifacts. The Fridrich pass corrects these artifacts
specifically, targeting the steganalytic detectability introduced by quantization.

## Implementation

Already possible with current code:
```bash
python train_distill.py --checkpoint qat_best.pt \
  --use-texture-loss --use-linf-penalty --use-markov-loss \
  --seg-weight 0.0 --pose-weight 0.0 --pixel-weight 0.0 \
  --texture-loss-weight 1.0 --linf-weight 0.1 --markov-weight 0.5 \
  --phase2-epochs 200 --phase2-lr 1e-5 --eval-roundtrip
```

Or: add as Phase 5 in the pipeline.py convergence loop.

## When to Try

After the WILDE/SHIRAZ A/B test produces a winner and QAT fine-tune is done.
This is the final polish step before submission.
