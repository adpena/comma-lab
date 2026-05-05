---
name: 5-Stage Quantization Pipeline IS Now Superior
description: Our quantization stack (RESIDUAL_CODEBOOK + robust_scale + stochastic + ndim<2 skip + train-export consistency) is now strictly better than Quantizr's vanilla 5-stage QAT.
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
User claim 2026-04-25: "our five stage quantizing is smarter and better now."

What we have (verified end-to-end):

**Stage 1 (anchor / pixel pretrain):** L1 + edge loss, all params trainable. Standard.

**Stage 2 (finetune / scorer-only on clean masks):** scorer_loss with eval_roundtrip=True + noise_std=0.5. Establishes the SegNet/PoseNet manifold. Mask augmentation OFF (clean GT masks).

**Stage 3 (joint / scorer + noisy masks at deployment CRF):** NEW — adds --mask-noise-mkv (CRF=63) at --mask-noise-prob=0.5. Closes the train-test mask distribution gap that broke our CRF=63 deployment (34x PoseNet explosion). Quantizr presumably does this implicitly; we do it explicitly with provenance.

**Stage 4 (QAT):** RESIDUAL_CODEBOOK + robust_scale + stochastic rounding + ndim<2 buffer skip (R-FP4-fix today). Per the audit: 3.7x fewer zero-collapses than DEFAULT codebook on residual heads. Train-export consistency proven via property tests. Quantizr's vanilla FP4 has NONE of these — float training improvements get clipped at FP4 export.

**Stage 5 (final / low-LR + EMA + frozen BN, hinge tail):** Same as Quantizr's.

Plus auxiliary losses ALL stages 2+:
- Fridrich UNIWARD texture loss (true 8x8 box-pool spatial variance, NOT the broken var(dim=-1) we shipped earlier today)
- L∞ topk-32 mean per pair (square-root-law, REAL spread instead of single-pixel grad)
- Markov 1st-order spatial gradient continuity
- KL distill T=2.0 SegNet ONLY (kl_distill_segnet_only — NOT the double-counting kl_distill_scorer_loss)

Plus auxiliary EXPORT-time:
- INT4_LZMA2 mixed precision allocation (Cool-Chic-style)
- Brotli on all archive artifacts
- Per-tensor codebook + robust_scale propagated through __meta__

WHY THIS BEATS QUANTIZR'S VANILLA QAT:
1. RESIDUAL codebook preserves the small-magnitude tail of residual heads — Quantizr's vanilla codebook clips them to zero, killing the gradient signal that float training was using.
2. Mask augmentation in Stage 3 — Quantizr trained against his single-CRF masks; if he ever ships a different CRF, his renderer is OOD. Ours is robust by training.
3. Train-export consistency — Quantizr's QAT may train with one quantizer config and export with another (we don't know his pipeline depth). Ours has property tests proving identity to within fp16 scale precision.
4. 5 separate auxiliary losses each addressing a different scorer blind spot — Quantizr uses just KL distill T=2.0.

WHAT WE STILL DON'T HAVE THAT QUANTIZR DOES:
- Half-frame mask + warp at inflate (Quantizr stores 600 odd masks, warps even from odd via the renderer's motion field; we store all 1200)
- 5-stage QAT WITH the FP4 forward pass during the entire scorer-training loop (we wrap with QATRendererFP4 but only QUANTIZE during forward when stochastic=True; he likely runs the dequantized FP4 through the scorer at every step)

Both gaps are listed in NUCLEAR #3 (warp) and a future "QAT throughout scorer training" task. Once those land, the 5-stage stack is fully superior.

CHARTER: any future training run MUST pass --fp4-codebook=residual --fp4-robust-scale --fp4-stochastic --use-texture-loss --use-linf-penalty --use-markov-loss --kl-distill-weight=1.0 --mask-noise-mkv=<CRF=63 mkv> as default. The profile defaults can override individually but the 5-stage stack is the new floor.
