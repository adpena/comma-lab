# Online research ledger — Domain I: Scorer-architecture exploitation

Per-paper notes; 7 entries.

---

## I.1 — FastViT (Vasu et al., ICCV 2023)
- **Authors**: Pavan Kumar Anasosalu Vasu, James Gabriel, Jeff Zhu, Oncel Tuzel, Anurag Ranjan (Apple)
- **Venue**: ICCV 2023
- **arXiv**: https://arxiv.org/abs/2303.14189
- **Repo**: https://github.com/apple/ml-fastvit
- **Apple ML Research**: https://machinelearning.apple.com/research/fastvit
- **Empirical claim**: Hybrid CNN+ViT; structural-reparameterization at inference; RepMixer token-mixing operator (Y = DWConv(BN(X)) + X simplified to Y = DWConv(X) at inference).
- **Relevance**: THE PoseNet architecture. RepMixer is REPARAMETERIZED at inference — meaning the "train-time" and "inference-time" architectures DIFFER. Implications for our gradient analysis: gradients during score-aware training flow through the train-time skip-connected form, NOT the inference-time simplified form.

## I.2 — EfficientNet-B2 (Tan-Le, ICML 2019)
- **arXiv**: https://arxiv.org/abs/1905.11946
- **Empirical claim**: Compound scaling of depth/width/resolution; B2 = mid-range model.
- **Relevance**: THE SegNet backbone. Stride-2 stem = bilinear blindspot per CLAUDE.md.

## I.3 — Segmentation Models PyTorch (SMP; Iakubovskii)
- **Repo**: https://github.com/qubvel-org/segmentation_models.pytorch
- **Empirical claim**: U-Net implementations with arbitrary backbones (incl. tu-efficientnet_b2).
- **Relevance**: The CANONICAL SegNet implementation in our contest. Reading SMP source = reading our scorer.

## I.4 — RepVGG (Ding et al., CVPR 2021)
- **Empirical claim**: Predecessor of structural-reparameterization in CNNs.
- **Relevance**: Architectural family our SegNet ALMOST-but-not-quite uses (smp.Unet uses EfficientNet, not RepVGG; FastViT uses RepMixer which is similar to RepVGG).

## I.5 — Yousfi's EfficientNet steganalysis insight (sister of D.3)
- **References**: Yousfi-Butora-Fridrich ALASKA series.
- **Relevance**: Stride-2 stem is what makes EfficientNet-B2 a STRONG steganalyzer (per the ALASKA paper). It's ALSO what makes it BLIND to artifacts at the lowest scale. We've confirmed this blindspot empirically.

## I.6 — YUV6 / chroma-subsampling differentiable transforms
- **References**: ITU-R BT.601 / BT.709 specifications.
- **Relevance**: Our patched `rgb_to_yuv6` (per CLAUDE.md eval_roundtrip discipline) is the upstream's `rgb_to_yuv6` made differentiable. The differentiable form must EXACTLY match the upstream constants (mean 127.5, std 63.75) byte-for-byte.

## I.7 — Argmax-only output trick (operator-confirmed)
- **Reference**: Per CLAUDE.md "Exact scorer architectures — VERIFIED from upstream modules.py": SegNet score is argmax-disagreement-rate.
- **Empirical**: Only the WINNING class needs to be correct; logit MARGIN matters but only at class-boundary pixels. The "tiny logit perturbations at class boundaries are the ENTIRE signal" rule.
- **Relevance**: Suggests an L0-class encoder: store ONLY the class-boundary pixel locations + argmax (not full mask logits). Per the operator's SABOR insight (session #3 — 99.27% pixel-stable boundary-only renderer), this is already partially exploited.

## I.8 — 12-channel YUV6 input control surface for PoseNet
- **References**: Per CLAUDE.md: PoseNet input = 2 frames × YUV6 = 12 channels, resized to (512,384), normalized.
- **Relevance**: The PoseNet input transform `rgb_to_yuv6 → resize → normalize` is a CONTROL SURFACE. Anything we can do at encode time to shift this transform's output toward POSE-FIDELITY is essentially free at inflate time. **[literature-prediction]**: a learnable pre-YUV6 transform layer trained against PoseNet output could close 1-3% of pose distortion.

## I.9 — Hydra head pose structure
- **Reference**: Per CLAUDE.md: PoseNet vision(2048) → summary(512) → ResBlock → 12-dim pose → first 6 used.
- **Relevance**: The 12-dim output is over-parameterized vs the 6-dim used. Compression should exploit this.

---

## Scorer-exploitation strategies

### Strategy I.A: Train against the REPARAMETERIZED inference form of FastViT
- Currently we may train against the train-time form. Switching to inference-time form during loss computation = match-the-deployed-model EXACTLY.
- Caveat: train-time form has more gradient pathways; loss landscape may be smoother. Empirical test needed.

### Strategy I.B: Encode mask class-boundaries only (SABOR formalized)
- Per session insight #3, SABOR already exploits this. Worth formalizing as a typed primitive in the substrate registry.

### Strategy I.C: Pre-YUV6 control surface
- Inject a learnable affine transform between PIL→tensor and rgb_to_yuv6. Trained against pose-fidelity. Strict-scorer-rule honored (no scorer load; just a fixed transform).

### Strategy I.D: Exploit 6-of-12-dim pose output
- Encode only 6 dims with high precision; 6 unused dims can be cheaper or omitted entirely if PoseNet's first 6 don't depend on the latter 6. (Likely independent.)

## Follow-up reads:
- FastViT supplementary materials (Apple ML Research)
- "Review — FastViT" Sik-Ho Tsang (pedagogical): https://sh-tsang.medium.com/review-fastvit-a-fast-hybrid-vision-transformer-using-structural-reparameterization-18d6a5eb05f0
- "FastViT" Andrey Lukyanenko review: https://andlukyane.com/blog/paper-review-fastvit
