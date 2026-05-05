---
name: 2026-04-28 arXiv 2604.24763 (Tuna-2) deep synthesis — 5 transferable lane proposals
description: Liu/Ren et al. "Tuna-2: Pixel Embeddings Beat Vision Encoders for Multimodal Understanding and Generation" — 7B-param unified multimodal on Qwen2.5-7B. Paper headline (encoder-free) is OPPOSITE of what we want at 100K — explicitly NO. But 4 sub-mechanisms transfer: Lane T2-XPRED (x-prediction + v-loss), T2-MASK (masking-based feature learning, Table 6 +1.4-4.2 perception delta), T2-RATIO (aggressive seg-weight sweep), T2-DROP (encoder-free ablation), T2-DUAL (controlled-variant methodology). Total $14 spend.
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## Paper identification

- **arXiv ID**: 2604.24763 — verified real (HTTP last-modified 2026-04-28 02:06 GMT)
- **Title**: "Tuna-2: Pixel Embeddings Beat Vision Encoders for Multimodal Understanding and Generation"
- **Authors**: Liu, Ren et al. (15 authors; Meta AI / HKU / U.Waterloo)
- **Date**: 2026-04-28 (today)
- **Architecture**: 7B-param unified multimodal model on Qwen2.5-7B-Instruct backbone
- **Core claim**: Pretrained vision encoders (SigLIP-2, VAE) are NOT necessary — a single Conv2d patchify layer suffices once you have enough end-to-end training data. Backed by controlled Tuna→Tuna-R→Tuna-2 ablation.

## Why the HEADLINE doesn't transfer

The encoder-free thesis only holds at **>7B params with massive data**. Our scale is 100K params at one fixed 60s clip. Quantizr's mask-only archive at 0.33 score proves scorer-aligned encoders ARE the right artifact at our scale. **Explicitly dispositioned NO** in synthesis §9 — do NOT propose "encoder-free renderer" as a primary lane.

## 4 sub-mechanisms that DO transfer → 5 lane proposals

### Lane T2-XPRED (#1 by EV, cheapest)
- **Premise**: replace MSE loss in self_compress.py training with Tuna-2's x-prediction + derived v-loss formulation. The v-loss is a velocity-prediction reformulation that empirically converges 1.5-3× faster on imbalanced losses.
- **Predicted band**: -0.03 from 1.15 baseline → ~1.12
- **Cost**: $0.50, ~2h
- **Composability**: orthogonal to all renderer-arch lanes; drop-in loss replacement
- **Risk**: low — purely a loss-function change, fully reversible

### Lane T2-MASK (#2 by EV)
- **Premise**: masking-based feature learning during training. Apply with probability p=0.5, mask ratio 15%, ONLY in the last 40% of training, using a single shared learnable mask token.
- **Predicted band**: -0.06 from 1.15 baseline → ~1.09
- **Cost**: $1.50, ~6h
- **Backed by**: Tuna-2 Table 6 ablation (+1.4 to +4.2 perception delta on benchmark suite)
- **Composability**: composes with Lane A, S, W, Ω, HF, HFM, SI-V2 (**7 existing lanes get free uplift**)
- **NOTE**: subagent recommended "retire Lane SAUG in favor of T2-MASK" — REJECT. SAUG targets proxy/auth distribution gap (training-target perturbation); T2-MASK targets feature robustness (training-input dropout). Different mechanisms. Let both land.

### Lane T2-RATIO (high-uncertainty)
- **Premise**: aggressively sweep seg_weight above the historical cap of 100. Tuna-2's loss-balancing analysis suggests we may be in a low-seg-weight regime that can scale 2-5× before saturation.
- **Predicted band**: -0.08 from 1.15 baseline → ~1.07 IF non-zero, but high variance
- **Cost**: $9, ~12h (sweep across 5 ratio values)
- **Risk**: HIGH — might hit the "segnet_loss_weight > 100 overwhelms PoseNet signal" failure mode (per CLAUDE.md FORBIDDEN PATTERNS)
- **Mitigation**: sweep with PoseNet-loss-floor protection (auto-revert if PoseNet > 2× baseline)

### Lane T2-DROP (informational)
- **Premise**: encoder-free renderer ablation — does our 100K-param renderer have enough capacity to operate without the SegNet-conditioning encoder?
- **Predicted band**: regression (likely 1.5-3.0 range)
- **Cost**: $3, ~12h
- **Purpose**: empirically test where the Tuna-2 crossover is for OUR architecture
- **CONFLICTS with**: Lane DI / Lane M / Lane MOS (different conditioning hypotheses)

### Lane T2-DUAL (process change, $0)
- **Premise**: adopt Tuna-2's controlled-variant methodology — every new lane should be paired with its "minimal-change" baseline so single-mechanism contributions are isolated.
- **Cost**: $0 (no GPU)
- **Apply to**: existing lane scripts going forward; doesn't retroactively change any landed lane

## 8 dispositioned NOs (council should NOT re-litigate these)

1. Tuna-2-as-feature-extractor at our scale
2. Pixel-space flow at inflate (violates strict-scorer-rule)
3. Tuna-2 generation head (7B-param-only signal)
4. 7g3u ratio (cross-architecture, doesn't apply)
5. Tuna-R intermediate variant (subsumed by Tuna-2)
6. Replacing SegNet (we don't train SegNet — it's the scorer)
7. SFT corpus (we have 1 fixed 60s video, not a corpus)
8. Scaling-curve quantitative extrapolation (only 7B → 100K is 5+ orders of magnitude; no confidence)

## Council deployment recommendation

| Order | Lane | Cost | Pred Δ | Confidence |
|---|---|---|---|---|
| 1 | T2-XPRED | $0.50 | -0.03 | HIGH |
| 2 | T2-MASK | $1.50 | -0.06 | MED-HIGH |
| 3 | T2-DUAL | $0 | (process) | HIGH |
| 4 | T2-RATIO | $9 | -0.08 | LOW |
| 5 | T2-DROP | $3 | (informational) | N/A |

**Total $14 — well within $300 budget.**

## Cross-references
- `feedback_compress_time_unlimited_archive_small_20260428` — confirms scale rejections must be reframed
- `project_cosmos_mae_lyra_telescope_synthesis_20260428` — original Cosmos research; SAUG vs T2-MASK distinction
- `project_council_eurekas_driving_geometry_20260428` — 8 council EUREKA lanes
- `project_outstanding_work_and_stacks_20260428` — TIER 3 catalog
- Full synthesis: `/Users/adpena/Projects/pact/.omx/research/arxiv_2604_24763_synthesis.md`
