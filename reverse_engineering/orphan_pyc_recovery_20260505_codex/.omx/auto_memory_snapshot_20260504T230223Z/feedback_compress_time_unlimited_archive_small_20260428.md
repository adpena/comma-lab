---
name: COMPRESS-TIME UNLIMITED — only archive size matters — reframes external-model integration
description: 2026-04-28 user clarification. Compute + training time are essentially unlimited (~$300 Vast.ai). Only the ~300KB SUBMISSION ARCHIVE size ships. External models (NVIDIA Cosmos, openpilot supercombo, etc.) can be invoked freely at COMPRESS TIME to extract embeddings, distill features, generate training data, fit tokenizers — none of that lands in the archive. Only the renderer.bin + masks.mkv + poses.pt + small auxiliary blobs go in.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## The rule

**Compress-time compute is essentially unlimited.** **Only the archive bytes matter.**

When evaluating external-model integration (NVIDIA Cosmos, openpilot supercombo, RAFT, MAE, Telescope, …), the relevant question is NOT "can it fit in the archive" — it's "can the COMPRESS-TIME OUTPUT fit in the archive."

## Why

- Vast.ai budget ~$300 secured (no $25 cap)
- Training time unlimited (multi-day runs OK if predicted-band justifies)
- Archive size = 300KB target = the 25×rate component of 100×seg + sqrt(10×pose) + 25×rate
- Strict-scorer-rule: scorers cannot load at INFLATE time (per Yousfi PR #35) — but they ARE loaded at compress time (we already do this)
- Same logic applies to Cosmos / openpilot / supercombo / any external model: load at compress, distill / extract / fit, ship only the small artifact

## What this unblocks

| Pattern | Compress-time use | Archive output |
|---|---|---|
| Cosmos Transfer 2.5 conditioning tokenizer | Encode 1199 mask pairs into latent tokens; distill the tokenizer down to ~30K params | Distilled tokenizer (~10-20KB) + token stream |
| Cosmos Predict 2.5 diffusion teacher | Run 36-step teacher on 1199 pairs; train tiny student at our scale | Student renderer (~80K params) |
| Cosmos RL FP4 recipe | Apply their protected-ops + QAT schedule at compress time | FP4 renderer.bin (~40KB) |
| openpilot supercombo penultimate features | Extract at compress; distill to 32-dim embedding per pair | scene_embedding.bin (~20KB) |
| RAFT optical flow (Lane FL, already done) | Run RAFT at compress; derive analytical poses | optimized_poses.bin (~7KB) |
| Telescope foveation (Lane HF) | Train hyperbolic params (4 floats × 600 pairs = 9.6KB) | foveation_params.bin (9.6KB) |
| MAE-style 75% mask in-painting | Train renderer with 75%-mask augmentation at compress | Smaller masks.mkv |

## What this does NOT change

- The strict-scorer-rule: NO scorers (or scorer-derived networks like supercombo) loaded at INFLATE time
- The 30-min inflate budget on T4
- The 1200-frame x 384x512 scorer-input geometry
- The canonical scorer archs: SegNet=EfficientNet-B2 5-class, PoseNet=FastViT-T12 YUV6 12-channel
- The deterministic-archive requirement (Check 13)

## Common error mode (avoid)

The prior Cosmos research subagent dismissed Cosmos as "wrong scale 1000×, can't fit in archive." That's wrong because **the Cosmos MODEL doesn't need to fit in the archive — only the COMPRESS-TIME OUTPUT does.** Always reframe before dismissing.

## Integration anchors (mandatory cross-checks for any external-model lane)

1. **openpilot integration**: We have Lane OS (supercombo seeding), Lane DI (penultimate features distill), Lane LM (lane-mark zero-cost poses). Any new lane should consider whether openpilot's free-to-use features (optical flow, lane geometry, depth) compose.
2. **Hardware exploits**: T4 has hardware-accelerated INT8 (NOT FP4); MPS has 23× drift vs CUDA on PoseNet (memory: `feedback_mps_cuda_drift_critical`); chroma half-res in YUV6. Some Cosmos NIM-style optimizations may translate.
3. **Canonical upstream auth eval**: Every artifact MUST score well on CUDA inflate.sh → upstream/evaluate.py (memory: `feedback_proxy_auth_math_useless`). Cosmos's training distribution ≠ our scorer's distribution; transferable-pattern proposals must pass auth eval to count.

## Cross-references
- `feedback_proxy_auth_math_useless` — auth eval is the only ground truth
- `feedback_strict_scorer_rule` — why scorers cannot load at inflate
- `project_cosmos_mae_lyra_telescope_synthesis_20260428` — original (incomplete) Cosmos coverage
- `project_lane_taxonomy_stacking_strategy_20260427` — lane composition rules
- `project_outstanding_work_and_stacks_20260428` — TIER 3 catalog
