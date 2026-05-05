---
name: Bleeding-Edge Research Survey (2024-2026) — Top Techniques for Sub-0.33
description: Literature survey identified LoRA TTO, Cool-chic, DSConv, Ghost modules, OASIS discriminator, per-class SegNet weighting.
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## Top 8 Techniques (ranked by impact/effort)

1. **FiLM on pose vectors** — HIGH impact, 1 day. Being implemented.
2. **Depthwise separable modulated conv** (MobileStyleGAN) — HIGH, 1-2 days. 287K→80-100K params.
3. **LoRA-based TTO** — HIGH, 2 days. Store base_weights + small delta. Huge rate reduction.
4. **Pose-conditioning** — HIGH, 1 day (same as FiLM, from driving world model lit).
5. **Ghost modules** — Medium, 0.5 days. Further param reduction.
6. **Per-class weighted SegNet loss** (OASIS insight) — Medium, 0.5 days. Upweight rare classes.
7. **Learnable loss weighting** (TDFusion) — Medium-High, 3 days. Automates Lagrangian.
8. **FP4 QAT** — Medium, 2 days. Better quantization than post-hoc.

## Key Strategic Insight

The gap from 0.87 to 0.33 is primarily ARCHITECTURE, not loss engineering. A smaller model (88K) conditioned on the right signals (pose + mask) beats a larger model (287K) conditioned only on masks.

## Techniques to Reconsider (previously killed/deprioritized)

- **DP-SIMS independent gen** — SegNet 0.003 was excellent. FiLM on pose could fix PoseNet.
- **Constrained gen from noise** — GPU Eureka projected 0.135. With FiLM + better arch, viable.
- **SIREN/NeRV memorization** — Cool-chic (ICCV 2023) validates per-video overfitting with tiny decoders.
- **LoRA for video-specific adaptation** — never tried. Store base + delta = massive rate savings.

## Cool-chic Validation

Cool-chic (629-800 params, VVC-competitive) proves that overfitting tiny decoders per-instance is a valid compression paradigm. This is exactly our TTO strategy. Their video codec gets close to AVC with 800 params.
