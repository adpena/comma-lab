---
name: 5 Moonshots — Embedding TTO, Pre-Computed Gradients, SegNet Mini-TTO, Ensemble, Constrained Gen from Renderer
description: Five moonshot techniques beyond pose-space TTO. Embedding-space TTO (30 values, 120 bytes) is the most exciting.
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## Moonshot 1: Embedding-Space TTO (HIGHEST PRIORITY)
- Optimize class embedding table: nn.Embedding(5, 6) = 30 values
- Archive cost: 120 bytes (essentially free)
- Each class's rendered appearance becomes scorer-optimal
- CLADE normalization amplifies 30 values into full spatial modulation
- Compounds with pose-space TTO: 36 total values optimized per video
- Implementation: 20 lines added to optimize_poses.py

## Moonshot 2: Pre-Computed Gradient Corrections
- At compress time: compute full scorer gradient at renderer output pixels
- Store sparse gradient direction in archive
- At inflate time: frames = renderer_output + alpha * stored_gradient
- One-step TTO with NO scorer at inflate time
- Contest-compliant, deterministic, instant
- Archive cost: depends on gradient sparsity (95% near boundaries)

## Moonshot 3: SegNet-Only Mini-TTO (MiniSegNet WORKS at 98.7%)
- Load MiniSegNet (25KB) from archive at inflate time
- Optimize ONLY SegNet after renderer generates frames
- PoseNet protected by hard pixel clamp (±3)
- 100 steps × 0.1ms = 12s on T4
- Archive: 25KB. Inflate time: +12s.

## Moonshot 4: Multi-Checkpoint Ensemble
- Checkpoints from ep200, ep300, ep400+
- Per-pair selection: pick best checkpoint per pair
- Selection mask: 150 bytes
- Higher rate (3× renderer.bin) but per-pair optimality

## Moonshot 5: Constrained Gen from Renderer Output
- Start from renderer output (not noise — already close to GT)
- SegNet-only optimization with MiniSegNet
- PoseNet protected by proximity to renderer output
- Essentially Moonshot 3 reframed
