---
name: 17 Optimization Opportunities — Council Binding Decisions
description: Full-stack optimization priorities. P0 items must be done before first training run. Mask eureka: train on AV1-decoded masks.
type: project
originSessionId: ecc348c1-2829-48d3-9d4c-10ed2aa188a8
---
## P0 — Before first training run
- #4: DALI mask validation (BLOCKER — run on Lightning T4, 50 lines, 10 min)
- #1: Renderer path (ALREADY BUILDING — asymmetric warp paradigm)
- #5: SegNet frame_t1 only (apply SegNet loss ONLY to frame_t1, free frame_t for PoseNet)
- #14: Match downscale kernel (verify bilinear everywhere)
- #7: CRF 36 CPU lane (one-line config change, queue sweep)
- Mask eureka: train on AV1-decoded masks (round-trip encode→decode before training)

## P1 — Activate in training profile
- #2: Hard argmax STE (already built, Fridrich curriculum Phase 3)
- #8: Eliminate postfilter for renderer path (flag: --use_postfilter False)
- #13: Margin SegNet loss (flag: --segnet_loss_margin 0.1)

## P2 — Gate for later
- #3/#10: FP4 quantization (flag: --quantization fp4, defer until model > 1M params)
- #6: Null-space chroma (flag: --use_null_space_chroma True)
- #11: Mixed-precision export (flag: --mixed_precision_export True)

## Deferred
- #9: Shrink motion_hidden, #12: Share stem, #15: Flow-only motion

## Key Eureka
- SegNet ONLY sees frame_t1 (x[:, -1, ...]). Frame_t is invisible to SegNet. Free it for pure PoseNet optimization.
- Train on AV1-decoded masks to match inflate distribution. Pre-encoding masks differ at boundaries.
- Mask spatial redundancy: nearest-neighbor downscale for mask conditioning at low res SPADE layers.
