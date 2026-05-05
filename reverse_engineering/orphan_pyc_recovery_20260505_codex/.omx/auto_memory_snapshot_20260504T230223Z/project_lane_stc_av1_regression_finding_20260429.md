---
name: Lane STC LANDED — but REGRESSES on AV1-noise anchors (50× larger). Need clean-source pipeline.
description: 2026-04-29 PM. Lane STC implementation complete (3d630e7c+6292d73a, 14/14 tests). EMPIRICAL FINDING: STC encoding of Lane A masks.mkv extrapolates to ~21 MB vs Lane A's 421KB — 50× REGRESSION. Root cause: STC losslessly re-encodes AV1 quantization noise, which AV1 itself encodes lossy and efficiently. The 60-80KB design-doc savings only apply to CLEAN-SOURCE masks (raw SegNet argmax pre-AV1).
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## Result

Lane STC fully shipped:
- src/tac/stc_boundary_codec.py (~510 LOC)
- src/tac/tests/test_stc_boundary_codec.py (14 tests, all pass)
- src/tac/mask_codec.py (stc_boundary registered)
- experiments/build_lane_stc_archive.py (CLI)
- scripts/remote_lane_stc_boundary_coding.sh (5-stage Modal-deployable)
- 0 STRICT preflight violations after smoke proof generation

## CRITICAL EMPIRICAL FINDING

| Layer | Bytes |
|---|---|
| Lane A masks.mkv (AV1 monochrome, lossy) | 421,483 |
| STC re-encoding extrapolated to 1200 frames | **~21,000,000** |
| Predicted savings per design doc | 60-80 KB |
| **Actual delta on AV1-noise anchor** | **+50× larger** |

## Root cause

Lane A masks.mkv is AV1-DECODED — ~50% of pixels are AV1-quantization speckle (non-majority per frame). STC's `detect_boundary_pixels` at `boundary_fraction=0.05` saturates to nearly 100% of pixels selected because the Sobel-magnitude kthvalue threshold lands on heavily-tied zero values. Lossless re-encoding of AV1 quantization noise CANNOT beat AV1 lossy compression on the same data.

## Implication

Design doc's 60-80KB projection assumed a CLEAN (pre-AV1) SegNet argmax source. For our current Lane A / Lane G v3 anchors, STC REGRESSES. Senior engineer's earlier prediction (-0.03 standalone) was based on the same flawed assumption.

## What Lane STC actually requires (clean-source pipeline)

1. Build a clean-source mask pipeline: extract raw SegNet argmax in `submissions/robust_current/compress.sh` and feed STC directly, never round-tripping through AV1.
2. Update `inflate_renderer.py` to recognize `masks.stcb` via `detect_mask_codec`. Currently hardwired to `masks.mkv`.
3. Re-measure savings on the clean source.

## How to apply

- DO NOT dispatch Lane STC on Lane A or Lane G v3 anchors. Will regress.
- Lane STC remains valid for FUTURE pipeline that produces masks pre-AV1 (e.g. SC++ training output before AV1 encode).
- Lane script auto-skips GPU auth eval when STC archive > anchor (saves $0.20 per false-positive deploy).
- Memory cross-ref to project_senior_engineer_review_floor_revised_245_20260429 (senior eng predicted -0.03 standalone — revised: 0 on AV1 anchor, requires clean-source build).

## Council update needed

Senior engineer revised top-3 had STC at #1 with -0.03 standalone. **NEW REALITY**: STC requires a clean-source pipeline that doesn't exist yet (~3-day build). With 4 days to deadline, need to either:
- (a) Build the clean-source pipeline (~2 days) then dispatch STC
- (b) Defer STC entirely; redirect $5/3h budget to another lane

Cross-refs:
- project_senior_engineer_review_floor_revised_245_20260429.md (senior eng top-1 prescription — NOW INVALIDATED)
- docs/paper/lane_stc_boundary_coding_design_20260429.md (Stage 1 design)
- project_codex_theoretical_floor_brutal_20260429.md (codex floor analysis)
