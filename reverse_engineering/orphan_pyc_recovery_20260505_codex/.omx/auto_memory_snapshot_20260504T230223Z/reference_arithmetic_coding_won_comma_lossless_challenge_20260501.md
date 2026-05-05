---
name: Arithmetic coding WON comma.ai's prior lossless video compression challenge — high-priority for current lossy challenge
description: User pointed out 2026-05-01 that arithmetic coding was the winning technique in comma.ai's previous lossless video compression challenge. This is durable competitive intelligence: arithmetic coding is a CONTEST-VALIDATED technique in the comma domain. We have partial coverage (Lane PD-V2 arithmetic-coded poses, Lane Ω-W-V2 water-fill + arithmetic) but are NOT exploiting arithmetic across ALL three streams (masks still AV1, renderer.bin still raw FP4 with brotli). Aggressive arithmetic-coded everything is a high-EV unblocked direction.
type: reference
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---

## The signal

Comma.ai ran a prior LOSSLESS video compression challenge. The winning entry used **arithmetic coding**. This validates:

- Comma.ai engineering culture appreciates information-theoretic compression
- Arithmetic coding works on real comma video data (not just paper benchmarks)
- The judges/scoring will reward archive-byte reduction from entropy coding
- The technique survives whatever production constraints comma's eval pipeline imposes

For the CURRENT lossy challenge (which adds the SegNet/PoseNet distortion terms), arithmetic coding is still valid for the rate component — and now there's the additional opportunity to pair it with score-aware entropy models that bias bits toward score-relevant regions.

## What we have arithmetic-coded today (partial coverage)

- **Lane PD-V2** (task #243 completed, +7-11 BP): arithmetic-coded pose deltas. Currently in `src/tac/pose_delta_v2.py` (or similar). Replaces fp16 poses_optimized.pt (15,620 bytes) with arithmetic-coded deltas. Predicted savings small because pose stream is already tiny.
- **Lane Ω-W-V2** (task #244 completed): water-filling + arithmetic on Selfcomp/block-FP renderer. +200-450 BP. Renderer-internal entropy coding.
- **Lane SH** (memory `feedback_patterns_that_work_20260429`): arithmetic coder using Shannon entropy of learned qint distribution. Per-Ballé-coucnil entry, the canonical R(D) rate term is `bits = -log2(p_y(y))`.

## What we are NOT arithmetic-coding (unblocked opportunity)

- **masks.mkv (421,483 bytes — DOMINATES rate)**: currently AV1 SVT-AV1 monochrome. 5 class IDs per pixel × 1200 frames × 384×512 = 944M pixels. AV1 is general-purpose video codec; an arithmetic coder over the 5-class symbol stream with a learned context model could be MUCH tighter:
  - Naive uniform prior: 944M pixels × log2(5) bits = 350 MB. Way too big.
  - Per-pixel learned prior (Ballé hyperprior, like Lane 20): closer to Shannon entropy of the empirical distribution over 5 classes.
  - Spatial+temporal context model (like CABAC or PixelCNN-prior arithmetic coder): exploits that adjacent pixels usually share class.
  - The Alpha matrix verdict (codex 13:10Z) showed pure-symbol RLE is 902KB (worse than current 412KB) — but that's WITHOUT arithmetic coding. RLE + arithmetic could be much smaller.
- **renderer.bin (211,903 bytes)**: currently raw FP4 weights with brotli wrapper. The FP4 quantized weights have a known empirical distribution; an arithmetic coder over the FP4 symbol alphabet (16 codes per param) with learned per-layer or per-channel priors should beat brotli. Lane J-NWC (task #246 completed) goes further with VQ-VAE codec; arithmetic-coded FP4 is the lighter intermediate.

## Predicted savings (rough)

If arithmetic-coded mask stream + arithmetic-coded renderer cuts each by 30-40% (typical gain over a general-purpose codec when you have a known distribution + context model):
- masks: 421KB → 250-300KB (-120-170KB)
- renderer: 212KB → 130-150KB (-60-80KB)
- Combined: -180 to -250KB on top of owv3_0120 (617,410 bytes → 367-440KB)
- Score impact (rate term only): -0.12 to -0.16 → **owv3_0120 1.0024 → ~0.84-0.88 [contest-CUDA]**

This would be a MAJOR jump (closer to Quantizr's 0.33 leader by 0.16 points in one move).

## Caveats

- Arithmetic coder + decoder must fit in archive + decode within the 30-min T4 inflate budget. A complex context-model decoder may push budget.
- Score-aware entropy models (bias bits toward score-relevant pixels) require the scorer at COMPRESS time only (per CLAUDE.md strict-scorer-rule, scorers are FORBIDDEN at inflate time).
- Need entropy-bias tests — if the entropy model adds rate elsewhere via overhead, the net could be neutral.

## Recommended dispatch

Spawn a focused subagent on:
1. Arithmetic-coded mask stream with learned context model (CABAC-style or PixelCNN-prior)
2. Arithmetic-coded renderer.bin (FP4 alphabet, per-layer priors)
3. Compose both with owv3_0120 deploy archive
4. Contest-CUDA eval on Vast.ai 35959478

Reference implementations to bootstrap from:
- `src/tac/joint_admm_proximal_pose_delta.py` (existing arithmetic coder for poses)
- `src/tac/sh_arithmetic_coder.py` (if exists — Lane SH was task #235 area)
- Standard lib: `arithmetic-coding` PyPI package (or roll our own)
- Reference: prior comma lossless challenge winner's code (if open-sourced)

## Cross-refs

- `feedback_codex_partner_coordination_state_20260501T1310Z.md` (codex Alpha matrix verdict)
- `project_codec_stacking_composition_canonical_orders_20260429.md` (canonical order — arithmetic is ALWAYS terminal)
- `project_lane_g_v3_owv3_0120_LANDED_1_002_20260501.md` (champion to stack on)
- AGENTS.md "Build Discipline" section (commit 377cf144) — applies: paper section will reference this winning-technique connection
