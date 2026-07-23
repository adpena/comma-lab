---
title: Codex premise falsification - DDM PT1 target custody and control identity
date_utc: 2026-07-23T20:50:28Z
lane_id: lane_ddm_pt1_continuous_paint_ceiling_20260723
research_only: true
score_claim: false
evidence_axis: "[macOS-CPU frozen-scorer advisory]"
verdict: TWO_PREPARED_PREMISES_FALSIFIED_AND_CORRECTED
verdict_scope: "PT1 target-reference and flat-paint control identity only"
pointer_moved: false
main_landing_review_required: true
---

# Falsified premise 1: current batch forward is not the cached-label authority

The prepared executor required the current frozen SegNet forward over a
16-frame batch of `gt_f1` to equal the SHA-bound cached `lstars` exactly. That
premise is false.

- The cache builder produced `lstars` with one-frame scorer forwards.
- One-frame re-forwards match the cache on the inspected first 16 frames.
- A 16-frame re-forward differs at one pixel in that same range.
- The exact n600 run records **3** batch-geometry drift pixels in total.

The correction keeps the SHA-bound cached `lstars` as the final argmax
authority and reports current batch-forward drift separately. It does not
silently replace the target. The current batch forward remains useful only as
the scorer-native activation reference.

# Falsified premise 2: E2 paint is not the PT1 flat-palette control

The prepared memo and executor called E2's inherited paint row a native-grid
flat-paint baseline and required equality with PT1's full-target flat-palette
control. The renderers are not the same:

- E2 composes chart paint on the base stream with semantic paint.
- PT1 paints a palette over every target label on the native SegNet grid.

The exact n600 totals make the mismatch explicit:

| Control | Errors | d_seg |
|---|---:|---:|
| inherited E2 chart-plus-semantic paint | 3,349,482 | 0.028393910726 |
| PT1 native-grid flat-palette control | 2,648,079 | 0.022448043823 |
| signed PT1 minus E2 | -701,403 | -0.005945866903 |

Therefore the E2 row remains a valid inherited anchor, but it is not the
within-experiment control for PT1. All PT1 arm deltas and the independent wall
now use the honestly named PT1 flat-palette control.

# Durable correction and custody

- Exact executed implementation and receipts are preserved in commit
  `38a62781b6ff3c89f243657719a2071b42ed90eb`.
- Prepared receipt SHA:
  `3977b5ab3f6fef7c2d7739be578a7024be54ce17903b68fa305f576c44df9a60`.
- Independent wall receipt SHA:
  `b056030ed38ac36d2643b60929053e029cf6931d63b0649cef59451f57d9dee3`.
- Measurement receipt SHA:
  `83e06ef47027a4997e598c824c8bb36b2185d4225ca8fcfaf8a8568fbff4b4b9`.

This falsifies two prepared assumptions, not E2, PT1, SegNet, or the geometry
family.

