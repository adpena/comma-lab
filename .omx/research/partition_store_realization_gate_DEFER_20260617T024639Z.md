# Non-neural partition STORE — Phase 1 realization gate: DEFER (boundary-survival wall)

**Date:** 2026-06-17 (UTC) · **Authority:** `[contest-CPU advisory]` NON-PROMOTABLE ·
**Lane:** `lane_partition_store_realization_gate` · **Verdict:** `DEFER_REALIZATION_WALL_DSEG_BELONGS_IN_TRAINING`
· **Frontier pointer: UNMOVED (0.19110).**

## TL;DR (the decisive measured row)

The prior probe (`reports/yousfi_partition_topaiml.json`) priced the non-neural
partition store at **d_seg = 0** because it stored `L*` (the SegNet argmax) losslessly.
That number was a **NO-FAKE gap**: the evaluator never scores a partition — it runs
SegNet on FRAMES at camera resolution and scores
`d_seg = mean[argmax SegNet(comp_frame1) != argmax SegNet(gt_frame1)]`. So the store
only helps if the inflate REALIZES the partition as a frame whose SegNet argmax — through
the exact eval chain — reproduces `L*`.

This gate MEASURED that realized d_seg (24 GT pairs, real CPU SegNet, GT via
`upstream/frame_utils.yuv420_to_rgb`). It is **NOT 0**:

| realization variant | realized d_seg | boundary flip | interior flip | store score |
|---|---:|---:|---:|---:|
| flat_fill (GT-mean color) | 0.00811 | 25.2% | 0.250% | 1.010 |
| **mu_optimized** (SegNet-best color) | **0.00641** | 23.6% | 0.112% | **0.840** |
| dilate1 / dilate2 | 0.00641 | 23.6% | 0.112% | 0.840 |
| gt_blend_0.5 (DIAGNOSTIC upper bound) | 0.00389 | 16.9% | 0.010% | 0.589 |

- Best practical store realized **d_seg = 0.00641 ≈ 2.5× the frontier's own d_seg (~0.00257)**.
- Store score (rate 0.182 + 100·d_seg + pose 0.017) = **0.84** vs frontier **0.191**.
- The d_seg **budget is NEGATIVE** (−8.3e-5 to beat frontier): at the store's lossless
  rate (0.182), even a *perfect* d_seg=0 realization is already +0.0083 over frontier.

## The operator hypothesis is FALSIFIED on both counts

Hypothesis: "interiors are flat → survive resampling trivially; only the boundary
(~0.45% px) faces the survival wall, so the store's realized d_seg should be far below
the witness sidecar's 37% wall."

1. **Interiors CAN be made nearly free** — μ-optimization (pick the per-class color
   SegNet most reliably maps to, PR#56 paradigm) drove interior flip from 0.55% → 0.11%
   → 0.01% (gt-blend). So that half of the hypothesis holds: interiors are not the wall.
2. **But the boundary band is the wall, and it is bigger and worse than claimed.** The
   boundary band is **2.25% of pixels** (not 0.45%), and **~24% of those pixels flip**
   through the eval bilinear downsample even with SegNet-optimal colors. The GT-blend
   *upper bound* (inject 50% real GT texture — not a valid store, a diagnostic) only
   drops boundary flip to 16.9%. **The wall is structural: the bilinear downsample at
   camera→384×512 mixes the two regions' colors in the 1-pixel-wide seg-grid boundary
   band → SegNet sees an intermediate color → argmax flips.** Natural texture barely
   helps; the resize is the cause.

This is the SAME failure mode as the per-pixel witness sidecar
(`reports/witness_seg_boundary_topaiml.json`, 37% boundary-survival wall,
`NO_GO_SURVIVAL_WALL`) — and the store is strictly worse because it ALSO pays the full
lossless partition rate (~0.182) on top of an unrealizable d_seg.

## Why STOP here (per the directive's Phase-1 gate)

The directive: "if realized d_seg hits a boundary wall (store has the same survival
problem as the sidecar), report that HONESTLY as the decisive finding (DEFER, d_seg
belongs in training) and STOP — do not tighten a coder for an unrealizable store."

Tightening the coder (Phase 2) cannot help: even a coder at the d_seg=0 break-even
(≤436 B/frame) leaves the store at +0.0083 over frontier, and the realization adds
another +0.64 of d_seg term. **A non-neural partition store is dead as a frontier
candidate.** The d_seg signal belongs in TRAINING (boundary-aware seg loss on a real
renderer), where the renderer's own frame-1 is scored — not in a stored-partition
realization that must survive the resize chain.

## Method / NO-FAKE custody

- `experiments/partition_store_realization_gate.py` — paints the stored partition
  (`SegNet(GT).argmax`, 384×512) into a camera-res (874×1164) frame with per-class
  canonical RGB, runs it through the EXACT eval chain
  (`SegNet.preprocess_input`: last-frame + bilinear→384×512 + `rgb_to_yuv6`), and
  measures realized d_seg = real argmax-flip rate. Variants: flat-fill, SegNet-optimized
  μ, morphological boundary smoothing, GT-blend diagnostic upper bound.
- `reports/partition_store_realization_gate.json` — the 24-pair measured row.
- `src/tac/tests/test_partition_store_realization_gate.py` — 8 NO-FAKE tests: realized
  d_seg is measured from the real chain (`0 < d_seg ≤ 1`, not hard-coded 0); the
  exact-chain forward returns the (384,512) argmax; pure helpers hand-checked.
- GT decode via `yuv420_to_rgb` ONLY; real CPU-torch SegNet; **NEVER MPS** (a live
  Track-A MPS job was running — CPU-only throughout).

## 6-hook wire-in (per Subagent coherence-by-default)

1. **Sensitivity-map** — N/A (no new per-byte sensitivity; the result is a
   realization-survival negative, recorded as a probe outcome instead).
2. **Pareto constraint** — ACTIVE (implicit): the non-neural partition store is
   confirmed dominated on the (rate, d_seg) frontier — d_seg=0 store already +0.0083,
   realized store +0.64. Records the dominated point.
3. **Bit-allocator** — N/A (no admitted bytes).
4. **Cathedral autopilot dispatch** — N/A (advisory, non-dispatchable).
5. **Continual-learning posterior** — ACTIVE: probe outcome
   `partition_store_realization_gate_20260616` registered via
   `tac.probe_outcomes_ledger.register_probe_outcome` (verdict DEFER, advisory,
   reactivation criteria pinned).
6. **Probe-disambiguator** — ACTIVE: this gate IS the disambiguator between
   "store realizes at d_seg≈0" (operator hypothesis) and "store hits the boundary
   survival wall" (measured) — the math arbitrates: it walls.

## Reactivation criteria (DEFER, not KILL — per Forbidden premature KILL)

The PARADIGM (store the partition, realize at inflate) is not killed; the specific
flat/μ-painted realization is falsified at the IMPLEMENTATION level (Catalog #307).
Reopen if EITHER:

1. A realization gets boundary-band flip below ~3% through the exact camera→384×512
   bilinear chain (current best 16.9% even with 50% GT-texture blend). Candidate:
   sub-pixel boundary placement that PRE-COMPENSATES the bilinear mixing so the
   downsampled color lands on the correct side — but the GT-blend upper bound suggests
   the headroom is small.
2. A coder drops the lossless store rate below the d_seg=0 break-even — but ≤436 B/frame
   is already +0.0083 over frontier even at perfect realization, so this alone is
   insufficient; it must combine with (1).

## Mission-honest framing

Per "THE GOAL — SUB-0.15": **this unit did NOT move the exact frontier; it produced a
decisive measured negative that closes a borderline-looking path.** The prior probe's
"+0.0083 from frontier, 1.05× of coding from beating it" was optimistic because it
assumed d_seg=0; the REAL realized store is at S≈0.84, not 0.199. The next exact-score
unit should aim the d_seg lever at TRAINING (boundary-aware seg loss on the live
renderer), not at a non-neural partition store.
