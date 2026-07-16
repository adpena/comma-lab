# TEMPORAL / ADVECTION lens (projection unification lens 5) — per-stratum ξ-transport + trajectory rate, n600

**UTC** 2026-07-15 · **authority** `[macOS advisory / research-signal]` (PRE-R label-space; zlib rate = PROXY) ·
**pointer 0.19108 UNMOVED** · `score_claim=false`, `promotable=false` — this is MEANS.
**Tool** `tools/temporal_advection_stratified_measure.py` · **JSON** `experiments/results/temporal_advection_stratified_n600_20260715/results.json`
**Equation** `partition_temporal_transport_amortization_jitter_bound_v1` (`tac.canonical_equations.partition_temporal_transport_amortization_20260715`)
**Frame** `projection_unification_and_eight_lenses_20260715.md` lens 5: project the Morse-Smale partition ONCE,
transport by the ego-screw ξ∈se(3); ξ-redundant strata = FREE.

## STORES CONSULTED (proactive recall — nothing re-derived)
- `tools/measure_pose_warp_dseg.py` + grok memo `grok_pose_warp_dseg_test_20260629T181000Z.md` (pair→pair
  pose-homography label warp, class-level: Road +15–17%, hood needs identity, sky rotation-only) — REUSED
  (`pose_to_homography`, `warp_labels`, `fit_calibration`, screw regimes).
- `screw_twist_warp_dseg_probe_20260629T192609Z.md` (SCREW_WIN_ZERO_BYTE_PHYSICAL: the single twist matches the
  per-class oracle on physical classes at ~0 marginal bytes) — this unit EXTENDS it to Morse-Smale strata + rate.
- `src/tac/boundary_math/stratified_depth_warp.py` (#365), `ego_xi_trajectory.py` (PoseNet 6-vec = ξ up-to-affine),
  `partition.py` (RAG), `se3.py`/`tac.lie` (#194), phase-carrier #424/#425.
- L85 flicker floor + #333 annulus boundary-jitter + L68 (dxi 7.2KB banked) — the residual identified below is
  the SAME object.

## What was measured (n600, 599 transitions, frozen CPU-torch SegNet argmax `lstars`, deterministic numpy)
Predictors per transition p→p+1 (2-frame non-overlapping-pair gap; ξ proxy = `gt_poses[p+1]` + 3 global scalars
fit on Road+Lane over the first 100 transitions — grok discipline, per-transition variation 100% from the stored
pose; fit: `s_t=-0.00322, s_r=0, pitch=-0.01`):
**PERSIST** (identity) · **GROUND** (single plane homography) · **SCREW** (single-twist stratified composite:
hood=identity, sky=rotation-only, ground classes=plane homography; decode-realizable).
Strata of the TARGET partition: **CELL** (non-boundary, 97.84% of px) · **EDGE** (separatrix band, 2.14%) ·
**SADDLE** (2×2 plaquettes with ≥3 classes, 0.022%; ~11.2 junctions/frame).

## Results — the per-stratum ξ-amortization table

| stratum | px share | transport recovery (screw) | recovery (persist) | ξ marginal |
|---|---|---|---|---|
| CELL (per-class palette/interior) | 97.84% | **99.43%** | 99.42% | +0.005pp (~0) |
| EDGE (per-pair separatrix δ(s)) | 2.14% | **68.38%** | 68.66% | −0.3pp (~0) |
| SADDLE (point-codes) | 0.022% | **54.81%** | 54.76% | ~0 |

**Edge geometric transport (the δ(s) offset residual a curve coder would store):** distance from target
separatrix sites to the transported same-class-pair separatrix — screw: d=0 40.4%, ≤1px 72.3%, ≤2px 79.8%,
>2px 20.2%, mean 4.60px (mean inflated by max-penalty on absent pairs, 276/599-transition-pair instances);
persist: 40.8%/72.3%/79.6%, mean 4.67px. **ξ improves the mean separatrix offset by only 1.5% over persistence.**

**Saddle transport:** 6,698 target saddles; same-signature match within 2px: screw 54.7% vs persist 54.4% —
**~45% of saddle 0-cells are born/dead or moved >2px per 2-frame step**: the LEAST transportable stratum
(as the lens-5 registration expected: "saddles may not").

**Total label-transport d_seg:** persist 0.012456 · screw 0.012465 · ground-only 0.015913. Consistent with the
grok/screw probes (ground-only destroys hood/sky; screw ≈ persist in aggregate — its Road win is real but small
at this gap).

## The trajectory rate — the decisive (negative) verdict

zlib-9 conditional-coding proxy (residual = min(sentinel-plane, packbits-mask+values); PROXY, flagged):

| accounting | bytes (600 frames) | per frame |
|---|---|---|
| NAIVE per-frame partition coding | 601,931 | 1,003 B |
| TRAJECTORY = L₀ + Σ residual (persist) | 841,690 | 1,403 B/residual |
| TRAJECTORY (screw) + ξ (7.2KB gross, **0 marginal** — dxi already banked for d_pose, L68) | 846,116 | 1,411 B/residual |
| **amortization ratio naive/trajectory** | **0.715 (persist) / 0.711 (screw) < 1** | |

**Temporal transport does NOT amortize the partition rate in the raster formulation.** Mechanism (measured):
the transportable content — the cell bulk, 99.4% recovered — is exactly the content zlib already codes at ~0
(the partition is 0.04 bpp); the irreducible residual (screw: **54.3% separatrix jitter + 44.9% interior flicker
(movable/lane-dash rebirth) + 0.8% saddle**) is scattered small-support structure whose spatial entropy EXCEEDS
the whole smooth partition's. The per-frame necessary content IS the boundary jitter — the same object as the
flicker floor (L85) and the annulus boundary-jitter (#333), now seen from the rate side.

**DERIVED cross-check (flagged):** even in the curve domain, the measured jitter prior (40/72/80/20) gives a
per-site δ-offset entropy ≈ 2.2 bits + a 20% escape (birth/movable) payload ≈ ~1.1KB/frame ≈ the naive frame
rate again. The jitter bound is representation-independent unless the escape set gets a real model.

## VERDICT (per the lens-5 charter)
- **Trajectory rate = |one partition| + |ξ| + |irreducible per-frame residual| does NOT beat |per-frame × N|**
  at this formulation: the residual alone out-costs per-frame coding (ratio 0.71×).
- **Which strata transport:** cells YES (99.4% — but persistence gets it too; the scene is slow, not ξ-free);
  edges PARTIALLY (72% within 1px — the remaining ±1px jitter is argmax flicker, not ego-motion); saddles
  POORLY (55%; ~45% need explicit birth/death coding).
- **ξ's marginal value over plain persistence at the 2-frame pair gap ≈ 0 for every stratum.** ξ's real,
  already-banked value stays where the grok/screw probes put it: Road-bulk d_seg modulation, d_pose dual-use
  (7.2KB dxi), and the curve-domain phase carrier (#424/#425) — NOT raster rate amortization.

**verdict_scope: FORMULATION** — raster label-grid transport × generic zlib-9 conditional proxy × adjacent-pose
ξ proxy × 2-frame gap. NOT family-dead. Reactivation criteria (in the equation row): (a) a boundary-context
conditional arithmetic coder (residual concentrates: 54% of residual px in 2.2% of area — a distance-to-predicted-
boundary context could beat zlib); (b) the 1-frame gap (unmeasured; jitter may drop superlinearly); (c) the
curve-domain δ(s) carrier with an explicit birth/escape model beating ~2.2 bits/site.

## V9·CGauge routing
- The per-stratum carriers (cell palette / edge δ(s) / saddle point-code) should be coded ONCE + per-frame
  jitter — but the jitter is the RATE, so the win must come from MODELING the jitter (phase carrier #424/#425:
  the measured 40/72/80/20 offset distribution IS the δ(s) prior), not from transporting the partition.
- Saddle carriers need explicit birth/death events (~45%/step non-transportable) — a per-saddle lifetime code,
  not a warp.
- The 44.9% interior-flicker residual share is the movable/lane-dash rebirth long tail — the erasure axis
  (L65), routed to the movable/dash carriers, not to ξ.

## Honesty firewall / NO-FAKE
- Agreement/d_seg = REAL argmax disagreement vs the frozen CPU-torch SegNet argmax cache (no surrogate), n600.
- PRE-R, label-space, PROXY rate (zlib-9), adjacent-pose ξ proxy — all flagged; INFERRED: pose-column physical
  interpretation (inherited from the grok probe, flagged there). DERIVED (not measured): the curve-domain
  2.2 bits/site estimate.
- Triality: equations leg = `partition_temporal_transport_amortization_jitter_bound_v1` (registered);
  DAG leg = FEED-adv-strat; DSL leg = N/A (measurement-only standalone tool; no new trainer lever — the
  curve-domain routing already lives in the #424/#425 phase-carrier levers).
