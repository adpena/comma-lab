# n600 FULL-SCALE witness error structure — erasure-vs-shift + lane-dash persistence (MEASURED)

- **UTC** 2026-07-01T00:56Z · **authority** `[macOS-CPU/MLX research-signal] NON-PROMOTABLE`
- **pointer UNMOVED 0.19110** · score_claim **false** · promotable **false** · ready_for_exact_eval **false**
- **Scope** CPU-only, $0, NO GPU, NO training launch, live n600 run (pid 38641) untouched. Supersedes the
  96-pair verdict-subset measurement (`residual_inr_adversarial_overturn_20260701T001600Z.md`) at the
  mandatory n600 scale. Parallel render (6 CPU workers, ~16 min) of the n600 EMA-best witness through R +
  frozen CPU-torch SegNet on **ALL 600 gt_n600 pairs**, then analysis. NEVER MPS.
- **Artifacts** `experiments/results/residual_inr_adversarial_overturn_20260630T235910Z/n600/`
  (`error_structure_n600_report.json`, `chunk_00..05.npz` render cache, `n600_aggregate_flip_density_
  classmass.png`, `n600_multipane_{worst,median}_pair*.png`, `n600_lane_dash_persistence.png`) +
  scripts `measure_n600_full_erasure_shift.py` (render) + `analyze_n600_from_chunks.py` (analysis).

---

## 0. HEADLINE (n600, all 600 pairs)

**Realized d_seg = 0.006655** (785,058 flips / 117,964,800 cells) — consistent with n96 (0.006842) and the
trainer's best (0.006771); the full clip does not change the picture. At n600 scale, all n96 findings hold
AND the operator's viz observation is quantified: **the error splits 76.5% BOUNDARY-SHIFT / 23.5%
FEATURE-ERASURE.** The erasure is **dominated by LANE dashes** (47% of lane flips are full erasures =
20.8% of ALL flips; **71% of the 13,528 GT lane dashes are erased**), and **error ∝ 1/persistence is
CONFIRMED** — the smallest, faintest dashes are erased first (erase-rate 98.5% at 2-5px → 1.4% at 160+px;
corr(erased, −log area) = +0.74). This is the **spectral-bias / finest-curvelet-scale signature**. The
CAR outline (Movable) is 16.5% erasure (small/distant cars dropped); the **HOOD/MyCar edge is NOT erased
(0.3% erasure / 99.7% shift)** — an honest correction to the viz read: the hood boundary is a ≤3px
separatrix wobble, not a dropped feature.

---

## 1. d_seg + per-class flip-mass (n600)

| class | GT area | disagree/class | **flip-mass share** | wrong px |
|---|---|---|---|---|
| Road | 23.2% | 0.85% | **29.6%** | 232,527 |
| **Lane** | 0.59% | **50.1%** | **44.1%** | 346,253 |
| Undrivable | 49.5% | 0.13% | 9.4% | 73,907 |
| Movable | 1.24% | 6.54% | 12.2% | 95,500 |
| MyCar | 25.4% | 0.12% | 4.7% | 36,871 |

Self-detected class signatures (area / vertical centroid) match the CLAUDE.md canonical comma10k order.
Lane is 0.59% of the frame but 44% of all flips (half of every lane pixel flips); the static hood (MyCar)
and sky (Undrivable) are essentially solved.

## 2. Boundary localization (n600) — the residual is codim-1

| GT annulus | flips within | interior beyond |
|---|---|---|
| r2 | **94.4%** | 5.6% |
| r4 | 96.0% | 4.1% |
| r8 | 97.4% | 2.6% |

Matches n96 (94.5% r2). The residual is a thin boundary annulus, NOT interior — the design-refine's
"50–86% interior" was a deterministic-warp-bulk artifact (overturned; see the n96 memo §2).

## 3. FEATURE-ERASURE vs BOUNDARY-SHIFT (the operator's ask, per class)

A flipped GT-class-g cell is **SHIFT** if the witness argmax has class g within 3px (separatrix
misplaced) — else **ERASURE** (the feature dropped below the argmax threshold; the class vanished locally).

| class | flip px | **erasure frac** | shift frac | erasure share of ALL flips |
|---|---|---|---|---|
| Road | 232,527 | 1.7% | 98.3% | 0.5% |
| **Lane** | 346,253 | **47.1%** | 52.9% | **20.8%** |
| Undrivable | 73,907 | 1.7% | 98.3% | 0.16% |
| **Movable (cars)** | 95,500 | **16.5%** | 83.5% | 2.0% |
| MyCar (hood) | 36,871 | **0.3%** | 99.7% | 0.01% |
| **TOTAL** | 785,058 | **23.5%** | **76.5%** | — |

**Reading:**
- **LANE dashes ARE erased** — CONFIRMED strongly. 47% of lane flips are full erasures; the erased mass
  is 20.8% of ALL flips (the single biggest erasure contributor).
- **CAR outline (Movable) partially erased** — 16.5% of movable flips are erasures (small/distant cars
  dropped); the rest is edge-shift.
- **HOOD/MyCar edge NOT erased** — 0.3% erasure. The hood is solved (0.12% disagree); its flips are a
  ≤3px boundary-shift, not a dropped feature. (Honest correction to the viz read.)
- **76.5% of ALL flips are boundary-SHIFT** — Road/Undrivable/hood edges + the persistent lane lines,
  off by ≤3px. These are the "primed" flips (realized margin −0.60, close to flipping back).

## 4. LANE-DASH PERSISTENCE — error ∝ 1/persistence CONFIRMED (spectral-bias signature)

13,528 GT lane connected components (8-conn = 0-dim features). **71.0% erased** (witness keeps <10% of the
dash as lane). Persistence proxies = area (spatial scale) + mean GT margin (superlevel prominence):

| dash area (px) | n | **erase rate** |  | GT margin | n | **erase rate** |
|---|---|---|---|---|---|---|
| 2–5 | 3536 | **98.5%** |  | 0.0–0.5 | 9511 | **86.3%** |
| 5–10 | 3055 | 94.9% |  | 0.5–1.0 | 3274 | 42.1% |
| 10–20 | 1946 | 84.1% |  | 1.0–2.0 | 742 | 2.3% |
| 20–40 | 1422 | 68.4% |  | | | |
| 40–80 | 1048 | 44.8% |  | | | |
| 80–160 | 967 | 12.6% |  | | | |
| 160+ | 1554 | **1.4%** |  | | | |

- erased dashes: median **6 px** / margin **0.29**; kept dashes: median **117 px** / margin **0.68**
  (erased are ~20× smaller and ~half as prominent).
- corr(erased, −log area) = **+0.742**; corr(erased, −margin) = **+0.578** — both strongly positive,
  monotone erase-rate-by-scale. **error ∝ 1/persistence CONFIRMED at 600-scale.**

This is the **spectral-bias / finest-curvelet-scale** signature: the INR's low-frequency bias cannot
represent the highest-spatial-frequency (smallest) lane dashes, so it drops them below the argmax
threshold. The erasure slice is a **representational-capacity** limit at the finest scale, distinct from
the shift slice (a boundary-optimization limit).

## 5. Margin + override reachability (n600) — the overturn holds at scale

- Realized witness margin at flips: mean **−0.60**, median −0.43 (correct cells +7.8) — flips are barely
  lost (primed). Cached GT margin at flips: mean 0.55; **86.5% of flips at GT margin < 1.0, 64.4% < 0.5**
  — flips sit on the decision boundary.
- Override annulus mask over the witness bulk: **boundary_annulus_d4 coverage 0.845, unreachable_dseg
  0.00103 (PASSES sub-0.15 gate 0.00123); d8 0.904 / 0.00064.** Matches n96 (0.00111 / 0.00071). The
  geometry overturn (residual reachable below budget against a good bulk) holds at 600-scale.

## 6. Composition implication (the two error slices need different mechanisms)

The n600 split maps directly onto the repair mechanisms:

- **SHIFT slice (76.5%, primed, ≤3px separatrix wobble)** → **θ\* levers / Muon (train-time, 0 bytes)**
  push the primed flips over. This is the free-lever slice; the annulus stall (RL memo) is the tail.
- **ERASURE slice (23.5%, dominated by finest lane dashes, spectral-bias-limited)** → needs finest-scale
  representational CAPACITY: (a) higher-frequency directional/curvelet basis (free, but the spectral-bias
  wall is exactly here), (b) FREE deterministic openpilot-lane raster IF it survives R (addresses the
  lane erasure = 20.8% of flips), (c) the additive residual INR (whose high-freq boundary field — ID
  ~27–38 — IS this erased-dash detail), (d) stored sidecar for the last dashes. The erasure slice is the
  harder, capacity-limited slice and the binding sub-0.15 residual.
- Rate: witness blob 81,819 B (rate 0.0545); with the stored-pose sidecar budget, **sub-0.19 needs d_seg
  ≤ 0.00118, sub-0.15 ≤ 0.00077** (from 0.006655 → ~6–9× to go). residual-on-witness is +bytes; the
  rate-shrink version needs a cheaper good-enough bulk (open, distinct from geometry).

## 7. NEXT $0 STEP (highest value)

Measure whether a FREE deterministic openpilot-lane raster **survives R + SegNet argmax** on the erased
lane dashes — that is the 20.8%-of-flips / 0-byte lever aimed exactly at the erasure slice. If it
survives, it closes the largest erasure contributor for free; if not, the erasure slice needs the
finest-scale directional basis or the residual INR. This is the single highest-value $0 gate.

## 8. Review + means≠ends

- **Faithfulness:** n600 d_seg 0.006655 vs n96 0.006842 vs trainer 0.006771 — same deploy-faithful path
  (int8-dequant + so_iters=4), all 600 pairs; not a proxy.
- **NO-FAKE:** every number measured through R + frozen CPU SegNet on the real gt_n600 cache; erasure/shift
  is a 3px-neighborhood class-presence test (defined, not asserted); persistence uses area + GT-margin
  proxies (labeled proxies, not full PH); pointer UNMOVED; no score claim.
- **Honest correction:** the hood/MyCar edge is a boundary-SHIFT (0.3% erasure), not erased — corrects the
  viz read; the erasure is LANE (and small cars), not the hood.
- **Robustness note:** the first n600 run's main process died in the analysis stage (GT-load balloon:
  re-inflating the 943 MB lstars member 600× — the `load_gt_from_cache` anti-pattern). The 16-min render
  was safely cached in chunk npzs; the analysis was rerun with GT members loaded ONCE. No re-render, no
  live-run contention.
- **means≠ends:** this maps the error at the mandatory scale (a MEANS). The pointer moves only on a
  byte-closed `upstream/evaluate.py` row (CPU/CUDA, never MPS) < 0.19110.
