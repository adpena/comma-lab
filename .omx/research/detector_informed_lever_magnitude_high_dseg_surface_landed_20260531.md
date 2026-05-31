# Detector-informed lever MAGNITUDE TEST on a high-baseline-d_seg render — LANDED 2026-05-31

`[macOS-CPU advisory]` / `[macOS-MLX research-signal]` — NON-PROMOTABLE per Catalog
#192/#341/#127/#323. $0 local, NO cloud, NO PR. All numbers are advisory; never a
contest score.

## The question
Sister #1585 (commit `59ba009f0`) VINDICATED the operator's best-chance thesis: at
optimal form (UWD1 direct-payload surface + REAL detector-informed cost-map), the
detector-informed allocation is the UNIQUE winner and the margin WIDENS with budget
(real-lever signature). BUT the magnitude was only ~1e-5 d_seg because the degraded
reconstruction was a slow-stride TEMPORAL prediction with a tiny baseline (0.0176) —
almost no boundary-flip headroom. **Does the lever scale into contest-relevant
territory on a HIGH-baseline-d_seg surface where there is 1-2 orders more headroom?**

## What changed vs sister #1585 (two things)
1. **High-baseline surface.** Instead of slow-stride temporal prediction, the degraded
   render is a genuine spatial collapse of the structure SegNet's argmax depends on
   (a palette/LUT-render artifact): `downsample_xN` (low-pass), `blockify_Npx`
   (NxN block average). Baselines: 0.045 / 0.058 / 0.404 — 2.5x to 23x above the 0.0176
   temporal proxy.
2. **FULL-GRID detector weight (codex Finding 2).** Per
   `codex_findings_z8_pixel_driver_and_segnet_grid_premise_20260531T153038Z` Finding 2,
   "SegNet interiors are free" is STALE — SegNet responds across the FULL 384x512
   argmax grid. So the detector weight is the FULL MEASURED SegNet response =
   `|∂ L_seg / ∂ pixel|` input-gradient saliency (one backward pass per frame, dense
   **77.3% non-zero** — boundary + class-interior + region), NOT the sparse
   `exp(-margin/τ)` boundary band (~4.58% of pixels, the sister's signal).

## Apples-to-apples
The degraded render is CORRECTED by a sparse δ = GT − degraded, packed THREE ways at
MATCHED `target_bytes` on the byte-closed UWD1 sidechannel
(`pack_sparse_delta`→`unpack`→`apply_delta_to_frame`):
`uniform` (rank by |δ|) vs `texture_only` (S-UNIWARD) vs `detector_informed`
(S-UNIWARD × full-grid SegNet response). REAL SegNet d_seg (argmax-flip RATE vs GT) of
each corrected reconstruction is measured at the SAME bytes. Real `upstream/videos/0.mkv`
frames (6, stride 30, 874×1164), real `segnet.safetensors`. l∞ budget 16.
Allocation-diff no-op guard (Catalog #105/#139/#220): detector vs uniform symdiff=8000,
vs texture=7914-7970 → REAL allocation change, not a no-op.

## Per-budget d_seg/byte table (margin = d_seg_uniform − d_seg_detector; >0 = detector better)

### downsample_x16 — baseline d_seg = 0.0451 — verdict CONTEST_RELEVANT (detector wins 5/5)
| bytes | detector | texture_only | uniform | margin vs uniform |
|---:|---:|---:|---:|---:|
| 800   | 0.044718 | 0.045081 | 0.045107 | **+0.000388** |
| 1600  | 0.044085 | 0.045067 | 0.045119 | **+0.001033** |
| 3200  | 0.043441 | 0.045091 | 0.045124 | **+0.001683** |
| 6400  | 0.042628 | 0.045138 | 0.045141 | **+0.002513** |
| 12800 | 0.041712 | 0.045047 | 0.045180 | **+0.003468** |

### blockify_16px — baseline d_seg = 0.0580 — verdict CONTEST_RELEVANT (detector beats uniform 5/5)
| bytes | detector | texture_only | uniform | margin vs uniform |
|---:|---:|---:|---:|---:|
| 800   | 0.058054 | 0.058041 | 0.058064 | +0.000010 |
| 1600  | 0.057875 | 0.058050 | 0.058083 | +0.000209 |
| 3200  | 0.057727 | 0.058055 | 0.058088 | +0.000361 |
| 6400  | 0.057543 | 0.058047 | 0.058081 | +0.000538 |
| 12800 | 0.056997 | 0.057978 | 0.058057 | **+0.001060** |

### downsample_x32 — baseline d_seg = 0.4039 — verdict NO_LEVER (detector HURTS 0/5)
| bytes | detector | texture_only | uniform | margin vs uniform |
|---:|---:|---:|---:|---:|
| 800   | 0.405485 | 0.404590 | 0.403870 | -0.001615 |
| 1600  | 0.405508 | 0.406067 | 0.404541 | -0.000967 |
| 3200  | 0.402426 | 0.405932 | 0.399268 | -0.003158 |
| 6400  | 0.411468 | 0.407941 | 0.394926 | -0.016541 |
| 12800 | 0.414920 | 0.409276 | 0.397630 | -0.017291 |

## Magnitude verdict: CONTEST_RELEVANT (with a precisely-measured boundary condition)
- **Does the lever scale >> 1e-5?** YES on moderate-degradation surfaces. The
  best margin is **+0.003468 d_seg** (downsample_x16) — **~2 orders of magnitude above**
  the sister's ~1e-5 temporal proxy, and it lands squarely in the contest-relevant
  1e-3..1e-2 range. The margin WIDENS monotonically with budget on both moderate
  surfaces (the real-lever signature). On downsample_x16 the detector is the UNIQUE
  lowest-d_seg method at every budget (5/5).
- **The honest boundary condition (why this is credible, not overclaimed):** the lever
  REQUIRES correctable structure. On the extreme-collapse surface (downsample_x32,
  baseline 0.404) the detector HURTS by up to -0.017 — uniform |δ|-ranking wins because
  when 40% of pixels flip, the dominant signal IS the large-|δ| pixels and the
  detector reweight pulls bytes away from them. The lever is a moderate-degradation
  phenomenon, not a universal "detector always wins."
- **Mechanism:** uniform allocation makes d_seg slightly WORSE on moderate surfaces
  (it spends bytes on |δ|-large but score-irrelevant pixels); the full-grid SegNet
  response concentrates bytes on argmax-sensitive pixels, buying score-relevant flips.

## Canonical equation (Catalog #344): REGISTERED
`detector_informed_recon_weight_d_seg_savings_v1` — 10 empirical anchors (the 2
contest-relevant surfaces × 5 budgets), residual 0.0 at registration (source-is-anchor).
Domain of validity pins `baseline_d_seg_range: [0.03, 0.55]` and the boundary condition
that the lever inverts past ~0.4 collapse. Producers: the smoke harness + the full-grid
module; consumers: `pack_sparse_delta` + sister #1585's boundary-band module.

## Op-routable (DO NOT FIRE)
~$0.06 paired-CUDA replay on a REAL PR-class moderate-degradation render to confirm the
advisory MPS margin holds on contest hardware. Surfaced per the task contract; NOT fired
($0 local mandate). Reactivation criterion: moderate baseline (~0.04-0.06), correctable
structure; full-grid SegNet input-gradient saliency as the detector weight.

## 6-hook wire-in (Catalog #125)
1. **Sensitivity-map** — ACTIVE: the full-grid SegNet input-gradient saliency IS a
   per-pixel sensitivity surface (`|∂ L_seg / ∂ pixel|`); the module exposes it as the
   ranking signal `cost_bhw`.
2. **Pareto constraint** — ACTIVE: the per-budget d_seg/byte table IS the Pareto
   frontier of the detector vs uniform vs texture allocation at matched bytes.
3. **Bit-allocator hook** — ACTIVE: `cost_bhw` feeds `pack_sparse_delta(cost_map_bhw=...)`
   directly — it IS the per-element bit-allocation prior for the UWD1 sidechannel.
4. **Cathedral autopilot dispatch** — N/A (research-signal, non-promotable; no archive
   bytes enter a contest packet). The op-routable is operator-gated, not autopilot-fired.
5. **Continual-learning posterior** — ACTIVE: canonical equation
   `detector_informed_recon_weight_d_seg_savings_v1` registered (10 anchors) + probe
   outcome `PROCEED-advisory` in the ledger; `update_from_anchor` hook on the module.
6. **Probe-disambiguator** — ACTIVE: the magnitude test IS the disambiguator between
   "real-but-negligible" (sister 1e-5) and "contest-relevant" (this, 1e-3) — and it
   further disambiguates the moderate-vs-collapse boundary condition (NO_LEVER at 0.40).

## Files / lane
- Module: `src/tac/substrates/uniward_per_pixel_distortion/full_grid_segnet_response_cost_map.py` (+18 NO-FAKE tests)
- Harness: `experiments/detector_informed_lever_magnitude_high_dseg_smoke.py`
- Register tool: `tools/register_detector_informed_recon_weight_eq.py` (fail-closed on verdict)
- Evidence: `experiments/results/detector_informed_lever_magnitude_high_dseg_20260531/smoke_output.json`
- Lane: `lane_detector_informed_lever_magnitude_high_dseg_surface_20260531`
- Sister-DISJOINT per Catalog #340 (NSCS06-render-only / uniward-substrate files; did
  NOT touch z8 or z5 or fire cloud). Sister #1585 module untouched (18/18 still pass).

## Single highest-EV next step
Re-measure on a REAL PR-class HNeRV/grayscale-LUT render at its NATURAL moderate-
degradation operating point (baseline ~0.04-0.06, not synthetic downsample) to confirm
the lever holds on a contest-shaped reconstruction, then surface the ~$0.06 paired-CUDA
op-routable if the advisory margin is confirmed.

mission_predicted_contribution=`frontier_breaking` (the detector-informed allocation
lever now has a contest-relevant, mechanistically-bounded magnitude on a high-baseline
render — a real score-lowering primitive for the UWD1 correction sidechannel, with its
operating regime precisely characterized).
