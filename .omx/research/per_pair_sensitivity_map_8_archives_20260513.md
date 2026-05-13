# Per-pair sensitivity map for 8 macOS-CPU-ranked archives

**Date:** 2026-05-13
**Lane:** `lane_local_hardware_maximization_sweep_20260513`
**Mode:** READ-ONLY analysis of existing canvas-sweep aggregates + 1 live per-pair diagnostic
**Evidence grade:** `[macOS-CPU advisory]` only — per CLAUDE.md axis discipline (Catalog #192)
**Score claim:** false. `promotion_eligible:` false. `ready_for_exact_eval_dispatch:` false.

---

## 1. Source

Per-archive aggregate scores from
`experiments/results/lane_macos_cpu_substrate_canvas_sweep_20260513_20260513T162636Z/canvas_manifest.jsonl`
(rebuilt by sister `lane_macos_cpu_substrate_canvas_sweep_20260513`).

Live per-pair diagnostic via new `tools/diagnose_per_pair_sensitivity.py`
(landed this lane) on A1 baseline (`a1_cuda_cpu_drift_discriminator_v_baseline_20260509T110211Z/submission_dir/archive.zip`).

## 2. Aggregate 8-archive per-component table (existing canvas-sweep)

All scores `[macOS-CPU advisory]`, N=600 contest pairs, M5 Max ARM64.

| Archive | Score | SegNet | PoseNet | Rate | Bytes |
|---|---:|---:|---:|---:|---:|
| `pr101_hnerv_ft_microcodec` | 0.192861 | 0.00056039 | 3.286e-05 | 0.0047478 | 178258 |
| `a1_baseline` | 0.192864 | 0.00056039 | 3.286e-05 | 0.0047479 | 178262 |
| `pr103_hnerv_lc_ac` | 0.194865 | 0.00057638 | 3.443e-05 | 0.0047469 | 178223 |
| `pr100_hnerv_lc_v2` | 0.195369 | 0.00057638 | 3.443e-05 | 0.0047670 | 178981 |
| `pr107_apogee` | 0.196640 | 0.00058935 | 3.580e-05 | 0.0047514 | 178392 |
| `pr105_kitchen_sink` | 0.197979 | 0.00060921 | 3.471e-05 | 0.0047371 | 177857 |
| `pr104_qhnerv_ft_best` | 0.198711 | 0.00061152 | 3.464e-05 | 0.0047579 | 178637 |
| `pr063_qpose14` | 0.345107 | 0.00072220 | 6.627e-04 | 0.0076593 | 287573 |

## 3. Per-axis marginal sensitivity (analytical, score-derived)

The contest scorer is `score = 100*seg_avg + sqrt(10*pose_avg) + 25*rate`.

At the PR101/A1 operating point (pose_avg ≈ 3.286e-05, seg_avg ≈ 5.604e-04):

```
d(score)/d(seg_avg)  = 100
d(score)/d(pose_avg) = 5 / sqrt(10 * pose_avg) = 5 / sqrt(3.286e-04) ≈ 275.9
d(score)/d(rate)     = 25
```

Comparing marginal value per unit improvement:
- SegNet marginal: 100 per unit seg_avg → 100 / (current 5.604e-04) = scale factor 1.78e+05
- PoseNet marginal: 275.9 per unit pose_avg → scale factor 8.39e+06 [POSE 47× more sensitive at margin]
- Rate marginal: 25 per unit rate → uniform $25/rate

**Interpretation per CLAUDE.md "SegNet vs PoseNet importance — operating-point dependent":**
At the OLD 1.x score operating point, SegNet was 77× more cost-effective. At the
A1/PR101 operating point (pose_avg ~3.29e-05 — well below the 2.5e-4 crossover threshold),
**PoseNet marginal sensitivity is 2.71× SegNet's TOTAL contribution** but ~47× larger in
absolute per-unit terms. Prioritize pose-axis lanes first.

## 4. Live per-pair diagnostic on A1 baseline (`tools/diagnose_per_pair_sensitivity.py`)

Smoke run with N=50 pairs (validates methodology; full N=600 in progress):
- pose_avg = 1.533e-05 (samples vary; first 50 pairs may differ from full 600 average)
- seg_avg = 5.379e-04
- top50_share = 1.0 (trivial when N=50; meaningful when full 600 lands)

Full N=600 result will land at:
`experiments/results/lane_local_hardware_maximization_sweep_20260513_<UTC>/per_pair_sensitivity_a1_baseline_600pair.json`

### 4.1 Predicted shape of full per-pair distribution (priors)

Per scoring formula + sister diagnostic patterns observed across PR100/PR103/PR107:
- **Top-50 high-leverage pairs** typically contain pairs where the renderer
  has a per-pair PoseNet distortion 10-100× the running average (e.g., scene
  transitions, lighting changes, ego-motion peaks). These are where pose-axis
  training-loss gradients should concentrate.
- **Bottom-50 low-leverage pairs** typically contain pairs with near-zero
  PoseNet distortion AND tiny SegNet disagreement. These are FREE-BYTE
  candidates — if a substrate could allocate fewer bytes per low-leverage
  pair (latent quantization, decoder coarsening), savings would flow
  straight to rate-term without touching distortion.

## 5. Top-50 / bottom-50 candidate strategy

For each of the 8 archives (FULL N=600 diagnostic operator-routable; cost
~10 min macOS-CPU per archive):
1. Run `tools/diagnose_per_pair_sensitivity.py` to compute per-pair JSON.
2. Identify top-50 pairs by `total_score_contribution`.
3. Identify bottom-50 pairs by same metric.
4. Stack per-pair top/bottom maps across the 8 archives → consensus high-leverage pairs.

Consensus high-leverage pairs feed:
- **bit-allocator hook** (Catalog #125 wire-in 3) — allocate more bytes to
  these pairs in next-substrate training.
- **sensitivity-map contribution** (Catalog #125 wire-in 1) — augments
  `tac.sensitivity_map` with empirical per-pair leverage.

## 6. Solver-stack posterior delta

The aggregate per-component table is ALREADY in the autopilot manifest
(`autopilot_manifest.json` of the canvas-sweep results). The new per-pair
diagnostic adds:
- 1 row per archive into `per_pair_sensitivity_<archive_id>.json`
- Top-50/bottom-50 pair indices per archive

These rows are NOT submitted to `cost_band_posterior.jsonl` because they
are diagnostic (not paired CPU+CUDA anchors). They feed the bit-allocator
and sensitivity-map hooks via direct read of the per-archive JSON.

## 7. Operator-routable decisions

1. **Operator-routable**: spend ~80 min macOS-CPU running the full 600-pair
   diagnostic on the remaining 7 archives (PR101, PR103, PR100, PR107,
   PR105, PR104, PR063). Output: stacked consensus high-leverage pair set.
2. **Operator-routable**: feed the consensus top-50 pair set into the
   next-substrate trainer's data-weighting (e.g. `--pair-weight-manifest
   high_leverage_top50.json`) to focus training compute on score-bottleneck
   pairs.
3. **No-op for now**: per-pair findings do NOT promote, kill, or rank
   contest candidates without paired CPU+CUDA verification per CLAUDE.md
   "Submission auth eval — BOTH CPU AND CUDA" non-negotiable.

## 8. Cross-refs

- `tools/diagnose_per_pair_sensitivity.py` — landed this lane
- `.omx/research/macos_cpu_canvas_pareto_ranking_20260513.md`
- `.omx/research/sabor_boundary_audit_20260513.md`
- CLAUDE.md "SegNet vs PoseNet importance — operating-point dependent" non-negotiable
- CLAUDE.md Catalog #192 macOS-CPU advisory canonical axis
- CLAUDE.md Catalog #125 subagent-coherence wire-in hooks
