# Battle Plan: Beat Quantizr (0.33)

**Deadline: May 3, 2026. 8 days remaining.**
**Current auth: 2.26. Target: < 0.33.**

## Active Experiments (A100s, overnight Apr 25)

| Experiment | Profile | GPU | Status | Key Variable |
|------------|---------|-----|--------|-------------|
| SHIRAZ | focal_ste + curriculum | A100 40GB | Phase 2 ep500, proxy 0.804 | TTO targets |
| WILDE | hinge + freeze/unfreeze | A100 40GB | Phase 1 ep225 | GT targets (A/B) |
| GREEN | WILDE + zoom_flow | A100 40GB | Phase 1 ep150 | RadialZoomWarp |

## Post-Training Pipeline (auto-chained on A100)

train → QAT (50 INT8 + 250 FP4) → pose TTO → auth eval → bundle

## Eureka Techniques (built, ready to deploy)

1. **Beneficial quant noise** (`--beneficial-quant-noise`): STE quantization between renderer and scorer during Phase 2. Renderer learns scorer-robust output.
2. **Scorer sensitivity sweep** (`step_sensitivity_sweep`): Yousfi's per-layer scorer Jacobian. Produces optimal bit allocation.
3. **Mixed-precision QAT** (`--mixed-precision-json`): Variable bit-depth per layer from sensitivity data. Knapsack-constrained to uniform FP4 budget.
4. **MXLZ export** (`export_int4_lzma2 + bit_allocation`): Mixed-precision LZMA2. Projected 25-35KB renderer.
5. **Engineered corrections** (`step_engineered_corrections`): Gradient-directed SegNet pixel fixes. Pre-computed, contest-compliant.

## Next-Gen Profiles (v2)

| Profile | Base | Addition |
|---------|------|----------|
| wilde_v2 | WILDE | + beneficial_quant_noise (6-bit) |
| shiraz_v2 | SHIRAZ | + beneficial_quant_noise (4-bit) |
| green_v2 | GREEN | + beneficial_quant_noise (6-bit) |

## Timeline

| Day | Action |
|-----|--------|
| Apr 25 AM | Download results. Auth eval. Pick winner. |
| Apr 25 PM | Sensitivity sweep on winner (full 600 pairs on A100). Deploy winner_v2. |
| Apr 26 | v2 results. Engineered corrections. Full auth eval. |
| Apr 27-28 | Optimize: half-frame masks, CRF sweep, archive compression. |
| Apr 29-30 | Final polish: postfilter, multi-pass TTO. |
| May 1-2 | Submit PR. 5-pass adversarial review. |
| May 3 | Deadline. |

## Score Projections

| Scenario | Score | Path |
|----------|-------|------|
| Current v1 (proxy 0.804) | 1.0-1.5 auth | SHIRAZ auto-chain |
| + Mixed-precision MXLZ | -0.09 rate | 25-35KB renderer |
| + Engineered corrections | -0.05-0.1 SegNet | Targeted pixel fixes |
| + v2 beneficial noise | -0.1-0.2 SegNet | Scorer-robust training |
| + Half-frame masks | -0.12 rate | 210KB → rate 0.20 |
| **Projected best** | **0.5-0.8** | Everything stacked |
| **To beat Quantizr** | **< 0.33** | Needs 2+ training cycles |

## Non-Negotiable

- All experiments through `pipeline.py --profile <name>`
- eval_roundtrip = True everywhere
- 3 consecutive auth evals within 0.01 before submission
- No ad-hoc scripts
- Destroy instances immediately after download
