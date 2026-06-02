# Codex Findings: NeRV Rate Controls, Cache Gates, And Variant Inventory

timestamp_utc: 2026-06-02T12:00:15Z
author: codex
axis: false-authority local/advisory infrastructure
score_claim: false
promotion_eligible: false
ready_for_exact_eval_dispatch: false

## Verdict

The current ~90 advisory failures are not all the same failure class.

- `pact_nerv_selector_v4` is blocked by a fundamental renderer/training/export
  bug: its MLX scorer cache baseline is constant gray at the SegNet input
  surface.
- `pact_nerv_vq` passes the local cache nondegeneracy gate, so its current
  failure is fit/representation quality rather than a dead renderer.
- HiNeRV/SNeRV/HNeRV/SR-NeRV/RT/VQ/FF/NeRV++ controls should be routed through
  measured byte-price ladders. Any large representation is rejected unless
  quantization, ablation, waterfilling, zero packing, entropy coding, or
  receiver-side generation makes the measured non-rate improvement per added
  byte exceed the fixed contest byte price.

## Landed Surfaces

- `src/tac/analysis/mlx_cache_quality_gate.py`
  - False-authority scorer-input cache quality gate.
  - Blocks degenerate `segnet_last_rgb.npy` and far-from-reference cache fits.
  - Integrated into shared PACT-NeRV section-value report building so selector
    and VQ profiles inherit the gate.

- `tools/gate_mlx_cache_quality.py`
  - Operator CLI for durable JSON cache-gate reports.

- `src/tac/substrates/_shared/mlx_score_aware/modelsize_budget_plan.py`
  - False-authority model-size budget planner.
  - Uses the contest byte price `25 / ORIGINAL_VIDEO_BYTES`.
  - Selects the measured model-size point that minimizes
    `nonrate_score + archive_bytes * byte_price`.
  - Marks a size step spendable only when its marginal non-rate improvement per
    byte exceeds the fixed byte price.

- `tools/plan_compact_carrier_modelsize_budget.py`
  - Operator CLI for measured HiNeRV/SNeRV/HNeRV model-size ladders.

- `src/tac/substrates/_shared/mlx_score_aware/carrier_training_plan.py`
  - Now consumes measured model-size budget rows and includes the selected
    archive-byte budget in the carrier training plan.

- `src/tac/analysis/nerv_control_inventory.py`
  - False-authority inventory of exploitable NeRV-family controls and missing
    local bindings.
  - Tracks HNeRV `--modelsize`, HiNeRV S/M/L and bitstream quantization,
    SNeRV LF/HF/DWT controls, SR-NeRV resolution-axis dead-zone, RT/VQ residual
    tokenization, FFNeRV flow/temporal redundancy, NeRV++ decoder efficiency,
    inverse-steg saliency, master-gradient/xray, and bitmask/zero packing.

- `tools/build_nerv_control_inventory.py`
  - Operator CLI for durable JSON inventory output.

## Concrete Artifacts

- `selector_v4_mlx_cache_quality_gate_20260602T115659Z.json`
  - verdict: `FUNDAMENTAL_RENDERER_OUTPUT_DEGENERATE`
  - candidate SegNet cache: min=127, max=127, mean=127, std=0, dynamic_range=0
  - blockers include:
    `candidate_segnet_last_rgb_degenerate_constant_or_flat`,
    `candidate_segnet_last_rgb_dynamic_range_too_low`,
    `candidate_segnet_last_rgb_far_from_reference_fit_gate`

- `pact_vq_mlx_cache_quality_gate_20260602T115658Z.json`
  - verdict: `CACHE_INPUTS_NONDEGENERATE_LOCAL_ONLY`
  - candidate SegNet cache: min=5, max=116.4609375, std=14.274662971496582
  - still false-authority and not exact-dispatch-ready

- `nerv_control_inventory_20260602T115929Z.json`
  - control_count: 15
  - binding_gap_count: 19
  - fixed contest byte price: `6.658589531221714e-07` score/byte
  - top work orders:
    cache quality gate before profile/spend, measured HiNeRV model-size ladder,
    HiNeRV decoder-weight saliency/VJP wiring, MLX-native SNeRV train/export,
    SNeRV measured model-size ladder, push saliency into HiNeRV weight groups
    and SNeRV wavelet groups, RNeRV-style config search.

## Current Advisory Score Context

- PACT-VQ best local MLX research-signal gate:
  - archive bytes: 37,580
  - non-rate score: 90.63699250067236
  - rate score: 0.0250229794583312
  - total: 90.66201548013069
  - exact axis blocked because distortion is terrible and codec changes improve
    rate only.

- HiNeRV full-600 MLX prefilter:
  - bytes: 19,962
  - best full-video MLX score: 90.72321618617728
  - blocked by `mlx_score_above_hard_demote_threshold`.

- SNeRV full-600 LF predictor profile:
  - current LF payload bytes: 9,996,235
  - best simple lossless predictor bytes: 9,995,880
  - verdict: simple lossless LF predictors do not collapse the full-600 LF
    payload.

## Interpretation

The rate term remains the blocker of blockers, but rate alone is not the win.
Tiny carriers that fail distortion stay uncompetitive; large carriers that
solve distortion but cannot be aggressively quantized/ablated/waterfilled/packed
also stay uncompetitive. The correct engineering surface is the measured
rate-distortion ladder: sweep the carrier capacity and codec jointly, then let
the fixed contest byte price decide which size, weights, wavelet groups, tokens,
or supports deserve bytes.

`pact_nerv_selector_v4` is reactivated as an implementation bug lane, not a
method negative. The immediate fix is to inspect training/export/renderer cache
generation until the baseline scorer cache becomes nonconstant on real video
frames. After that, section value and model-size planning can matter.

`pact_nerv_vq` remains a saved-byte primitive, but current exact-axis spend is
blocked. The next useful VQ work is scorer-faithful retraining or RT/VQ-style
residual token rebuild, not another byte-only repack.

HiNeRV and SNeRV are not fully optimized. Both need source-faithful OSS controls
ported into the local MLX/NumPy surfaces and bound to scorer-aware training,
model-size budget ladders, cache gates, archive/runtime proof, and exact
CPU/CUDA replay.

## Verification

- `pytest` focused slice: 22 passed.
- `ruff` focused slice: all checks passed.
- `py_compile` touched implementation and CLI files: passed.
- Existing `src/tac/tests/test_compact_renderer_mlx_spine_runner.py` currently
  has 41 passing tests and one adjacent failure:
  `test_adapt_pr95_mlx_report_emits_spine_acquisition_and_runner`.
  The failure is an empty `selected_runner_rows` list caused by the current
  implementation-readiness gate blocking a 2-pair smoke fixture as partial
  coverage. I did not weaken that gate or overwrite the HPRC runner policy in
  this landing.

## Runnable Commands

Cache-gate selector-v4 baseline:

```bash
.venv/bin/python tools/gate_mlx_cache_quality.py \
  --candidate-cache-dir /Volumes/VertigoDataTier/pact/hprc_section_value_profiles/pact_nerv_pact_nerv_selector_v4_psv4_ccb349766791f52a/mlx_caches/baseline \
  --reference-cache-dir /Users/adpena/Projects/pact/experiments/results/mlx_scorer_input_cache_reference_video_20260521T2304Z_full600 \
  --output-json .omx/research/selector_v4_mlx_cache_quality_gate_$(date -u +%Y%m%dT%H%M%SZ).json
```

Build NeRV control inventory:

```bash
.venv/bin/python tools/build_nerv_control_inventory.py \
  --output-json .omx/research/nerv_control_inventory_$(date -u +%Y%m%dT%H%M%SZ).json
```

Plan a measured model-size ladder:

```bash
.venv/bin/python tools/plan_compact_carrier_modelsize_budget.py \
  --rows-json /Volumes/VertigoDataTier/pact/<carrier_modelsize_ladder_rows>.json \
  --carrier-id hi_nerv \
  --baseline-id pr95_hnerv \
  --output-json .omx/research/hi_nerv_modelsize_budget_plan_$(date -u +%Y%m%dT%H%M%SZ).json
```

## Next Work

1. Fix selector-v4 renderer/training/export until cache gate passes on real
   full-video scorer cache inputs.
2. Build measured HiNeRV and SNeRV model-size ladders; do not launch long runs
   from a single capacity point.
3. Wire decoder-weight saliency/VJP into HiNeRV full_main and wavelet-group
   saliency into SNeRV.
4. Make SR-NeRV resolution-axis training receiver-closed and scorer-preserving.
5. Add RNeRV-style config search over capacity, quantization, SR, saliency,
   codec, and packet layout.
6. Only after full-video local cache/replay wins, run byte-closed archive/runtime
   proof and exact contest CPU/CUDA replay.
