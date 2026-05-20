# Codex Findings: Inflate Postprocess Surface Smoke

UTC: 2026-05-20T20:27:52Z
Owner: Codex
Lane: `lane_inflate_postprocess_surface_smoke_20260520`
Status: L1 `research_only=true`; no score claim; no live PR110 files edited.

## Summary

Built and ran the first bounded execution artifact for `inflate.py`-side techniques:

- `src/tac/optimization/inflate_postprocess_surface.py`
- `tools/run_inflate_postprocess_smoke.py`
- `src/tac/tests/test_inflate_postprocess_surface.py`

The tool applies deterministic raw-output transforms to an already-inflated raw file, then optionally runs `tools/run_raw_advisory_eval.py`. It is deliberately advisory-only: a positive result must be converted into a stock `inflate.py` runtime candidate with charged archive/runtime parameter custody before exact eval.

## Inputs

- Baseline raw: `experiments/results/pr110_provisional_hfv1_engineering_20260520_codex/contest_oracle_search/op3v3_decoder_q_inflate_controls_20260520_codex/baseline/inflated/0.raw`
- Archive used for advisory rate: `experiments/results/pr110_provisional_hfv1_engineering_20260520_codex/runtime_hfv1/archive.zip`
- Baseline score for deltas: `0.19206142414659494`
- Axis: `[macOS-CPU advisory inflate-postprocess]`

## Results

Artifacts:

- Luma smoke: `experiments/results/pr110_provisional_hfv1_engineering_20260520_codex/contest_oracle_search/inflate_postprocess_smoke_20260520_codex/summary.json`
- Temporal smoke: `experiments/results/pr110_provisional_hfv1_engineering_20260520_codex/contest_oracle_search/inflate_postprocess_temporal_smoke_20260520_codex/summary.json`

| spec | score | delta vs baseline | PoseNet | SegNet | changed frames | changed bytes | verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `odd_luma_bias_m1` | 0.199360779360876 | +0.007299355214281 | 0.00004515 | 0.00059245 | 600 | 1,677,315,167 | negative |
| `odd_luma_bias_p1` | 0.201528326857261 | +0.009466902710666 | 0.00005485 | 0.00059241 | 600 | 1,828,367,205 | negative |
| `odd_temporal_blend_a1_8` | 0.459389874395422 | +0.267328450248827 | 0.00054044 | 0.00267008 | 600 | 1,725,058,834 | strongly negative |

All three candidates produced visible raw-output changes and scored successfully. All three are worse than baseline. Candidate raws were deleted after successful advisory scoring; the artifact directories are small retained manifests/logs only.

## Interpretation

This falsifies the easiest constant-source postfilter ideas for the current FEC6/PR110 raw:

- Uniform odd-frame luma nudges are too blunt. Even a single RGB level on odd frames materially worsens PoseNet and SegNet.
- Naive temporal blending is catastrophic for PoseNet. The odd-frame 12.5% blend toward neighboring frames increased score by `+0.2673`.

This does **not** kill the `inflate.py` surface. It narrows it:

- Do not spend exact-eval budget on constant brightness or naive temporal smoothing.
- Do not add a generic source-only postfilter to PR110.
- The next runtime-side candidates need learned/charged parameters, scorer-informed pixel residuals, or geometry-aware transforms, not global constants.

## Next Frontier-Moving Path

Highest-EV next step:

1. Build a charged residual/postprocess parameter carrier: a tiny archive-side table or weight-embedded circuit that affects only top sensitivity pixels/frames.
2. Use master-gradient and pair/pixel sensitivity to pick sparse locations.
3. Apply the generic runtime operation in `inflate.py` only after the parameters are carried by charged bytes.
4. Run raw advisory first, then stock-inflate runtime custody, then exact eval only if advisory improves.

The clean surface is not "postprocess everything." It is "generic deterministic operator + charged sparse parameters + sensitivity-selected target set."

## Verification

Commands:

```bash
.venv/bin/python -m pytest -q \
  src/tac/tests/test_inflate_postprocess_surface.py \
  src/tac/tests/test_decoder_q_decision_packet.py \
  src/tac/tests/test_fec6_decoder_mutations.py \
  src/tac/tests/test_percepta_microprogram_plan.py
```

Result: `19 passed`.

Lane registry:

- `lane_inflate_postprocess_surface_smoke_20260520` added at L0.
- Marked `impl_complete=true`, now L1.
- `research_only=true`.
- Reactivation criterion: charged learned/sidecar postprocess parameters or scorer-informed pixel residual proposals; constant luma and naive temporal blend are advisory-dominated.
