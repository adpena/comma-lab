# A1 Inflate-Time Bias Sweep Exact CPU Review (Codex, 2026-05-09)

## Summary

The A1 zero-byte inflate-bias hypothesis produced a useful exact-negative
result, not a new champion. The half-magnitude PR101 bias variant was evaluated
on GHA Linux x86_64 and regressed:

| candidate | evidence | score | seg | pose | bytes | verdict |
|---|---|---:|---:|---:|---:|---|
| A1 baseline | `[contest-CPU]` run `25588422622` | 0.192847577437 | 0.00056023 | 0.00003286 | 178262 | current CPU anchor |
| V2 half-magnitude | `[contest-CPU]` run `25598904298` | 0.194295755690 | 0.00057342 | 0.00003333 | 178262 | measured-config regression |

Classification: exact CPU negative for the V2 half-magnitude config only.
It does not kill inflate-time bias search. The baseline re-run and concurrent
variant runs confirm that the full PR101 inherited bias remains the local best
among the observed GHA CPU results so far.

## Harvester Bug Found And Fixed

The first local dispatcher attempt returned workflow run `25598904351`, whose
artifact was `eval-a1_bias_v7_pr101_pr102_stack_20260509`, not the requested
`a1_bias_correction_sweep_v2_half_magnitude_20260509T103000Z`.

Root cause: `tools/dispatch_cpu_eval_via_github_actions.py` selected the newest
workflow run after `workflow_dispatch`. Concurrent GHA dispatches in the same
minute made that non-custodial.

Fix: the dispatcher now identifies the matching workflow by scanning new run
logs for the requested `submission_name`, and its artifact download fallback
selects `report.txt` by matching `submission_dir: submissions/<submission_name>`.
Tests cover both the wrong-concurrent-submission rejection and the fallback
artifact selector.

## Additional Observed GHA CPU Signals

These were downloaded from the fork's completed workflow artifacts while
debugging the harvester race. Treat them as observed external artifacts unless
their corresponding local custody rows are attached by the owning agent.

| submission | score | seg | pose | note |
|---|---:|---:|---:|---|
| `a1_bias_v1_baseline_20260509` | 0.192847577437 | 0.00056023 | 0.00003286 | matches A1 |
| `a1_bias_v7_pr101_pr102_stack_20260509` | 0.192930137424 | 0.00056023 | 0.00003316 | slight regression |
| `a1_bias_v2_half_magnitude_20260509` | 0.194295755690 | 0.00057342 | 0.00003333 | regression, same as Codex V2 |
| `a1_bias_v6_pr102_pattern_20260509` | 0.195165755690 | 0.00058212 | 0.00003333 | regression |
| `a1_bias_v0_control_20260509` | 0.195213986806 | 0.00058211 | 0.00003351 | removing PR101 bias is bad |
| `a1_bias_v5_opposite_sign_20260509` | 0.198381186140 | 0.00060885 | 0.00003534 | strong regression |

Implication: the PR101 bias block is load-bearing for A1. The next non-arbitrary
retry is not broad sign/magnitude guessing; it is a small local coordinate
search around the full PR101 bias with exact same-archive controls, or a
sidecar/latent correction search that preserves the full PR101 bias.

## New Tooling

- `tools/build_a1_inflate_time_bias_correction_sweep.py` builds same-archive
  runtime-bias variants with per-variant manifests and no score promotion.
- `tools/build_a1_per_pair_latent_correction_sidecar.py` smoke-proved the
  sidecar resampling path changes charged bytes; the proxy now uses upstream
  `frame_utils.yuv420_to_rgb` rather than raw PyAV `rgb24`.
- `tools/xray_archive_section_entropy_heatmap.py` emits diagnostic-only archive
  entropy heatmaps for byte-target selection.

## Reactivation Criteria

- Exact CPU/CUDA pair for a coordinate-search variant that preserves the full
  PR101 bias and perturbs one channel by a smaller increment than the V2
  half-magnitude change.
- Full 600-pair sidecar resample with runtime preflight, no-op proof, lane
  claim, and exact GHA CPU result.
- Modal/Lightning CUDA replay of any CPU-positive runtime-only candidate before
  internal promotion.
