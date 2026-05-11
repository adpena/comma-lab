# CPU/CUDA exact-pair mechanism analyzer (Codex, 2026-05-11)

## Purpose

Convert CPU/CUDA drift interpretation from prose into a reusable, non-promoting
tool that can ingest exact auth-eval JSON pairs across HNeRV alternatives,
stacked wrappers, full-pipeline packets, and future non-HNeRV substrates.

## Landing

Tool upgraded: `tools/analyze_cpu_cuda_eval_drift.py`

New exact-pair mode:

```bash
.venv/bin/python tools/analyze_cpu_cuda_eval_drift.py \
  --exact-pair CPU_JSON CUDA_JSON \
  --json-out OUT/analysis.json \
  --markdown-out OUT/analysis.md
```

The analyzer records:

- archive SHA/bytes parity
- runtime tree parity
- inflated raw-output aggregate SHA parity when available
- CPU/CUDA PoseNet, SegNet, rate, and score-term deltas
- mechanism class:
  - `same_raw_outputs_scorer_or_loader_drift`
  - `different_raw_outputs_runtime_or_inflate_drift`
  - `same_archive_runtime_raw_outputs_unmeasured`
  - `custody_incomplete`

It always emits `score_claim=false`, `promotion_eligible=false`, and
`rank_or_kill_eligible=false`.

## PR103-on-PR106 v2 calibration run

Artifact:
`experiments/results/dual_device_auth_eval/pr103_pr106_dual_runtime_mechanism_analysis_20260511T030251Z/analysis.json`

Result:

- pair valid for mechanism analysis: `true`
- same archive SHA: `true`
- same runtime tree SHA: `true`
- raw output status: `raw_output_manifest_missing`
- mechanism class: `same_archive_runtime_raw_outputs_unmeasured`
- CUDA minus CPU score gap: `-0.02067461068280979`

Interpretation: PR103-on-PR106 remains a real CUDA-better full-pipeline result,
but existing artifacts predate raw-output hashing. The next exact rerun should
close the remaining raw-output-custody gap before attributing the mechanism to
runtime/inflate drift versus scorer/loader drift.

## Verification

```bash
.venv/bin/python -m pytest \
  src/tac/tests/test_analyze_cpu_cuda_eval_drift.py \
  src/tac/tests/test_auth_eval_records.py \
  src/tac/tests/test_plan_dual_device_auth_eval.py \
  src/tac/tests/test_harvest_cuda_cpu_axis_profile_registry.py -q
```

Result: `32 passed`.
