# CPU/CUDA drift mechanism validity guard (2026-05-11)

## Scope

The PR103-on-PR106 paired exact artifacts are valid for score-axis comparison,
but not yet valid for runtime-vs-scorer mechanism attribution because both
older artifacts lack `inflated_outputs_manifest.json`.

This pass hardens the analysis/planning tools so they cannot silently promote
that missing raw-output custody into a mechanism conclusion.

## Code changes

- `tools/analyze_cpu_cuda_eval_drift.py`
  - separates `valid_for_pair_score_analysis` from
    `valid_for_mechanism_analysis`;
  - reports `mechanism_blockers` when raw-output manifests are missing or
    partial;
  - keeps the pair non-promotable and non-ranking.
- `tools/plan_dual_device_auth_eval.py`
  - separates `paired_score_artifacts_complete` from
    `drift_mechanism_complete`;
  - keeps `global_priority_eligible=false` until raw-output manifests are
    paired, while still allowing score-pair custody to be recognized.
- `tools/harvest_cuda_cpu_axis_profile_registry.py`
  - preserves `mechanism_analysis_complete` and `mechanism_blockers` inside
    the registry payload's `inflated_outputs` section.

## Verification

```bash
.venv/bin/python -m pytest -q \
  src/tac/tests/test_analyze_cpu_cuda_eval_drift.py \
  src/tac/tests/test_plan_dual_device_auth_eval.py \
  src/tac/tests/test_harvest_cuda_cpu_axis_profile_registry.py \
  src/tac/tests/test_auth_eval_records.py
```

Result: `33 passed`.

## PR103-on-PR106 reclassification

Rerunning the exact-pair analyzer on the existing artifacts now reports:

- `valid_for_pair_score_analysis=true`
- `valid_for_mechanism_analysis=false`
- `mechanism_class=same_archive_runtime_raw_outputs_unmeasured`
- `mechanism_blockers=["raw_output_manifest_missing"]`

This preserves the real signal: CUDA scored lower than CPU for this exact
same-archive/same-runtime-content pair, but the cause is not yet localized.

## Score-lowering implication

Do not use this pair to conclude that runtime arithmetic, DALI/PyAV loader
behavior, or scorer kernels caused the gap until paired raw-output aggregate
hashes exist. The next exact paired run must use the Modal raw-output harvest
fix from `modal_auth_eval_raw_output_harvest_20260511_codex.md`.
