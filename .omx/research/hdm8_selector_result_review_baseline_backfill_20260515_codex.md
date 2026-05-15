# HDM8 Selector Result-Review Baseline Backfill - 2026-05-15

score_claim: false
promotion_eligible: false
ready_for_exact_eval_dispatch: false

## Fix

`tools/build_result_review_packet.py` no longer labels exact-CUDA reviews with
no supplied baseline as `not_negative_against_supplied_baseline`. Missing
baseline now becomes:

- `failure_class=exact_cuda_result_reviewed_baseline_missing`

This prevents machine consumers from interpreting "no baseline supplied" as a
positive or neutral result.

## Backfilled HDM8 Reviews

Regenerated two HDM8 selector/film-grain exact-CUDA review packets with the
matched HDM8 fixed-length exact-CUDA baseline:

- baseline score: `0.20636166502462222`
- baseline axis: `[contest-CUDA]`

Backfilled packets:

- `.omx/research/hdm8_film_grain_selector_charged_mps_aggressive_v2_exact_cuda_result_review_20260515_codex.json`
- `.omx/research/hdm8_film_grain_selector_charged_mps_aggressive_v2_exact_cuda_evidence_row_20260515_codex.json`
- `.omx/research/hdm8_even_frame_selector_exact_cuda_result_review_20260515_codex.json`
- `.omx/research/hdm8_even_frame_selector_exact_cuda_evidence_row_20260515_codex.json`

Both now classify as:

- `measured_config_status=measured_config_retired`
- `failure_class=legitimate_score_regression_or_component_collapse`
- evidence grade `[contest-CUDA A-negative]`
- `promotion_eligible=false`
- `ready_for_exact_eval_dispatch=false`

## Verification

```bash
PYTHONPATH=src:upstream:$PWD .venv/bin/python -m pytest -p no:cacheprovider -q \
  src/tac/tests/test_build_result_review_packet.py \
  src/tac/tests/test_pr106_latent_sidecar_recode.py
```

Observed: `17 passed`.

```bash
.venv/bin/ruff check \
  tools/build_result_review_packet.py \
  tools/profile_pr106_latent_sidecar_recode.py \
  src/tac/tests/test_build_result_review_packet.py \
  src/tac/tests/test_pr106_latent_sidecar_recode.py
```

Observed: `All checks passed`.

## Impact

This is result-custody hardening, not a score claim. It blocks accidental broad
film-grain/waterfill redispatch from stale "not negative" metadata and keeps
the exact-CUDA calibration table consistent with the individual review packets.
