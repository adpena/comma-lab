# Cost-Band NV7 Successful-Anchor Discipline (2026-05-12)

## Finding

REVIEW-OMNI surfaced that the cost-band posterior could treat failed provider
dispatches as empirical training-cost anchors. A short `returncode=1` crash
measures time-to-failure, not time-to-train, and can make GPU spend estimates
look hundreds of times cheaper than reality.

## Patch

- `tac.cost_band_calibration.CostBandAnchor` now records `outcome` and
  `returncode`.
- `predict()` and `summary_by_bucket()` exclude failed, timed-out, and partial
  anchors by default.
- `include_failed=True` remains available for explicit crash-time diagnostics.
- `append_platform_training_anchor()` derives outcome from `returncode` and
  `timed_out`.
- `tools/append_cost_band_anchor.py` accepts explicit `--outcome` and
  `--returncode`.
- `tools/migrate_cost_band_posterior_failed_anchors.py` tags pre-NV7 rows by
  reading `returncode=<nonzero>` from notes. It is dry-run by default and
  requires `--apply` to rewrite live state.

## Evidence

- `PYTHONPATH=src:upstream:$PWD .venv/bin/python -m pytest src/tac/tests/test_cost_band_calibration.py -q`
- `PYTHONPATH=src:upstream:$PWD .venv/bin/python -m ruff check pyproject.toml src/tac/cost_band_calibration.py tools/append_cost_band_anchor.py src/tac/tests/test_cost_band_calibration.py src/tac/source_index.py src/tac/tests/test_source_index.py`
- `GITHUB_ACTIONS=true PACT_PREFLIGHT_DISABLE_INCREMENTAL_CACHE=1 PACT_PREFLIGHT_PARALLEL_WORKERS=8 PYTHONPATH=src:upstream:$PWD .venv/bin/python -m tac.preflight`

## Score-Lowering Impact

This does not directly lower score. It prevents false-cheap GPU fanout and
forces the canary-first sequencing needed to spend on SIREN/HNeRV/Ballé
dispatches with calibrated cost evidence.
