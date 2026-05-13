# Modal harvest summary signal preservation - 2026-05-13

## Scope

Repeated bulk harvest of `experiments/results/lane_*_modal/modal_metadata.json`
must not degrade the aggregate custody file
`experiments/results/_modal_harvest_summary.json`.

## Bug class

`tools/harvest_modal_calls.py` skipped provider calls that already had
`harvested_artifacts/`, but rewrote the aggregate summary row using only
`status=already_harvested`, `call_id`, `cost_band_anchor`, and
`terminal_claim`. That dropped the original `rc`, `elapsed_seconds`,
`timed_out`, `n_artifacts`, and `crash_kind` fields from already-harvested
calls. This is signal loss in the provider lifecycle ledger, not just display
churn.

## Fix

- Added `tac.deploy.modal.harvest_summary.modal_training_summary_entry`.
- Rewired `tools/harvest_modal_calls.py` so already-harvested rows read
  `harvested_artifacts/_harvest_summary.json` once and preserve result fields.
- Added a focused regression test:
  `src/tac/tests/test_modal_training_harvest_summary.py`.

## Current harvest state

Command:

```bash
PYTHONPATH=src:upstream:$PWD .venv/bin/python tools/harvest_modal_calls.py
```

Aggregate summary:

- rows: `65`
- rows with `rc`: `38`
- already-harvested rows with preserved `crash_kind`: `38`
- no live `not_ready` Modal training calls observed
- unrecoverable historical provider states: `6` `error_NotFoundError`,
  `1` `expired`, `1` `error_ModuleNotFoundError`

Classification: training/provider custody only. `score_claim=false`; no
promotion or score-lowering claim is made from Modal training harvest rows.

## Verification

```bash
PYTHONPATH=src:upstream:$PWD .venv/bin/python -m pytest \
  src/tac/tests/test_modal_training_harvest_summary.py -q
```

Result: `1 passed`.
