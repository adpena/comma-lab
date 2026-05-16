# L5 v2 prediction-band current ledger custody

Date: 2026-05-16

Scope: `src/tac/optimization/l5_staircase_v2.py`

Finding:

The L5-v2 prediction-band payload carried the expanded `time_traveler_l5_v2`
research-basis IDs, but its local ledger paths still pointed only at older
20260513/20260514 campaign memos. That made paper/source custody harder to
audit after the 20260516 literature and proof landings.

Change:

`l5_v2_prediction_band_payload().band_source.local_ledger_paths` now includes
the 20260516 paper-fidelity, source-basis, neural-video-codec source-basis,
sidecar, and TT5L side-info proof ledgers. This is provenance only; the
prediction band remains rank-blocked until an empirical L5-v2 anchor exists.

Verification:

- `.venv/bin/python -m pytest src/tac/tests/test_l5_staircase_v2.py -q`
- `.venv/bin/ruff check src/tac/optimization/l5_staircase_v2.py src/tac/tests/test_l5_staircase_v2.py`
