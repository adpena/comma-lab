# Codex Findings: DP1 Modal Harvest Runtime Preservation

Timestamp: 2026-05-21T03:30:15Z

## Verdict

PROCEED. The corrected DP1 baseline/procedural Modal calls are still running,
but the pre-existing Modal recovery path had a harvest blocker that could have
made a successful run non-dispatchable for paired CPU/CUDA auth eval.

## Findings Fixed

1. `experiments/modal_train_lane.py` flattened files collected from
   `/modal_results/<label>/...` to basename-only artifact keys and filtered out
   `.sh` / `.py`. A DP1 full run writes `output/submission/inflate.sh` and
   `output/submission/inflate.py`; losing those files would block
   `tools/plan_dp1_procedural_paired_harvest.py`.

2. `experiments/modal_recover_lane.py` used the timestamped label as
   `lane_id` for terminal Modal call-ledger events. The recovered
   `modal_metadata.json` already carries the canonical recipe lane id and
   should be authoritative for terminal rows.

3. `tools/operator_authorize.py` accepted `--agent`, but
   `experiments/modal_train_lane.py` hardcoded call-id ledger rows to
   `agent="claude"`. The caller agent is now threaded through to Modal so
   future dispatch provenance is honest.

## Verification

- `.venv/bin/python -m py_compile experiments/modal_train_lane.py experiments/modal_recover_lane.py tools/operator_authorize.py`
- `.venv/bin/python -m pytest -q src/tac/tests/test_modal_train_lane_hardening.py src/tac/tests/test_modal_train_lane_silent_no_spawn_fix.py src/tac/tests/test_remote_auth_eval_hardening.py::test_modal_recover_terminal_ledger_metadata_uses_recovered_lane_id src/tac/tests/test_remote_auth_eval_hardening.py::test_modal_recover_terminal_ledger_metadata_falls_back_without_metadata src/tac/tests/test_remote_auth_eval_hardening.py::test_modal_recover_closes_terminal_claim_on_artifact_materialization_failure`
- `git diff --check`

Known unrelated test issue: the full `src/tac/tests/test_remote_auth_eval_hardening.py`
module currently contains older fixtures missing `score_axis`, which fail the
newer adjudicator gate before reaching the modal recovery tests.

