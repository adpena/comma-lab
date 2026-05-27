# 5D Follow-Up Execution Queue And Bundle Parser Hardening

UTC: 2026-05-27T06:23:00Z
Agent: codex

## Verdict

The 5D coverage-acquisition lane now has a second queue-owned bridge: emitted
follow-up readiness rows can be compiled into an execution queue when their
inputs are concrete, while all-blocked reports compile to a disabled sentinel
instead of silently disappearing or granting false readiness.

## What Changed

- Added `build_coverage_followup_execution_queue(...)` and
  `tools/build_5d_coverage_followup_execution_queue.py`.
- Exact-axis follow-up rows remain operator-gated and frozen by default; only
  dry-run `tools/paired_auth_eval_cli.py` commands are accepted.
- MLX negative-delta rows become `local_mlx` queue work only when the readiness
  report has concrete cache manifests and archive bytes.
- Added `submission_bundle_result_from_dict(...)` as the canonical inverse of
  `SubmissionBundleResult.as_dict`.
- Routed paired-auth, compliance, and linter CLIs through the canonical parser
  instead of maintaining divergent bundle reconstruction code.
- Tightened exact-axis readiness so schema-only bundle JSON no longer marks a
  request ready; the full canonical bundle contract must parse.

## Authority

The execution queue is still false-authority by construction. It does not claim
score, promote, rank, kill, or dispatch paid exact eval. Exact-axis rows are
frozen dry-run plans until an operator or higher-level actuator deliberately
unfreezes them after dispatch-claim discipline.

## Verification

- `.venv/bin/python -m ruff check --fix src/tac/submission_packet/builder.py src/tac/submission_packet/__init__.py tools/paired_auth_eval_cli.py tools/submission_linter_cli.py tools/submission_compliance_cli.py src/comma_lab/scheduler/pair_frame_5d_coverage_acquisition_queue.py src/comma_lab/scheduler/__init__.py tools/build_5d_coverage_followup_execution_queue.py src/tac/tests/test_pair_frame_5d_coverage_acquisition_queue.py`
- `.venv/bin/python -m pytest src/tac/tests/test_pair_frame_5d_coverage_acquisition_queue.py src/tac/tests/test_pair_frame_5d_extended_operator_queue.py src/tac/tests/test_pair_frame_scorer_geometry_lattice_5d_canvas_coverage.py src/tac/tests/test_modal_paired_dispatch_contract.py src/tac/tests/test_submission_packet_paired_auth_eval.py src/tac/tests/test_submission_compliance.py -q`
- Live smoke at `/tmp/pact_5d_followup_execution_smoke_20260527T062241Z`:
  current readiness emitted 2 blocked requests and 0 ready requests; the
  follow-up execution queue validated with one disabled `no_ready_followup_requests`
  sentinel and `score_claim=false`.

Result: 229 tests passed; live queue validation returned `valid=true`.
