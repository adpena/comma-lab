# Codex Findings: Materializer Runner Generated Queue Identities

UTC: 2026-05-24T18:25:38Z
Lane: `codex_materializer_runner_auto_queue_identity_20260524`
Author: Codex

## Verdict

The materializer campaign runner now auto-generates local queue-performance
runtime and cache identity artifacts when the operator does not provide pinned
identity inputs. A successful telemetry-producing runner invocation can now
emit a `queue_feedback_replan_request.json` that is ready for the inverse
steganalysis action-functional builder without requiring ad hoc follow-up
identity files.

This closes a concrete feedback-loop gap in the byte-shaving materializer
campaign path: queue performance events can be passed forward with runtime and
cache identity, but the generated identities are explicitly false-authority and
are valid only as local queue-performance observation identity.

## What Changed

- `tools/run_byte_shaving_materializer_campaign.py` writes:
  - `queue_performance_runtime_identity.json`
  - `queue_performance_cache_identity.json`
- The runner records both paths in the run summary and in
  `queue_feedback_replan_request.json`.
- Replan readiness now uses the effective identity paths, whether supplied by
  CLI flags or generated locally.
- The generated identity payloads carry `score_claim=false` and forbidden-use
  metadata so they cannot become score, promotion, dispatch, rank, or kill
  authority.
- Runner tests assert that the replan request becomes ready only when the queue
  performance summary is valid and has completed step events, and that the
  generated identity payloads are consumable by the canonical queue-performance
  observation converter.

## Verification

- `.venv/bin/python -m pytest src/tac/tests/test_byte_shaving_materializer_campaign_runner.py -q`
  - Result: `37 passed`
- `.venv/bin/python -m pytest src/tac/tests/test_inverse_steganalysis_action_functional_cli.py src/tac/tests/test_inverse_steganalysis_acquisition.py src/tac/tests/test_experiment_queue.py src/tac/tests/test_byte_shaving_campaign_queue.py::test_inverse_surface_cells_compile_to_action_functional_work_queue -q`
  - Result: `101 passed`
- `.venv/bin/python -m ruff check tools/run_byte_shaving_materializer_campaign.py src/tac/tests/test_byte_shaving_materializer_campaign_runner.py src/tac/tests/test_byte_shaving_campaign_queue.py`
  - Result: `All checks passed!`
- `.venv/bin/python tools/lane_maturity.py validate`
  - Result: `OK`
- `git diff --check`
  - Result: clean

## Remaining Gaps

- Cross-run comparison can be made stronger by allowing run configs to pin
  externally supplied runtime/cache identity artifacts instead of relying on the
  local generated identities.
- The higher-EV frontier work remains the reusable operation-set compiler and
  materializer expansion: HNeRV/NeRV/bolton/non-NeRV candidates need multiple
  deterministic materializers, not only the current chain.
- The action-functional feedback consumer must continue to treat these local
  queue identities as observation identity only; exact CPU/CUDA auth eval
  remains the only score and promotion authority.

## 6-Hook Wire-In

- Sensitivity map: indirect. The generated identity artifacts make queue
  performance observations reusable by the inverse action-functional builder.
- Pareto constraint: indirect. Readiness blockers prevent invalid queue
  performance from entering Pareto feedback.
- Bit allocator: indirect. Feedback-ready queue summaries can inform later byte
  allocator decisions after canonical action-functional conversion.
- Cathedral/autopilot dispatch: active. The runner now emits a complete replan
  request artifact that an autopilot can consume without manual identity-file
  recovery.
- Continual-learning posterior: active once the replan request is consumed by
  the canonical response/action-functional path.
- Probe disambiguator: active. Generated identity vs externally pinned identity
  is explicit in the artifact so consumers can distinguish local observation
  identity from stronger cross-run provenance.
