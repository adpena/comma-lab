# Codex Findings: Local CPU Eureka Input Guard

UTC: 2026-05-23T00:35:11Z

## Evidence Axis

This memo records a guardrail change for `[macOS-CPU advisory]` local scorer
evidence feeding `[contest-CPU drift-projected; false authority]` spend-triage
signals. It is not score authority, not promotion evidence, and not rank/kill
evidence.

## Finding

The local CPU drift/eureka helper previously blocked non-macOS-CPU axis labels,
but it did not model three malformed-input classes before computing a eureka
trigger:

- candidate local JSON with missing or non-advisory `evidence_semantics`;
- candidate local JSON that was not a full 600-sample public-test advisory row;
- candidate local JSON carrying truthy score/promotion/dispatch authority flags.

This was not observed in the completed DQS1 queue artifacts, but it was a bug
class at the helper boundary: a malformed local JSON could still be converted
into an eureka artifact instead of being explicitly marked blocked.

## Fix

`src/tac/optimization/local_cpu_contest_drift.py` now exposes
`local_cpu_advisory_payload_blockers(...)` and threads those blockers into the
candidate trust-region gate before any eureka trigger can fire.

The helper now requires:

- an accepted local CPU advisory axis label, including bracketed
  `[macOS-CPU advisory]` labels;
- `evidence_semantics == "non_contest_cpu_auth_eval_advisory"`;
- `n_samples == 600`;
- no truthy forbidden authority fields.

It also refuses boolean numeric values as scores or component floats.

## Tests

Added coverage in `src/tac/tests/test_local_cpu_contest_drift.py` for:

- partial local JSON plus truthy authority flags forcing `observe_only`;
- bracketed `[macOS-CPU advisory]` labels remaining valid;
- existing MPS-axis rejection under the stricter advisory payload contract.

Verification:

- `.venv/bin/python -m ruff check src/tac/optimization/local_cpu_contest_drift.py src/tac/tests/test_local_cpu_contest_drift.py`
- `.venv/bin/python -m pytest -q src/tac/tests/test_local_cpu_contest_drift.py`
- `.venv/bin/python tools/experiment_queue.py --queue configs/experiment_queues/dqs1_pairset_local_first.yaml validate`
- `.venv/bin/python -m pytest -q src/tac/tests/test_dqs1_local_first_queue_builder.py src/tac/tests/test_local_cpu_contest_drift.py`

## Integration

The rank029 eureka artifact remained observe-only with no candidate blockers:

`.omx/research/local_cpu_contest_drift_eureka_pairset_drop_one_rank029_pair0259_20260523T003158Z.json`

The local-first queue is now routed to:

`pairset_drop_one_rank008_pair0496`

Any future malformed local advisory eureka input remains a false-authority row
and cannot become an exact-auth-anchor spend trigger.
