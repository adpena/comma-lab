# Codex Findings: Repair Child Component Replay Binding

UTC: 2026-05-27T02:31:50Z

## Verdict

Repair-budget child manifests now have a typed path from receiver-consumed
candidate archive proof into component-response replay evidence. The binding
layer can preserve the replay path, axis tag, and evidence grade, and the
execution audit can mark a spent-budget repair child ready for local
materialization only when archive materialization, receiver consumption, and
component replay evidence are all present.

## What Changed

- Extended repair materializer manifest ingestion to read
  `component_response_replay` evidence from direct child manifests.
- Propagated `component_response_replay_path`,
  `component_response_replay_axis_tag`, and
  `component_response_replay_evidence_grade` through the repair-budget
  materializer binding row.
- Taught execution rows to preserve replay evidence while keeping
  `budget_spend_allowed=false`, `ready_for_budget_spend=false`, and
  `ready_for_exact_eval_dispatch=false`.
- Added regression coverage for both sides of the gate:
  receiver-consumed child without component replay remains blocked, while a
  receiver-consumed child with `[macOS-MLX research-signal]` replay evidence
  becomes local-execution-ready only.

## Compliance

- The receiver remains decode-only. No receiver-side optimization, scorer
  inspection, sidecar fetch, or eval-time adaptation is introduced.
- MLX/local replay evidence is preserved as local materialization audit signal
  only. It cannot promote, rank, kill, spend budget, or dispatch exact eval.
- Exact score authority still requires contest CPU/CUDA auth-axis payloads with
  full custody.

## Verification

- `.venv/bin/ruff check src/comma_lab/scheduler/frontier_rate_attack_feedback.py src/tac/tests/test_repair_budget_materialization_execution.py`
- `.venv/bin/python -m pytest src/tac/tests/test_repair_budget_materialization_execution.py -q`
- `.venv/bin/python -m pytest src/tac/tests/test_frontier_rate_attack_feedback.py -q`

## Open Signal

The live queue still lacks direct repair-child materializer manifests carrying
component-response replay artifacts. This landing makes the queue able to
consume such manifests without losing axis/path evidence; the next actuator
slice is to generate those child manifests from MLX-local SegNet/PoseNet replay
probes and feed them into the existing binding report.
