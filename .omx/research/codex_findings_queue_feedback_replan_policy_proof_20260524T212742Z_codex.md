# Codex Findings: Queue Feedback Replan Policy Proof

UTC: 2026-05-24T21:27:42Z
Lane: codex_queue_feedback_replan_policy_loop_20260524

## Finding

The inverse-steganalysis spine is partially real and wired, but it is not yet a
frontier-lowering automated final-rate attack. The strongest implemented path is
strict MLX/scorer-response signal with compiler hints -> action functional ->
signal surface -> byte-shaving campaign/PacketIR -> local materializer queue ->
queue feedback -> bounded local replan. The weak point is execution coverage:
many action cells still become planning records or fixture-backed materializers
instead of real family-specific archive/runtime rewrites.

The local feedback policy itself also needed to become proof-carrying. A summary
boolean that says a feedback child queue was emitted is not enough authority for
an autopolicy to resume or continue a loop.

## Patch

- Added `queue_feedback_replan_policy.v1` as a reusable scheduler policy record.
- Added `queue_feedback_replan_child_queue_validation.v1`, including child queue
  SHA-256, queue schema, paused/local-first/local-CPU proof, command proof,
  output path confinement, and recursive false-authority validation.
- Wired the materializer campaign runner to emit
  `queue_feedback_replan_policy.json` and to share the same validator for the
  local autopolicy guard.
- Hardened command validation against shell/remote wrappers, forbidden exact or
  provider flags including `--flag=value`, non-local resources, and outputs
  outside the campaign run directory.
- Added policy blockers for truncated queue performance telemetry before local
  continuation.
- Surfaced policy path, decision, blocker count, and continuation count in
  operator briefing.

## Authority

This is local feedback-loop automation only. It is not score authority,
promotion authority, rank/kill authority, paid dispatch authority, or exact eval
authority. Exact handoffs remain separate recommended actions and still require
normal lane claim, runtime custody, and contest CPU/CUDA auth-axis payload gates.

## Sidecar Audit Integration

Descartes reviewed the larger inverse-steganalysis implementation surface and
found the current system is still too leaf-greedy. The action functional covers
byte/pixel/region/frame/pair/batch/full-video coordinates, and strict MLX rows
can carry operation-set compiler hints, but the active water-fill path is still
a sorted local fill rather than a recomputing combinatorial portfolio optimizer.
Executable materializer coverage is also thin: archive-section entropy recode,
packet-member recompress, and tensor-factorize are the main real families;
HNeRV/PR95/NeRV-specific receiver rewrites, header elision, member reorder/merge,
tensor quantize/prune/codebook, and decoder-q actions need real adapters.

The next frontier-relevant implementation axis is therefore not more prose. It is
more executable materializers plus a portfolio search actuator that recomputes
interactions and queue-cost feedback across operation families.

## 4-Week Tranche

1. Week 1: land the queue-feedback policy under `experiment_queue.v1`, require
   storage preflight or an explicit waiver for heavy local artifact loops, add a
   local host/IO claim, and run one no-paid real artifact campaign through the
   existing materializer queue with feedback harvest.
2. Week 2: implement family-specific compiler/materializer adapters for the
   highest-EV HNeRV/PR95 archive sections and packet members.
3. Week 3: replace greedy water-fill with beam/knapsack portfolio search over
   bytes, regions, frame-pairs, batches, substrate families, queue cost, and
   observed feedback.
4. Week 4: close proof/handoff for best candidates: runtime-consumption proof,
   full-frame parity, exact-readiness dry-run, and paused exact-eval consumer
   queue. Only exact CPU/CUDA auth artifacts can become score-relevant.

## Verification

- `.venv/bin/python -m ruff check src/comma_lab/scheduler/queue_feedback_replan_policy.py tools/run_byte_shaving_materializer_campaign.py src/comma_lab/scheduler/__init__.py src/tac/tests/test_queue_feedback_replan_policy.py src/tac/tests/test_byte_shaving_materializer_campaign_runner.py src/tac/tests/test_operator_briefing.py`
  - clean
- `.venv/bin/python -m pytest src/tac/tests/test_queue_feedback_replan_policy.py src/tac/tests/test_byte_shaving_materializer_campaign_runner.py src/tac/tests/test_operator_briefing.py -q`
  - 83 passed
- `.venv/bin/python -m pytest src/tac/tests/test_byte_shaving_campaign.py src/tac/tests/test_inverse_steganalysis_acquisition.py src/tac/tests/test_byte_shaving_campaign_queue.py src/tac/tests/test_staircase_dag.py -q`
  - 135 passed
- `tools/review_tracker.py policy-check` on touched Python surfaces
  - clean

## Remaining Gaps

- Move the continuation decision deeper into durable experiment-queue ownership
  instead of runner-owned policy execution.
- Implement family-specific adapters so higher-order action cells materialize
  real archive/runtime rewrites.
- Replace greedy water-fill with a portfolio/interaction optimizer.
- Preserve strict local MLX/proxy versus contest auth-axis boundaries.
