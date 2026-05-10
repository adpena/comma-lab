# Parallel dispatch exact-ready audit guard (2026-05-10)

## Result

`tools/parallel_dispatch_top_k.py` now runs the same exact-ready live-custody
audit used by `tools/operator_briefing.py` before any dry-run or provider
fan-out over selected candidates.

## Bug class

The actuator previously rechecked archive bytes/SHA and local candidate flags,
but did not consult terminal lane-claim evidence or the richer live custody
surface for `inflate.sh`, `report.txt`, `archive_manifest.json`, runtime-tree
SHA, and stale exact-ready rows. A generated queue could therefore remain
`ready_for_exact_eval_dispatch=true` after a same-lane terminal CUDA result had
already retired that exact archive/runtime.

## Guard

Selected rows are audited by candidate id before dispatch. The tool refuses
provider launch when the audit reports stale exact-ready blockers, including:

- active or stale nonterminal lane claims for the same canonical lane, including
  the legacy `lane_<id>` alias used by the Lightning PR106 stack wrapper;
- terminal CUDA score for the same lane/archive that does not beat the active
  floor;
- terminal negative/retired status for the same lane/archive;
- archive/runtime/report/manifest custody drift;
- missing or invalid runtime-tree SHA.

The same tranche also retired two bypass surfaces:

- `provider=vastai` in `tools/parallel_dispatch_top_k.py` is dry-run-only until
  the Vast.ai launcher owns a mandatory pre-instance dispatch claim and terminal
  claim update.
- `tools/feedback_loop_sweep.py --allow-paid-dispatch` now fails closed. The
  recovered feedback loop remains a research/dry-run scaffold; paid jobs must
  go through a promoted exact-ready queue and the audited actuator.

## Evidence

- Code: `tools/parallel_dispatch_top_k.py`
- Shared audit: `src/tac/optimizer/exact_ready_audit.py`
- Retired legacy paid path: `tools/feedback_loop_sweep.py`
- Regression: `tests/test_parallel_dispatch_top_k_exact_ready_audit.py`
- Regression: `tests/test_feedback_loop_sweep_retired_paid_dispatch.py`
- Regression: `src/tac/tests/test_optimizer_exact_ready_audit.py`

Focused verification:

```text
.venv/bin/python -m pytest tests/test_parallel_dispatch_top_k_exact_ready_audit.py tests/test_feedback_loop_sweep_retired_paid_dispatch.py src/tac/tests/test_optimizer_exact_ready_audit.py src/tac/tests/test_dispatch_command_builder_shapes.py
.venv/bin/python -m pytest tests/test_parallel_dispatch_top_k_exact_ready_audit.py src/tac/tests/test_optimizer_exact_ready_audit.py tests/test_audit_exact_ready_queues_cli.py src/tac/tests/test_operator_briefing.py tests/test_promote_optimizer_candidate_for_exact_eval_cli.py
.venv/bin/python -m py_compile tools/parallel_dispatch_top_k.py src/tac/optimizer/exact_ready_audit.py
```
