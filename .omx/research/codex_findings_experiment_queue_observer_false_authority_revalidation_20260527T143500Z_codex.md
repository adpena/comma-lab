# Experiment Queue Observer False-Authority Revalidation

Generated: 2026-05-27T14:35:00Z
Author: Codex

## Finding

`experiment_queue_observer` could treat a passed
`json_false_authority`/`jsonl_false_authority` postcondition as sufficient to
surface materializer feedback and local MLX training signals. That made false
authority necessary but not enough: archive custody, receiver/runtime proof, and
local-training refusal evidence were not independently revalidated before the
observer promoted a succeeded artifact into operator-facing signal surfaces.

## Fix

- Added observer-side file custody revalidation for candidate archives.
- Added nested optimizer `top_k` materializer row revalidation.
- Added receiver/runtime proof path checks and JSON proof blocker checks.
- Added local-training signal revalidation that requires either independent
  proof or explicit readiness blockers/exact-readiness refusal.
- Wired revalidation blockers into `succeeded_artifact_steps`,
  `succeeded_signal_steps`, health, and blocker reporting.

## Regression Coverage

- Materializer manifests with declaration-only proof flags no longer surface.
- Optimizer `top_k` materializer rows with missing archive/proof custody no
  longer surface.
- Local MLX training signals without readiness blockers no longer emit
  replanning observations.

## Verification

```
.venv/bin/python -m pytest src/tac/tests/test_experiment_queue_observer.py -q
# 22 passed in 2.53s

.venv/bin/ruff check src/comma_lab/scheduler/experiment_queue_observer.py src/tac/tests/test_experiment_queue_observer.py
# All checks passed

.venv/bin/python -m py_compile src/comma_lab/scheduler/experiment_queue_observer.py
```
