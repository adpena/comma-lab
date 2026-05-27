# Codex Findings: Repair Stackability Posterior Ingest

Date: 2026-05-27T16:44:05Z

## Verdict

Repair stackability learning signals now have a durable append-only posterior
ingest path. The local MLX signal still has no score, budget, promotion, or
dispatch authority, but it is no longer stranded as a per-run artifact: the
queue can append it to a locked JSONL posterior with deterministic row ids and
duplicate suppression.

## Landed Integration

- `repair_campaign_posterior` builds false-authority posterior rows from
  `repair_campaign_learning_signal.v1` artifacts.
- Posterior row identity is deterministic from typed response id, candidate,
  family, replay hash, source-record hash, and replay argv hash.
- `append_repair_campaign_stackability_posterior_signal(...)` writes under an
  fcntl lock, appends only new row ids, and reports duplicate skips without
  modifying prior rows.
- `tools/append_repair_campaign_stackability_posterior.py` provides the queue
  executable.
- `repair_campaign_stackability_queue` now adds a fourth ready-row step after
  learning-signal generation: append to the stackability posterior and emit a
  false-authority append report.
- Public `tac.optimization` exports now expose the posterior append helper.

## Safeguards

- The posterior input must be a schema-valid repair learning signal.
- The learning signal and posterior row are both checked for truthy authority
  fields.
- Queue postconditions validate the append report, false authority, dispatch
  false, and JSONL posterior false-authority rows.
- Duplicate replay identity does not append another row.

## Verification

- `.venv/bin/python -m ruff check src/tac/optimization/repair_campaign_posterior.py tools/append_repair_campaign_stackability_posterior.py src/comma_lab/scheduler/repair_campaign_stackability_queue.py src/tac/optimization/__init__.py src/tac/tests/test_repair_campaign_stackability_queue.py`
- `.venv/bin/python -m py_compile src/tac/optimization/repair_campaign_posterior.py tools/append_repair_campaign_stackability_posterior.py src/comma_lab/scheduler/repair_campaign_stackability_queue.py`
- `.venv/bin/python -m pytest src/tac/tests/test_repair_campaign_stackability_queue.py -q`
  - `6 passed`

## Remaining Work

The posterior is now durable and queryable, but acquisition rules still need to
consume it automatically. The next implementation step is a repair-family
priority reader that summarizes posterior rows by family, entropy position,
targeted dimensions, and improvement-per-byte, then feeds those priors into the
default repair scorer and stackability queue selection.
