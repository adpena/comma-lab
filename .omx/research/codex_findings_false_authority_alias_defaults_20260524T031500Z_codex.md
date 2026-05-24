# Codex Findings - False Authority Alias Defaults

UTC: 2026-05-24T03:15:00Z
Author: Codex
Lane: `codex_false_authority_alias_defaults_20260524`

## Finding

`json_false_authority` and `materializer_chain_complete` guarded the canonical
score authority fields, but downstream scheduler payloads could still carry
truthy score- or dispatch-looking aliases such as `score_claim_valid`,
`promotable`, `exact_cuda_auth_eval`, `dispatch_ready`, `exact_eval_ready`, or
`charged_bits_changed`. Those aliases are enough to mislead automation even
when the canonical fields are false.

## Landing

The queue postcondition layer now uses shared default false-authority field
sets. Canonical fields remain required-false; authority-looking aliases are
false-or-missing by default. Explicit `required_false` or `false_or_missing`
condition overrides stay exact so legacy or intentionally scoped checks do not
silently expand.

## Verification

- `PYTHONPATH=. .venv/bin/python -m pytest -q src/tac/tests/test_experiment_queue.py`
- `PYTHONPATH=. .venv/bin/python -m ruff check src/comma_lab/scheduler/experiment_queue.py src/tac/tests/test_experiment_queue.py`
- `PYTHONPATH=. .venv/bin/python -m pytest -q src/tac/tests/test_ssh_experiment_queue_executor.py src/tac/tests/test_byte_shaving_campaign_queue.py src/tac/tests/test_byte_shaving_materializer_campaign_runner.py src/tac/tests/test_experiment_queue.py`

Result: focused experiment-queue tests passed, and the touched files passed
ruff. The combined scheduler/queue guard suite passed with `142 passed`.

## Remaining Work

Exact-score authority still belongs only to contest CPU/CUDA auth artifacts.
Future alias discoveries should extend the default field set and add a
regression test rather than relying on operator memory or per-lane prose.
