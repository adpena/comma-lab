# Codex Findings - Planning-Only Materializer Exact-Followup Guard

UTC: 2026-05-24T01:24:21Z
Author: Codex
Lane: `codex_planning_materializer_exact_followup_20260524T012421Z`

## Finding

`tools/run_byte_shaving_materializer_campaign.py` now always asks the
materializer execution queue for exact-readiness follow-ups. That is correct
for archive-producing materializer chains, but it exposed a type bug in
`build_materializer_execution_queue`: planning-only inverse-scorer action
functional rows were forced through candidate-chain handoff logic even though
their contract explicitly emits no archive and carries no score authority.

This blocked the safest first CPU/SSH proof row for inverse-steganalysis
planning and risked future agents either disabling follow-ups globally or
working around the row type by hand.

## Landing

`build_materializer_execution_queue` is now row-aware for exact-readiness
follow-ups. Candidate-chain materializers still require a
`materializer_chain_complete` postcondition and fail closed if it is missing.
Planning-only inverse action functional rows remain executable, but the queue
metadata records:

- `exact_readiness_followup_requested=true`
- `exact_readiness_followup_enabled=false`
- `exact_readiness_followup_skipped_reason=planning_only_inverse_action_functional_not_candidate_archive`

The row remains false-authority and cannot claim a score, promote, rank/kill,
or dispatch exact eval.

## Verification

- `PYTHONPATH=. .venv/bin/python -m pytest -q src/tac/tests/test_byte_shaving_campaign_queue.py::test_materializer_execution_queue_skips_exact_followup_for_planning_only_rows src/tac/tests/test_byte_shaving_campaign_queue.py::test_materializer_execution_queue_followup_requires_chain_postcondition_for_candidates src/tac/tests/test_byte_shaving_campaign_queue.py::test_materializer_execution_queue_can_append_exact_readiness_followups`
- `PYTHONPATH=. .venv/bin/python -m ruff check src/comma_lab/scheduler/byte_shaving_campaign_queue.py src/tac/tests/test_byte_shaving_campaign_queue.py`
- `PYTHONPATH=. .venv/bin/python -m py_compile src/comma_lab/scheduler/byte_shaving_campaign_queue.py src/tac/tests/test_byte_shaving_campaign_queue.py`

Result: all passed.

I also ran a local bounded materializer smoke:

- run dir: `experiments/results/inverse_action_materializer_smoke_20260524T012518Z/campaign`
- queue id: `inverse_action_materializer_smoke_20260524T012518Z`
- worker: `success_count=1`, `failure_count=0`
- output: `experiments/results/inverse_action_materializer_smoke_20260524T012518Z/action_functional.json`
- output schema: `inverse_steganalysis_discrete_action_functional.v1`
- authority: `score_claim=false`, `promotion_eligible=false`, `rank_or_kill_eligible=false`, `ready_for_exact_eval_dispatch=false`
- selected planning cells: `1`

## Remaining Work

1. Commit and push this guard so SSH peers can satisfy same-HEAD preflight.
2. Create or refresh the `tertiary` checkout at `/Users/adpena/Projects/pact`.
3. Run the same bounded planning-only row through
   `--staircase-ssh-execute` with artifact pullback to prove queue-owned
   distributed execution without conflating it with score authority.
4. Then run the archive-producing byte-range chain smoke recommended by the
   sidecar explorer once remote dependencies and source PR103 artifacts are
   confirmed present.
