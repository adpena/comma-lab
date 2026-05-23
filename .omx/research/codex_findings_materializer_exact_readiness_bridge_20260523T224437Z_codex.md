# Codex Findings: Materializer Exact-Readiness Bridge

- timestamp_utc: 2026-05-23T22:44:37Z
- agent: codex
- lane_id: codex_materializer_exact_readiness_bridge_20260523
- research_only: false

## Finding

The materializer chain harvester now emits validated optimizer source queues,
but exact-readiness promotion still required a manual second step. That left
the DAG handoff partially automated: completed local chains were discoverable,
yet the readiness blockers were not automatically expanded into per-candidate
exact-readiness reports.

## Fix Landed

Added an explicit optional bridge in
`comma_lab.scheduler.materializer_chain_harvest` and exposed it through
`tools/harvest_materializer_chain_candidates.py --exact-readiness-out-dir`.

The bridge:

- consumes the harvested planning source queue written by the harvest CLI;
- invokes the existing `tac.optimizer.exact_readiness` promoter per candidate;
- clears only materializer-harvest source blockers after the promoter validates
  archive/runtime custody;
- writes per-candidate exact-readiness reports;
- writes a per-candidate exact-ready queue only when the existing promoter
  returns `ready_for_exact_eval_dispatch=true`;
- keeps the aggregate bridge report false-authority / planning-only and does
  not dispatch, claim score, promote, rank, or kill.

The materializer chain adapter now preserves
`runtime_consumption_proof_required`, `runtime_consumption_proof_status`, and
`runtime_consumption_proof_path` from chain manifests into optimizer source
rows. That is required because score-affecting materialized bytes must still
prove runtime consumption before exact-ready promotion.

## Live Bridge Smoke

Command:

```bash
.venv/bin/python tools/harvest_materializer_chain_candidates.py \
  --work-queue experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/inverse_action_functional_autosaturated_queue_20260523T220300Z/materializer_work_queue.json \
  --source-queue-out /tmp/pact_scheduler_materializer_harvest_source_queue.json \
  --report-out /tmp/pact_scheduler_materializer_harvest_report.json \
  --exact-readiness-out-dir /tmp/pact_scheduler_materializer_exact_readiness \
  --exact-readiness-bridge-report-out /tmp/pact_scheduler_materializer_exact_readiness_bridge_report.json \
  --allow-unfinished-state \
  --require-accepted
```

Result:

- harvested `1/1` materializer chain manifest;
- exact-readiness bridge checked `1` candidate;
- ready candidates: `0/1`;
- fail-closed blockers include missing full-frame inverse-scorer inflate parity,
  missing `inflate.sh`, missing `report.txt`, missing archive manifest, missing
  runtime-tree hashes, and missing runtime-consumption proof.

## Verification

- `PYTHONPATH=. .venv/bin/python -m pytest src/tac/tests/test_materializer_chain_harvest_scheduler.py -q`
  - `7 passed`
- `PYTHONPATH=. .venv/bin/python -m pytest src/tac/tests/test_materializer_chain_harvest_scheduler.py src/tac/tests/test_optimizer_candidate_queue.py src/tac/tests/test_optimizer_exact_readiness.py -q`
  - `85 passed`
- `.venv/bin/python -m ruff check src/comma_lab/scheduler/materializer_chain_harvest.py tools/harvest_materializer_chain_candidates.py src/tac/optimizer/materializer_chain_harvest.py src/tac/tests/test_materializer_chain_harvest_scheduler.py`
  - passed
- `git diff --check`
  - passed

## Next Gate

The live inverse-scorer chain needs a runtime packet closure artifact before it
can become exact-ready: full-frame inflate parity, submission-dir `inflate.sh`,
archive manifest, `report.txt`, runtime-tree custody, and runtime-consumption
proof. The new bridge now exposes those blockers as a deterministic report
instead of leaving them as manual operator inference.
