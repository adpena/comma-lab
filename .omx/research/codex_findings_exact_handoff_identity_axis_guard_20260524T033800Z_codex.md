# Codex Findings - Exact Handoff Identity Axis Guard

UTC: 2026-05-24T03:38:00Z
Author: Codex
Lane: `codex_exact_handoff_identity_axis_guard_20260524`

## Finding

The live materializer exact-eval consumer run found two exact-ready queues for
`ias1_runtime_parity_top4_20260523T1930Z`. Both queues were shaped correctly,
but neither carried an explicit `score_axis` or `target_score_axis`. The prior
handoff code defaulted missing axis to `contest_cuda`, which is a false-authority
risk: an exact-ready planning row must declare its target axis instead of
letting downstream consumers infer it.

The run also showed that archive + runtime-content identity alone is not enough
for exact-eval handoff custody. `runtime_tree_sha256` is the submitted runtime
packet boundary, so duplicate/evaluated keys now include it explicitly. This
supersedes the narrower identity note in
`codex_findings_exact_dispatch_plan_identity_dedupe_20260524T034000Z_codex.md`.

## Landing

`materializer_exact_eval_consumer`, `materializer_exact_eval_dispatch_plan`,
`exact_ready_audit`, and `packetir_exact_closure` now require and preserve a
stable identity containing:

- candidate archive SHA-256;
- runtime content tree SHA-256;
- runtime tree SHA-256;
- explicit score axis.

Missing score axis blocks before authority checks with
`stable_identity_score_axis_missing`; unsupported exact-dispatch axes block with
`stable_identity_score_axis_unsupported:<axis>`. No handoff code now invents
`contest_cuda`. The exact-readiness producer stamps promoted rows with
`score_axis=contest_cuda` and `target_score_axis=contest_cuda` so newly promoted
queues satisfy the stricter contract. Exact-CUDA result-review packets with
missing score axis now fail closed in the stale-ready audit instead of becoming
invisible duplicate evidence.

`tools/operator_briefing.py` now surfaces materializer exact-ready handoff
state in normal operator flows: recent bridge reports, consumer reports,
paused queue outputs, and the next safe consumer command. The briefing summary
keeps score, promotion, rank/kill, and dispatch authority false. It also keeps
superseded consumer reports input-aware rather than queue-id-only, treats hard
plan blockers as `BLOCKED` even when candidate counts are nonzero, and exposes
the top row-level blockers instead of hiding them behind aggregate counts.

## Live Artifact

The current live consumer audit is:

- `.omx/research/materializer_exact_eval_consumer_live_20260524T033624Z.json`
- `.omx/research/materializer_exact_eval_consumer_live_20260524T033624Z.experiment_queue.json`

It consumed the two inverse-scorer exact-ready queues and authorized zero rows.
Both rows fail closed on `stable_identity_score_axis_missing`; no dispatch was
attempted.

## Verification

- `.venv/bin/python -m pytest -q src/tac/tests/test_operator_briefing.py`
- `.venv/bin/python -m pytest -q src/tac/tests/test_materializer_exact_eval_consumer.py src/tac/tests/test_materializer_chain_harvest_scheduler.py src/tac/tests/test_optimizer_exact_readiness.py src/tac/tests/test_optimizer_exact_ready_audit.py src/tac/tests/test_packetir_exact_closure.py`
- `.venv/bin/python -m ruff check src/comma_lab/scheduler/materializer_exact_eval_consumer.py src/comma_lab/scheduler/materializer_exact_eval_dispatch_plan.py src/tac/optimizer/exact_readiness.py src/tac/optimizer/exact_ready_audit.py src/tac/packetir_exact_closure.py src/tac/tests/test_materializer_exact_eval_consumer.py src/tac/tests/test_materializer_chain_harvest_scheduler.py src/tac/tests/test_operator_briefing.py src/tac/tests/test_optimizer_exact_readiness.py src/tac/tests/test_optimizer_exact_ready_audit.py src/tac/tests/test_packetir_exact_closure.py tools/operator_briefing.py`

Result: operator briefing suite passed with `27 passed`; exact-ready/materializer
suite passed with `143 passed`; ruff passed.

## Remaining Work

The blocked inverse-scorer queue needs regeneration through the stricter
exact-readiness producer or an explicit, reviewed axis backfill. Even with axis
present, the current archive is blocked by stale exact-CUDA review and runtime
consumption proof gaps, so it is not a paid-dispatch candidate.
