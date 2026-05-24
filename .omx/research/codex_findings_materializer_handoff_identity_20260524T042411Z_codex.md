# Codex Findings - Materializer Handoff Identity

- generated_at_utc: 2026-05-24T04:24:11Z
- agent: codex
- scope: operator briefing materializer exact-ready handoff summary
- authority: no score claim; no promotion authority; no dispatch authority

## Finding

Banach sidecar review found that `tools/operator_briefing.py` could collapse
materially different materializer exact-ready consumer reports when rows shared
the same `candidate_id` and `archive_sha256` but differed by runtime content,
runtime tree, or score axis. That made the briefing summary vulnerable to a
false-green undercount.

## Fix

The handoff summary now extracts `stable_identity`,
`runtime_content_tree_sha256`, `runtime_tree_sha256`, and `score_axis` from
consumer rows and includes them in the deduplication identity. Stable identities
take precedence when present. A regression test now keeps two same-candidate,
same-archive reports distinct when their scorer/runtime stable identities
differ.

## Verification

- `.venv/bin/python -m pytest -q src/tac/tests/test_operator_briefing.py -k 'materializer_exact_ready_handoff_summary'`
- `.venv/bin/python -m py_compile tools/operator_briefing.py src/tac/tests/test_operator_briefing.py`
- `.venv/bin/python -m ruff check tools/operator_briefing.py src/tac/tests/test_operator_briefing.py`
- `.venv/bin/python tools/review_gate_hook.py`
- `.venv/bin/python tools/operator_briefing.py --json --skip-pareto --skip-dashboard --skip-reconciler --skip-provider-readiness`

