# Codex Findings: Modal Runtime Content Fail-Closed Custody

UTC: 2026-05-23T20:52:11Z
Agent: Codex
Lane: codex_modal_runtime_content_fail_closed_20260523

## Summary

Exact-eval dispatch and recovery needed a stricter runtime-content custody
boundary. Runtime-tree SHA alone can conflate path/package shape with actual
runtime content, and recovery could still look successful when the canonical
`contest_auth_eval.json` artifact was missing.

## Change

- Modal CPU/CUDA auth-eval upload paths now require
  `--expected-runtime-tree-sha256` whenever `--submission-dir` is uploaded.
  Dispatch exits before claims if the expected tree hash is missing.
- Modal non-authoritative payloads now force false-authority fields to `false`
  instead of preserving truthy caller-provided values.
- Modal recovery converts missing or invalid canonical auth-eval artifacts into
  failed recovery statuses with `returncode=97`.
- `tools/recover_modal_auth_eval.py` maps those failed recovery statuses to
  terminal failed claim rows.
- Exact readiness, exact dispatch authority, and parallel dispatch now require
  `runtime_content_tree_sha256` alongside `runtime_tree_sha256`, and stale
  terminal-claim matching checks runtime-content mismatches too.
- Inverse-scorer full-frame parity rows now fail exact readiness when they grow
  archive bytes unless the row is explicitly tagged as a rate-only control.

## Verification

- `.venv/bin/python -m pytest src/tac/tests/test_modal_auth_eval.py src/tac/tests/test_modal_auth_eval_recovery.py src/tac/tests/test_recover_modal_auth_eval_tool.py src/tac/tests/test_optimizer_exact_readiness.py src/tac/tests/test_exact_dispatch_authority.py tests/test_parallel_dispatch_top_k_exact_ready_audit.py -q`
- `.venv/bin/python -m ruff check experiments/modal_auth_eval.py experiments/modal_auth_eval_cpu.py src/tac/deploy/modal/auth_eval.py src/tac/optimizer/exact_dispatch_authority.py src/tac/optimizer/exact_readiness.py tools/parallel_dispatch_top_k.py tools/recover_modal_auth_eval.py src/tac/tests/test_modal_auth_eval.py src/tac/tests/test_modal_auth_eval_recovery.py src/tac/tests/test_recover_modal_auth_eval_tool.py src/tac/tests/test_optimizer_exact_readiness.py tests/test_parallel_dispatch_top_k_exact_ready_audit.py`
