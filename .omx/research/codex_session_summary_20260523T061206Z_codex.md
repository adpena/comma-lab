# Codex Session Summary

Date: 2026-05-23T06:12:06Z
Agent: Codex
Lane: `lane_codex_mlx_cache_scheduler_guard_hardening_20260523`

## Landed

- Hardened MLX scorer-input cache audit stamps with shared dereference,
  SHA-256, false-authority, verdict, and current-cache-identity validation.
- Wired the shared validator into MLX scorer-response generation and windowed
  scorer-response dataset ingestion so copied stale manifest stamps fail closed.
- Hardened MLX cache materialization forced replacement with owned-directory
  markers and preserved unowned cache directories on failed audits.
- Extended Catalog #207 preflight coverage to `tools/materialize_*.py` and
  tightened one ATW2 materializer output namespace.
- Extended artifact retention planning to block known nested auth-eval raw
  workdirs instead of missing deep `.raw` payload inflation surfaces.

## Evidence

- `.venv/bin/python -m pytest -q src/tac/tests/test_mlx_scorer_response.py src/tac/tests/test_mlx_cache_audit.py src/tac/tests/test_scorer_response_dataset.py src/tac/tests/test_materialize_mlx_scorer_cache_from_auth_eval.py src/tac/tests/test_artifact_retention.py src/tac/tests/test_check_207_no_unguarded_rmtree_in_build_tools.py`
  returned `174 passed in 6.51s`.
- `.venv/bin/python -m ruff check ...` over the touched MLX, dataset,
  materializer, retention, preflight, and test files returned `All checks
  passed!`.
- `git diff --check` passed.
- `.gitignore` was checked and did not need changes; no new rebuildable output
  namespace was introduced.

## Frontier State

No contest-authoritative score changed in this patch. The current remembered
frontier remains `[contest-CPU Linux x86_64] 0.1920282830` from the DQS1
pairset-drop/selective-decoderq lane, with `[contest-CUDA T4] 0.2053300290` as
the latest remembered T4 anchor.

## Next Tranche

- Land the resource-bounded experiment queue worker: claim/start/finalize,
  parent-owned SQLite state mutation, subprocess polling, timeouts, observer
  telemetry, and configurable CPU/MLX/cloud concurrency.
- Generate DQS1 and cross-family queue builders that emit multi-candidate,
  combinatorial local-first sweeps instead of one active row at a time.
- Use the hardened MLX/cache stamps as the admission gate for local substrate
  training and score-triage rows; exact auth dispatch stays required for
  promotion.
- Extend the same canonical stamp and custody patterns to MLX calibration,
  quality-speed delta, PR95/HNeRV variants, NeRV-family substrates, and
  non-NeRV candidate stacks.

## Authority

`score_claim=false`; `promotion_eligible=false`;
`ready_for_exact_eval_dispatch=false`; `rank_or_kill_eligible=false`.
