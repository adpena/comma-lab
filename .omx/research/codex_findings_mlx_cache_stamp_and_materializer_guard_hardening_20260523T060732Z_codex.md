# Codex Findings: MLX Cache Stamp And Materializer Guard Hardening

Date: 2026-05-23T06:07:32Z
Agent: Codex
Lane: `lane_codex_mlx_cache_scheduler_guard_hardening_20260523`

## Summary

This pass closed the next MLX custody bug class after the unit/scope fix:
cache-audit stamps could be copied by value unless every consumer dereferenced
the audit JSON, verified the file hash, and compared the referenced audit
payload back to the current cache identity.

Landed fixes:

- Added shared `cache_audit_stamp_blockers(...)` validation in
  `tac.local_acceleration.mlx_cache_audit`.
- MLX scorer-response candidate cache acceptance now verifies the referenced
  auth/local-advisory audit path, SHA-256, false-authority fields, verdict, and
  current cache identity before allowing transfer or local-advisory debug use.
- Windowed MLX scorer-response dataset ingestion now uses the same shared
  dereference validator before admitting auth-audited MLX windows.
- `tools/materialize_mlx_scorer_cache_from_auth_eval.py --force` now refuses to
  delete pre-existing work/cache directories unless they carry this tool's
  owned-directory marker.
- Failed-audit cleanup preserves unowned pre-existing output caches instead of
  deleting them opportunistically.
- Catalog #207 rmtree preflight now scans `tools/materialize_*.py`, not only
  `build_*.py` and `promote_*.py`.
- The ATW2 materializer now validates its `--output-dir` under
  `experiments/results` before any forced replacement.
- Retention planning now reports known nested raw workdirs such as
  `contest_auth_eval_cpu_workdir` as blocked unknown raw surfaces instead of
  only noticing directories that directly contain `.raw` files.

## Evidence

- Focused verification:
  `.venv/bin/python -m pytest -q src/tac/tests/test_mlx_scorer_response.py src/tac/tests/test_mlx_cache_audit.py src/tac/tests/test_scorer_response_dataset.py src/tac/tests/test_materialize_mlx_scorer_cache_from_auth_eval.py src/tac/tests/test_artifact_retention.py src/tac/tests/test_check_207_no_unguarded_rmtree_in_build_tools.py`
  returned `174 passed in 8.11s`.
- Ruff over the touched MLX/cache/materializer/retention/preflight bundle passed.

## Remaining Work

- Implement Galileo's resource-bounded parallel queue worker as a separate
  patch: split claim/start/finalize, parent-owned SQLite mutation, process
  polling/timeout, observer PID/resource telemetry, and DQS1 multi-candidate
  queue generation.
- Extend the same stamp validator into any future MLX calibration consumer that
  accepts cache identity by manifest copy.
- Continue the tracked experiment-binary cleanup as a separate repository
  hygiene pass; this patch only prevented new raw/custody loss classes.

## Authority

`score_claim=false`; `promotion_eligible=false`;
`ready_for_exact_eval_dispatch=false`; `rank_or_kill_eligible=false`.
