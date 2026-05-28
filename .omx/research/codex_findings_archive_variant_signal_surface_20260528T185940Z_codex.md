# Codex findings: archive variant signal surface

Date: 2026-05-28T18:59:40Z
Lane: `repair_archive_variant_signal_surface`
Axis: `[macOS-MLX research-signal]`; exact score authority remains contest CPU/CUDA only.

## Finding

The archive repair executor was producing real signal that was still too easy to
orphan: non-selected archive transform variants, entropy probes, and range/ANS
runtime prototypes lived inside `candidate_archive_transform_variants`, while
the stack optimizer primarily consumed the selected candidate plus coarse
entropy-substrate coverage.

This was especially risky for range/ANS work. The probes estimate entropy
pressure and the prototypes prove deterministic member decode, but neither is
a score claim or contest-ready runtime adapter. They need to update acquisition
pressure and backlog routing without becoming authority.

## Landing

- Added `repair_archive_variant_signal_surface.v1` to every byte-transform
  execution report.
- Every archive transform variant now emits a typed signal row with transform
  kind, selected/non-selected status, materialized/prototype/probe class,
  runtime-proof readiness, blocker list, deterministic hash of rows, and
  explicit false-authority fields.
- Stack search now consumes the surface as first-class acquisition features:
  variant counts, probe counts, prototype counts, runtime-proof counts,
  blocked-signal counts, signal kinds, blockers, and an acquisition penalty for
  unresolved probe/runtime-adapter pressure.
- The bounded autonomous floor-loop summary now exposes the same aggregate
  fields so operator-visible rollups no longer hide nested probes.

## Authority boundary

Range/ANS prototype rows can be archive-bound and runtime-proven locally, but
they remain blocked from exact dispatch until a contest runtime adapter and
contest CPU/CUDA adjudication exist. MLX-local rows and entropy estimates are
advisory only.

## Review

Pass 1, orphan-signal audit: clean. Non-selected archive variants are no
longer nested-only side reports.

Pass 2, authority audit: clean. The signal rows and surfaces pass the shared
false-authority guard and do not grant score, promotion, rank/kill, budget, or
dispatch authority.

Pass 3, loop-integration audit: clean. Execution report -> stack row ->
interaction tensor -> learning signal -> bounded floor-loop summary now all
carry variant signal fields.

## Verification

- `.venv/bin/ruff check --fix src/tac/optimization/repair_archive_entropy_substrate_coverage.py src/tac/optimization/repair_family_byte_transform_executor.py src/tac/optimization/repair_family_stack_search.py tools/run_repair_campaign_autonomous_floor_loop.py src/tac/tests/test_repair_family_materializers.py src/tac/tests/test_repair_campaign_materialization_queue.py`
- `.venv/bin/python -m py_compile src/tac/optimization/repair_archive_entropy_substrate_coverage.py src/tac/optimization/repair_family_byte_transform_executor.py src/tac/optimization/repair_family_stack_search.py tools/run_repair_campaign_autonomous_floor_loop.py src/tac/tests/test_repair_family_materializers.py src/tac/tests/test_repair_campaign_materialization_queue.py`
- `.venv/bin/pytest src/tac/tests/test_repair_family_materializers.py -q`
- `.venv/bin/pytest src/tac/tests/test_repair_campaign_materialization_queue.py -q`
- `tools/review_tracker.py mark-file` for three reviewers on all touched Python files, then scoped `tools/review_tracker.py policy-check`: 0 violations.
