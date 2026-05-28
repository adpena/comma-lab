# Codex Findings: Archive Entropy Probe Anti-Pattern Protections

Timestamp: 2026-05-28T18:37:23Z

## Search Inputs

Searched `.omx/research`, `CLAUDE.md`, `AGENTS.md`,
`src/tac/canonical_anti_patterns`, `src/tac/optimization`, and `tools` for
anti-pattern, cargo-cult, false-authority, orphan-signal, custody, scaffold,
side-report, and promotion-leakage patterns.

## Protected Against

- Proxy/advisory probe masquerading as score or exact-dispatch authority.
- Probe-only entropy side reports becoming orphaned from optimizer consumers.
- Scaffold/probe bytes being treated as operational without receiver/runtime
  consumption.
- Zero-order entropy estimates being promoted as materialized savings.
- Entropy-coder ordering cargo-cult: after-coder repack, coder-boundary recode,
  and before-coder distribution shaping are kept distinct.

## What Changed

- Archive entropy coverage now emits `anti_pattern_protections` rows with
  `denied_uses` and explicit false-authority fields.
- Range/ANS zero-order savings are surfaced as planning pressure only; real
  `saved_bytes` remains blocked until a materializer and runtime adapter exist.
- Stack rows and learning signals carry anti-pattern IDs, probed-substrate
  counts, and estimated entropy pressure.
- Autonomous floor-loop summaries expose the same anti-pattern and entropy
  pressure fields so the runner does not need nested artifact spelunking.

## Verification

- `.venv/bin/ruff check --fix src/tac/optimization/repair_archive_entropy_substrate_coverage.py src/tac/optimization/repair_family_stack_search.py tools/run_repair_campaign_autonomous_floor_loop.py src/tac/tests/test_repair_family_materializers.py src/tac/tests/test_repair_campaign_materialization_queue.py`
- `.venv/bin/python -m py_compile src/tac/optimization/repair_archive_entropy_substrate_coverage.py src/tac/optimization/repair_family_stack_search.py tools/run_repair_campaign_autonomous_floor_loop.py`
- `.venv/bin/pytest src/tac/tests/test_repair_family_materializers.py -q`
- `.venv/bin/pytest src/tac/tests/test_repair_campaign_materialization_queue.py -q`
