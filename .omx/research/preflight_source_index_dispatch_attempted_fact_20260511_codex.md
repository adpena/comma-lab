# Preflight SourceIndex dispatch_attempted fact hardening (2026-05-11)

## Scope

`tools/audit_tooling_consolidation.py` queries both `score_claim` and
`dispatch_attempted` to inventory repeated audit/dispatch metadata patterns.
`score_claim` was already part of `tac.source_index` one-pass text facts, but
`dispatch_attempted` was not. That forced `files_containing_substrings(...)` to
fall back to full cached text scans for an otherwise indexable audit token.

## Change

- Added `dispatch_attempted` to the SourceIndex default fact vocabulary and
  bumped the persistent fact schema to `pact.source_text_facts.v21`.
- Added a regression test proving `score_claim` + `dispatch_attempted` queries
  are served from the substring index without unknown-substring text fallback.
- Kept the audit result complete: pattern counts still report total findings,
  while metadata occurrences remain capped at the existing documented limit.
- Added bounded deterministic per-file scan parallelism for standalone
  dispatch-hazard and tooling-consolidation scans. All-lanes still uses the
  shared SourceIndex path for broad scans.

## Verification

Focused tests:

```bash
.venv/bin/python -m pytest -q \
  src/tac/tests/test_source_index.py \
  src/tac/tests/test_dispatch_cli_shell_hazards.py \
  src/tac/tests/test_audit_tooling_consolidation.py \
  src/tac/tests/test_all_lanes_preflight_timing_profile.py
```

Result: `51 passed in 1.37s`.

Full all-lanes preflight after cache rebuild:

```bash
.venv/bin/python tools/all_lanes_preflight.py \
  --timings \
  --timings-json /tmp/all_lanes_preflight_parallel_scanners_v4_steady.json \
  --timeout-s 30
```

Result: `ALL 29 PREFLIGHT CHECKS PASSED`; wall `2.55s`; Gate #0 `1.78s`;
Gate #8 `2.46s`; hard 30s budget still active.

## Classification

- score_claim: `false`
- dispatch_attempted: `false`
- remote_or_gpu_eval_started: `false`
- purpose: local DX/preflight correctness and speed hardening
