# Autopilot Exact-Eval Authorization Custody Hardening

Date: 2026-05-16
Operator directive: L5/Cathedral hardening, contest compliance, source-of-truth
main, and no false dispatch authority.

## Finding

Adversarial audit found that the operator-authorized le-$5 path could authorize
a contest exact-eval target when a row had a dispatch-packet hash but no archive
SHA and runtime-tree SHA. A dispatch packet can be a useful launch manifest, but
for exact-eval authority it does not replace scored archive/runtime custody.

Impact: a self-authorized exact-eval dispatch could enter the journal with blank
archive/runtime hashes, weakening reproducibility and apples-to-apples evidence
tracking.

## Patch

- `CandidateRow.dispatch_authority_blockers()` now requires
  `archive_sha256` and `runtime_tree_sha256` whenever
  `target_modes` includes `contest_exact_eval`.
- Dispatch-packet hashes remain validated, but they no longer substitute for
  exact archive/runtime custody on contest exact-eval targets.
- Test helper defaults now produce fully custodied contest candidates.
- Added a regression proving dispatch-packet-only exact-eval candidates are
  refused.
- Updated CLI authorized-mode test fixture to include archive/runtime custody.

## Verification

```
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q \
  src/tac/tests/test_cathedral_autopilot_autonomous_loop.py
# 113 passed

.venv/bin/python -m ruff check \
  tools/cathedral_autopilot_autonomous_loop.py \
  src/tac/tests/test_cathedral_autopilot_autonomous_loop.py
# All checks passed

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile \
  tools/cathedral_autopilot_autonomous_loop.py \
  src/tac/tests/test_cathedral_autopilot_autonomous_loop.py
```

## Follow-Up

Next campaign-hardening pass should wire the same archive/runtime custody
requirement into any non-loop exact-eval actuator that consumes Cathedral
candidate rows directly.
