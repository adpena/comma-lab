# Cathedral Zero-Cost Planning-Row Hardening

Date: 2026-05-16
Owner: codex
Scope: `tools/cathedral_autopilot_autonomous_loop.py`

## Finding

Some read-only planning sources, including probe-disambiguator and composition
ranking rows, legitimately carry `estimated_dispatch_cost_usd=0.0` when the
row is blocked or the timing-smoke cost has not been measured yet. The loaders
already accepted nonnegative cost, but the ranker and loop preflight required
strictly positive cost for every row before emitting halt events.

The result was bad DX and signal loss: a blocked cost-unknown planning row
could crash the loop before operator review.

## Fix

- Split planning cost validation from dispatch authorization cost validation.
- Planning/ranking accepts finite nonnegative cost.
- Zero-cost planning rows return `eig_per_dollar=0.0`, sorting behind priced
  rows rather than raising.
- Operator-authorized self-dispatch still refuses non-positive cost inside the
  authorization path.
- Added regressions for zero-cost ranking and loop halt-event surfacing.

## Evidence

- `.venv/bin/python -m pytest src/tac/tests/test_cathedral_autopilot_autonomous_loop.py -q`
- `.venv/bin/python -m ruff check tools/cathedral_autopilot_autonomous_loop.py src/tac/tests/test_cathedral_autopilot_autonomous_loop.py`
- `.venv/bin/python -m py_compile tools/cathedral_autopilot_autonomous_loop.py src/tac/tests/test_cathedral_autopilot_autonomous_loop.py`
- `git diff --check -- tools/cathedral_autopilot_autonomous_loop.py src/tac/tests/test_cathedral_autopilot_autonomous_loop.py`

## Result

Cost-unknown planning rows remain visible as blocked work; they cannot
self-authorize until a positive cost estimate exists.
