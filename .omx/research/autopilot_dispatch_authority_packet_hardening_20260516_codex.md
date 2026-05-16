# Autopilot Dispatch Authority Packet Hardening - 2026-05-16

## Scope

Patch class: production hardening / adversarial review follow-up.

Primary finding: the cathedral autopilot le-$5 authorization path could infer
dispatch authority from rank, cost, and absence of blockers. Planning-only
rows from composition/probe sources are useful for ranking, but they are not
executable dispatch packets.

## Change

- Added explicit dispatch authority fields to `CandidateRow` and `HaltEvent`:
  lane id, claim keys, target modes, dispatch packet hash, archive/runtime
  hashes, and exact-eval readiness.
- `OperatorAuthorizedModeConfig.can_authorize()` now refuses rows that are not
  explicit dispatch-authority packets.
- Dispatch conflict checks now scan candidate id plus lane/claim keys.
- Planning-only composition and probe rows receive
  `planning_only_source_requires_operator_dispatch_packet` even if upstream
  forgot to include a blocker.

## Evidence

- `.venv/bin/python -m pytest -q src/tac/tests/test_cathedral_autopilot_autonomous_loop.py src/tac/tests/test_cathedral_autopilot_substrate_composition_wire.py`
  - `98 passed in 0.29s`
- `.venv/bin/ruff check tools/cathedral_autopilot_autonomous_loop.py src/tac/tests/test_cathedral_autopilot_autonomous_loop.py src/tac/tests/test_cathedral_autopilot_substrate_composition_wire.py`
  - `All checks passed`
- `.venv/bin/python -m py_compile tools/cathedral_autopilot_autonomous_loop.py`

## Evidence Boundary

No score claim. No dispatch. No promotion. This is a guardrail that prevents
planning rows from becoming autonomous spend authority.
