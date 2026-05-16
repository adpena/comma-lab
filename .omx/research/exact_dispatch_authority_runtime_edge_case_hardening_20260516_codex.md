# Exact Dispatch Authority Runtime Edge-Case Hardening

- date: `2026-05-16`
- agent: `codex`
- scope: exact-readiness terminal evidence, detached exact-ready queue custody
- score_claim: `false`
- promotion_eligible: `false`
- dispatch_attempted: `false`

## Why

The Cathedral autopilot authority wire-in exposed two adjacent exact-dispatch
custody edge cases while running the broader exact-readiness suite:

1. `tools/parallel_dispatch_top_k.py` recomputed runtime custody for detached
   exact-ready bundles against this checkout's `upstream/evaluate.py` even when
   the ranked input directory carried its own `upstream/evaluate.py` and live
   submission paths. That created false `runtime_tree_sha256_mismatch` blockers
   for byte-closed detached queues.
2. `terminal_claim_result_conflicts(...)` did not distinguish a terminal row
   whose own notes identify a different runtime from an older active row whose
   notes mention a different runtime. The first case is runtime-disambiguated
   terminal evidence and should not stale-block the candidate by default. The
   second case remains fail-closed unless `score_affecting_runtime_changed=true`
   because the terminal row itself did not bind the runtime.

## Landed

- `tools/parallel_dispatch_top_k.py`
  - added `_authority_repo_root_for_candidate(...)`;
  - keeps production queues checked against `REPO`;
  - switches to the ranked-input directory only when it looks like a detached
    custody root with `upstream/evaluate.py` and live candidate paths under it.
- `src/tac/optimizer/exact_readiness.py`
  - prefers runtime SHA evidence recorded on the terminal claim row itself;
  - allows a terminal-row runtime mismatch to disambiguate the terminal result
    unless `block_runtime_mismatch_for_same_archive=true`;
  - preserves the stricter blocker when only older grouped claim notes carry a
    mismatching runtime and the candidate lacks
    `score_affecting_runtime_changed=true`.

## Verification

- `.venv/bin/python -m ruff check tools/cathedral_autopilot_autonomous_loop.py src/tac/tests/test_cathedral_autopilot_autonomous_loop.py src/tac/optimizer/exact_dispatch_authority.py src/tac/optimizer/exact_readiness.py tools/parallel_dispatch_top_k.py src/tac/tests/test_optimizer_exact_readiness.py src/tac/tests/test_optimizer_exact_ready_audit.py` -> clean
- `.venv/bin/python -m py_compile tools/cathedral_autopilot_autonomous_loop.py src/tac/optimizer/exact_dispatch_authority.py src/tac/optimizer/exact_readiness.py tools/parallel_dispatch_top_k.py` -> clean
- `.venv/bin/python -m pytest src/tac/tests/test_optimizer_exact_readiness.py src/tac/tests/test_optimizer_exact_ready_audit.py -q` -> `55 passed`
- `.venv/bin/python -m pytest src/tac/tests/test_cathedral_autopilot_autonomous_loop.py -q` -> `148 passed`
- `.venv/bin/python -m pytest src/tac/tests/test_operator_briefing.py src/tac/tests/test_all_lanes_operator_briefing_gate.py -q` -> `44 passed`
- `git diff --check` -> clean

## No-Signal-Loss Status

This is a guard and custody fix only. It makes no score claim, does not launch
or retire any lane, and does not alter existing exact-eval artifacts. The
relevant signal is that Cathedral/autopilot exact dispatch can now depend on the
shared authority helper while the dispatcher and terminal-evidence audit avoid
two false-negative custody classifications.
