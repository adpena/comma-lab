# Exact Dispatch Guard Hardening - 2026-05-10

Scope:
- Hardened `tools/promote_optimizer_candidate_for_exact_eval.py`.
- Added focused CLI tests for skip-flag fail-close, active same-lane claims,
  and `lane_` alias conflicts.
- No dispatch, no score claim, no queue promotion artifact created.

Guard change:
- `--skip-active-claim-check` now fails closed for exact-eval dispatch
  promotion. There is no dispatch-ready skip path in this tool.
- The promotion CLI now requires a readable dispatch-claim ledger before
  writing an exact-ready queue.
- Active claim checks now cover both canonical and `lane_` alias spellings
  before promotion.
- Terminal same-archive evidence is also rechecked across alias spellings
  before the exact-ready queue is written.

Verification:
- `.venv/bin/python -m pytest tests/test_promote_optimizer_candidate_for_exact_eval_cli.py -q`
- `.venv/bin/python -m py_compile tools/promote_optimizer_candidate_for_exact_eval.py tests/test_promote_optimizer_candidate_for_exact_eval_cli.py`

Unified solver wire-in:
- sensitivity_map: N/A - dispatch custody guard, no model sensitivity signal.
- pareto_constraint: N/A - no candidate frontier or objective term changed.
- bit_allocator: N/A - no bit allocation policy changed.
- cathedral_autopilot_dispatch: strengthened by fail-closing exact-ready
  queue promotion before provider dispatch can consume it.
- continual_learning_posterior: N/A - no empirical score anchor produced.
- probe_disambiguator: N/A - no competing defensible runtime behavior shipped;
  exact-dispatch promotion always requires the active claim ledger.
