# 2026-05-10 Preflight SourceIndex Scan Migration

research_only=true

Scope: wall-clock optimization for strict preflight source scanners. Owned files:
`src/tac/preflight.py`, `tests/test_preflight_source_index_equivalence.py`, and
this ledger.

Migrated checks:

- `check_phase3_dispatch_gate_fail_closed`: SourceIndex candidate filtering on
  `Phase3DispatchGate`, with cached text reads.
- `check_setup_first_seen_uses_transactional_update_inside_lock`: SourceIndex
  candidate filtering on `_load_` + `_save_`, with cached text reads.
- `check_packet_compiler_no_op_proof_promotes_to_blocker`: SourceIndex
  candidate filtering on `_build_no_op_proof` + `blockers`, with cached text
  reads.
- `check_paid_job_register_before_submit`: SourceIndex candidate filtering on
  `Job.run`, with cached text reads.
- `check_setup_first_seen_no_split_transactions`: SourceIndex intersection of
  observed-first-seen and left-first-seen helper candidates, with cached text
  reads.

Equivalence proof:

- Added one temp-repo equivalence test for each migrated check in
  `tests/test_preflight_source_index_equivalence.py`.
- `py_compile`: passed for `src/tac/preflight.py` and the focused test file.
- Focused pytest: `17 passed in 0.60s` on the final focused run.

Live timing sample on current checkout, strict false, verbose false:

- No SourceIndex context total: 2.9490s, all migrated checks returned 0
  violations.
- SourceIndex context total: 1.6473s, all migrated checks returned 0 violations.
- SourceIndex context after `_prewarm_preflight_source_index`: 1.0686s, all
  migrated checks returned 0 violations.

Unified-solver hook disposition:

- Sensitivity-map contribution: N/A; no model, tensor, or component sensitivity
  signal changed.
- Pareto constraint: N/A; this is preflight infrastructure, not a scoring
  candidate.
- Bit-allocator hook: N/A; no charged-byte allocation behavior changed.
- Cathedral autopilot dispatch hook: N/A; no dispatch tool or lane queue was
  modified.
- Continual-learning posterior update: N/A; no empirical score anchor was
  produced.
- Probe-disambiguator: N/A; no competing design interpretation was introduced.

Residual direct-scan hotspots remain in later catalog checks, especially newer
Lightning/provider/state-helper guards that still read candidate files directly
after `_iter_python_files`, plus several shell-script `remote_lane_*.sh` scans.
Those are follow-up SourceIndex migration candidates.
