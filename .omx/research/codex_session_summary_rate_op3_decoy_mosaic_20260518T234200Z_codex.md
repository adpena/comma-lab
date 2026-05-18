# Codex Session Summary: RATE-OP-3 Decoy/Mosaic Residual Basis

timestamp_utc: 2026-05-18T23:42:00Z
agent: codex
task_id: rate_attack_op_3_decoy_mosaic_residual_basis
status: implementation_plus_probe_complete

## Landed

- `src/tac/contest_exploits/decoy_mosaic_residual_basis.py`
- `tools/build_rate_attack_op3_decoy_mosaic_residual_basis_probe.py`
- `src/tac/contest_exploits/tests/test_decoy_mosaic_residual_basis.py`
- `src/tac/contest_exploits/__init__.py` exports and module registry update
- `experiments/results/rate_attack_op3_decoy_mosaic_probe_20260518T233642Z/route_entropy_report.json`
- `.omx/research/codex_findings_rate_op3_decoy_mosaic_20260518T234200Z_codex.md`

## Result

OP3 is now wired as a planning-only Cathedral-visible artifact. The current
fec6 probe found a balanced 4-route cheap-probe partition over 50 advisory
pairs and a 278-byte canonical route-table model, but no charged route-label
payload, no specialist-head byte proof, and no monolith control. All rows remain
non-promotable and non-dispatchable.

## Verification

- `.venv/bin/python -m pytest src/tac/contest_exploits/tests/test_decoy_mosaic_residual_basis.py`
  - 4 passed
- `.venv/bin/python -m pytest src/tac/contest_exploits/tests/test_decoy_mosaic_residual_basis.py src/tac/contest_exploits/tests/test_tropical_argmax_boundary_grammar.py`
  - 7 passed
- `.venv/bin/ruff check src/tac/contest_exploits/decoy_mosaic_residual_basis.py src/tac/contest_exploits/tests/test_decoy_mosaic_residual_basis.py tools/build_rate_attack_op3_decoy_mosaic_residual_basis_probe.py src/tac/contest_exploits/__init__.py`
  - passed
- Cathedral JSONL loader smoke on generated OP3 artifact:
  - 3 rows loaded
  - all `score_claim=false`
  - all `promotion_eligible=false`
  - all `ready_for_exact_eval_dispatch=false`

## Open Blockers

- `op3_export_first_route_table_grammar_required`
- `op3_single_monolith_control_missing`
- `op3_specialist_heads_missing`
- `op3_specialist_head_bytes_unmeasured`
- `op3_known_overhead_or_materiality_floor_not_cleared`
- `requires_runtime_consumption_proof`
- `requires_full_frame_inflate_parity`
- `requires_exact_cuda_auth_eval`

## Next Action

Continue the canonical pending queue. The highest OP3-specific next artifact is
an export-first route-table grammar prototype that writes deterministic
2-bit route labels inside `0.bin`, plus a same-family monolith-control report.
