# Codex Session Summary - PR101 OP-7 Exact-Eval Score Response

Timestamp: 2026-05-19T11:05:00Z
Agent: codex:gpt-5.5

## What Changed

- Completed paired Modal exact eval for PR101 OP-7 raw-byte-delta candidate on `[contest-CPU]` and `[contest-CUDA]`.
- Fixed `tac.scorer_response_probe` so Modal CPU auth-eval rows with `inflate_device_policy=auto` normalize to `cpu(auto)` only when the full CPU-authority tuple is present: `axis=contest_cpu`, Linux provenance, `device=cpu`, `cuda_available=false`, and `--device cpu`.
- Hardened negative tests so the exception does not apply to `cuda_available=true`, `--device cuda`, non-Linux provenance, non-`contest_cpu`, or generic `auto` rows.
- Hardened `tac.master_gradient_pr101_score_response_matrix` so refreshed matrices do not keep stale missing-claim/missing-result blockers after exact-eval artifacts and score-response outputs exist.
- Updated score-response markdown authority wording from misleading `dispatch_attempted=false` to probe-local authority fields.
- Backfilled the four OP-7 Modal call IDs into `.omx/state/modal_call_id_ledger.jsonl`.
- Recorded the measured negative in `.omx/state/probe_outcomes.jsonl` and `reports/cathedral_autopilot_evidence.jsonl`.

## Exact Result

The measured candidate `30826b37093ee3af9512a1b46bd0b569fecbc4ccf75b8ff2dd746de113a5144a` regressed both exact axes at unchanged bytes (`178258`):

- `[contest-CPU]`: `0.1928480127024255 -> 0.19454175105500707`, delta `+0.0016937383525815752`.
- `[contest-CUDA]`: `0.22634945874409151 -> 0.2277121199933223`, delta `+0.00136266124923079`.

This retires only `pr101-op7-rank1-raw-byte-delta-same-length`. It does not retire master-gradient, per-pair gradient, null-space, procedural packet, or deterministic-byte-derivation families.

## Verification

- `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q src/tac/tests/test_scorer_response_probe.py src/tac/tests/test_auth_eval_roundtrip_matrix.py src/tac/tests/test_master_gradient_pr101_score_response_matrix.py` -> `33 passed`.
- `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m ruff check ...` -> passed.
- `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile ...` -> passed.
- CPU score-response probe -> `SCORE_REGRESSION`, no blockers.
- CUDA score-response probe -> `SCORE_REGRESSION`, no blockers.

## Follow-Up Routing

Next useful work is not another rerun of this archive. Route toward:

- smaller trust-region byte perturbations;
- SegNet-boundary-preserving projection;
- per-pair/per-region perturbation instead of raw stream replacement;
- procedural or deterministic packet compiler variants with a new archive SHA and paired exact eval.
