# Codec-Op Search Hardening - Worker H - 2026-05-07

## Scope

Owned files only:

- `tools/codec_op_optuna_search.py`
- `tools/codec_op_cma_search.py`
- `src/tac/tests/test_codec_op_search_full_decode_guard.py`
- `.omx/research/codec_op_search_hardening_20260507_worker_h.md`

## Hardening Changes

- Default decode success now requires every tensor key in the input contract to
  be decoded as a tensor with matching shape before fitness/Pareto success is
  allowed.
- Partial decode is permitted only with an explicit
  `--allow-partial-decode --partial-decode-waiver-reason ...` waiver. Waived
  evaluations record `partial_decode_waived`, the waiver reason, coverage
  status, matched/decoded keys, and remain non-dispatchable planning evidence.
- CMA search now evaluates the init/default baseline at `eval_idx=0` and
  treats `--max-evals` as a hard cap. A final short CMA generation is recorded
  without `tell()` instead of exceeding the requested budget.
- Optuna reports now explicitly record the enqueued init/default baseline.
- Reports now include custody/planning fields: generated timestamp, evidence
  grade, requested max evals, max-eval semantics, baseline status/params,
  state-dict path/bytes/SHA-256, tensor contract, decode coverage policy, and
  explicit non-dispatchability / no exact CUDA / no archive fields.
- Ledger rows now include baseline role, decode coverage details, exact-CUDA
  false, archive SHA/bytes null, promotion false, and partial-decode waiver
  blockers when applicable.

## Evidence Semantics

These tools still produce CPU-only CodecOp search evidence:

- `score_claim=false`
- `dispatchable=false`
- `ready_for_exact_eval_dispatch=false`
- `score_affecting_payload_changed=false`
- `charged_bits_changed=false`
- `exact_cuda_auth_eval=false`

The output remains planning-only until a byte-closed archive exists and exact
CUDA auth eval validates the exact archive bytes.

## Verification

Focused ruff/pytest commands are intended for:

- `tools/codec_op_cma_search.py`
- `tools/codec_op_optuna_search.py`
- `src/tac/tests/test_codec_op_search_full_decode_guard.py`

