# PR91 HPM1 Runtime Contract Gate

Date: 2026-05-06

## Status

Implemented guarded readiness intake. This does not recover PR91 HPM1 decode
parity by itself and does not allow exact-eval dispatch.

## What Landed

- `tac.pr91_hpm1_runtime_contract.audit_pr91_hpm1_runtime_contract(...)` now
  resolves the HPAC device contract when every visible HPAC decompress call
  passes literal CPU and there are no ambient-device contradictions.
- `tac.pr91_hpm1_readiness.audit_pr91_hpm1_readiness(...)` now accepts an
  optional `pr91_hpm1_decode_reencode_parity_v1` report.
- `tools/audit_pr91_hpm1_readiness.py` exposes `--parity-report`.

## Parity Report Contract

The readiness audit accepts a parity report only if it proves all of:

- `score_claim=false`
- `dispatch_attempted=false`
- archive SHA-256 matches the audited archive
- HPM1 mask SHA-256 matches the audited payload
- device contract resolves to CPU
- full decode passed for 600 frames
- decoded mask SHA-256 is recorded
- byte-exact re-encode SHA-256 equals the source HPM1 mask segment
- runtime loader was sidecar-free and did not use fallback

When that report is valid, the readiness audit can clear:

- `full_hpm1_decode_600_frames`
- `byte_exact_hpm1_reencode`
- `runtime_hpm1_loader_without_sidecars`

It still keeps `exact_cuda_auth_eval_after_parity` blocked. Exact CUDA auth eval
must be run on a byte-closed candidate before any score claim or dispatch-ready
state.

## Remaining Work

Produce the real parity report from the recovered HPM1 decoder/re-encoder. The
current code only defines and enforces the proof contract.
