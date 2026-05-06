# JCSP Byte-Closure Tranche - 2026-05-06 Codex

## Scope

This tranche advances JCSP beyond metadata dry-run by writing a deterministic
local archive member skeleton. It is not a score claim, not a GPU dispatch, and
not an exact-eval-ready runtime path.

## Artifact Contract

- Container magic/version: `JCSK` / `1`, intentionally distinct from runtime
  `JCSP`.
- Archive member: deterministic one-member ZIP carrying `jcsp.bin`.
- Payload: canonical JSON manifest with per-stream source metadata, qint/raw
  preview bytes as hex, byte counts, SHA-256s, score-marginal artifact custody,
  and stream-spec manifest SHA.
- Runtime status: `ready_for_runtime_loader=false`,
  `ready_for_submission_runtime_consumption=false`,
  `ready_for_exact_eval_dispatch=false`.

## Blockers Preserved

- `local_skeleton_preview_only_not_runtime_payload`
- `full_codec_payload_not_encoded`
- `runtime_loader_parity_missing`
- `jcsp_local_skeleton_not_submission_runtime_container`
- `submissions_robust_current_jcsp_bin_consumption_missing`
- `strict_preflight_proof_missing`
- `exact_cuda_auth_eval_missing`

## Verification Plan

Focused tests cover deterministic archive bytes, manifest SHA/byte custody,
non-dispatch readiness flags, pipeline-side artifact writing, and preservation
of fail-closed blockers. No lane claim, remote job, GPU eval, or score path is
used by this tranche.
