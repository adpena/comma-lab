# JCSP True-Runtime Bridge Probe - 2026-05-06 Codex

## Scope

This tranche advances the `JCSK` local skeleton toward real runtime
consumption without making a dispatch-ready claim. The submission runtime now
has a stdlib-only probe for `jcsp.bin` that runs from
`submissions/robust_current/inflate.sh` before branch dispatch.

## Runtime Contract

- Real `JCSP` member: parse the container header, per-stream names,
  `codec_kind`, payload lengths, payload SHA-256s, and payload magic. Write a
  deterministic `jcsp_submission_runtime_bridge_probe_v1` manifest.
- Local `JCSK` preview member: detect and refuse it as
  `jcsp_local_skeleton_not_submission_runtime_container`.
- Unknown or malformed `jcsp.bin`: fail closed with a deterministic refusal
  manifest.
- No stream decoder or frame-emission path is implemented in this tranche.

## Non-Dispatch Status

`ready_for_runtime_loader=true` is allowed only for structurally valid real
`JCSP` bytes. The runtime contract still records:

- `consumes_required_member=false`
- `ready_for_submission_runtime_consumption=false`
- `ready_for_exact_eval_dispatch=false`
- `submissions_robust_current_jcsp_bin_consumption_missing`
- `jcsp_stream_decode_emit_frames_missing`
- `exact_cuda_auth_eval_missing`

## Verification

Focused tests cover deterministic manifest proof for real `JCSP`, refusal of
`JCSK` preview members, CLI fail-closed behavior when `jcsp.bin` is present,
and the `inflate.sh` hook ordering before runtime branch dispatch. No GPU,
remote dispatch, lane claim, or score claim was used.
