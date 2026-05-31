# Codex Findings: Entropy Adapter Contract Review

UTC: 2026-05-31T01:01:17Z

## Scope

Reviewed the archive/entropy repair path around the new range/ANS prototype
surface:

- `.omx/research/repair_multi_archive_autonomous_stage_disciplined_psv3_fec6_20260528T0648Z/**`
- `src/tac/optimization/repair_entropy_coder_runtime_adapters.py`
- `src/tac/optimization/archive_bound_candidate_contract.py`
- `src/tac/optimization/archive_bound_candidate_adapter_spine.py`
- `src/tac/optimization/repair_family_byte_transform_executor.py`

## Finding

The range/ANS runtime adapter manifest was too strong. It correctly described a
stdlib decode helper with no scorer access, network access, or sidecar fetch,
but it also marked `contest_runtime_decoder_adapter_ready=true`. The actual
repair-family prototype rows correctly treated these adapters as
`member_decode_helper_only` and not contest-runtime-integrated, so the manifest
was the stale duplicate readiness surface.

This is score-lowering relevant because acquisition and exact handoff must be
able to trust one contract field. A member-level decoder proof is useful
pipeline signal and should open the smallest byte-closed materializer task, but
it must not masquerade as an integrated contest inflate/runtime adapter.

## Remediation

Landed the contract-level fix:

- `entropy_coder_runtime_adapter_manifest()` now declares
  `runtime_adapter_scope=member_decode_helper_only`,
  `contest_runtime_decoder_adapter_ready=false`, and the explicit blocker
  `contest_runtime_decoder_adapter_integration_missing`.
- `archive_bound_candidate_contract_fields_for_row()` and
  `build_archive_bound_candidate_contract()` now consume nested
  `runtime_adapter_manifest.contest_runtime_decoder_adapter_ready` directly,
  so future real runtime adapters do not need a parallel top-level readiness
  field.
- Added regression coverage for both behaviors.

## Current Verdict

`range_coder_lzma_prototype` and `ans_coder_rans_prototype` remain valuable
byte-closed, receiver-proven prototype signals. They are not exact-dispatch or
score authority until a real contest inflate/runtime adapter consumes the coded
member under the official archive/runtime path.

## Remaining Follow-Up

The next score-lowering integration step is to replace member-scope prototypes
with contest-runtime adapter variants that rewrite the candidate member and
prove `inflate.sh archive_dir output_dir file_list` consumes the decoded stream,
then feed those rows through the same archive-bound candidate adapter spine.
