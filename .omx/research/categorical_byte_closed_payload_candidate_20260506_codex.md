# Categorical Byte-Closed Payload Candidate - 2026-05-06

Evidence grade: local payload custody / readiness hardening
Score claim: false
Dispatch attempted: false
Ready for exact eval dispatch: false

## Context

The categorical/QMA9/CLADE-SPADE/openpilot lane had a deterministic fixture and
strict manifest audit, but no local candidate artifact carrying a real
categorical payload member. This patch moves the lane one step forward by
packaging the PR91 HPM1 mask segment as a charged categorical payload with the
canonical class codebook and a charged runtime consumer proof skeleton.

## Artifact

Output directory:
`experiments/results/categorical_openpilot_payload_candidate_20260506_codex/`

Generated files:

- `archive.zip` - ignored rebuildable local archive, 152235 bytes,
  SHA-256 `106af3ed6917d6115586463ef35e43119add5db002fd19f3da7cd8065a63eb8d`
- `archive_member_manifest.json`
- `candidate.json`
- `construction_plan.json`
- `readiness.json`
- `summary.json`

Charged archive members:

| member | role | bytes | sha256 |
|---|---|---:|---|
| `categorical_payload.bin` | `categorical_payload` | 145087 | `a4ed57ff0af1d8c914f004de165aeead50ec8dd61e99b0afdfbfa2d1e7fd9fcc` |
| `class_codebook.json` | `decoder_table` | 1878 | `6db870e7e05cc310ca25897e8d3ccac4f6fff06ad22ea57c157dbf8086ba29d0` |
| `inflate.sh` | `inflate_entrypoint` | 233 | `96c609ffb891c2717ccc055f81a7255a7946383f8f54fca2f1b41fdb88f26062` |
| `runtime_consumer.py` | `decoder_or_runtime_consumer` | 2690 | `4fb6740b3e859068dbdbe159545c49b927a8b017b6046412ef5edddcb5fcba6d` |
| `runtime_consumer_proof_skeleton.json` | `runtime_consumer_proof` | 1731 | `72cd6236aeff5b62657c99be07ee7f8d3d8857759daff74601976e86ee3e21f0` |

Payload source:

- PR91 public archive:
  `experiments/results/public_pr91_intake_20260504_codex/archive.zip`
- Source archive SHA-256:
  `4c16d04c746c981feb902e4dd508ffadaf3615e532d351993c3d2f6eccda1b4f`
- Source member `x` SHA-256:
  `5c213c61cc4d29b62286063bfdcb97e812af6b06c0021aeaecc8bc46644e17bf`
- Extracted mask segment magic: `HPM1`

## Dispatch Boundary

`readiness.json` is intentionally blocked. The archive is byte-closed, but it
does not prove semantic decode/re-encode parity or runtime output parity. The
runtime member is a fail-closed skeleton that verifies charged member presence
and exits non-dispatchable.

Current readiness blockers:

- `no_op_control_not_passed:decode_reencode_identity_control`
- `no_op_control_not_passed:label_permutation_fail_closed_control`
- `no_op_control_not_passed:runtime_consumes_conditioning_control`
- `runtime_loader_parity_not_passed`
- `decode_reencode_parity_not_passed`
- `decode_reencode_full_decode_not_proven`
- `decode_reencode_byte_exact_reencode_not_proven`

## Verification

Focused local verification:

```text
.venv/bin/python -m pytest src/tac/tests/test_categorical_candidate_readiness.py src/tac/tests/test_build_categorical_candidate_payload.py src/tac/tests/test_build_categorical_candidate_fixture.py src/tac/tests/test_materialize_comma_lab_public_export.py -q
```

Result: 31 passed, with the expected duplicate-ZIP warning in the negative
custody test.

## Next Required Work

- Recover full 600-frame HPM1 decode for `categorical_payload.bin`.
- Prove byte-exact decode/re-encode back to
  `a4ed57ff0af1d8c914f004de165aeead50ec8dd61e99b0afdfbfa2d1e7fd9fcc`.
- Replace the runtime skeleton with a charged decoder that consumes only
  archive members.
- Only after those local gates pass, claim a lane before any exact CUDA auth
  eval. No lane claim or dispatch happened in this patch.

## Charged Label-Prior Manifest Addendum - 2026-05-07

Evidence grade: local byte-closed contract hardening
Score claim: false
Dispatch attempted: false
Ready for exact eval dispatch: false

Worker B added a charged `label_prior_payload_manifest.json` archive member so
the categorical/openpilot conditioning priors are now bound to deterministic
archive bytes instead of living only in `candidate.json`.

Regenerated archive custody:

- `archive.zip`: 160400 bytes, SHA-256
  `3455c82708b1d628e17fb21cf3ccb334a4375e023a80217681c10912224881ac`
- `label_prior_payload_manifest.json`: 5403 bytes, SHA-256
  `0be3c41d3f0f83eafe7f2c38a8d257a4dcb815973b22595b003c7592d1107f0d`
- `runtime_consumer.py`: 5110 bytes, SHA-256
  `d4473f3f031e1949bc9f90be49c54e759a4bd0039921dc81bb4667d6140576a8`

Readiness now reports `label_prior_payload_manifest.accepted=true` and still
reports `ready_for_exact_eval_dispatch=false`. Current blockers remain the
intended local proof gates:

- `no_op_control_not_passed:decode_reencode_identity_control`
- `no_op_control_not_passed:label_permutation_fail_closed_control`
- `no_op_control_not_passed:runtime_consumes_conditioning_control`
- `runtime_loader_parity_not_passed`
- `runtime_execution_proof_artifact_missing`
- `decode_reencode_parity_not_passed`
- `decode_reencode_full_decode_not_proven`
- `decode_reencode_byte_exact_reencode_not_proven`
- `decode_reencode_independent_proof_artifact_missing`
- `exact_eval_dispatch_requirements_missing`

No remote dispatch, lane claim, CUDA eval, or score claim happened in this
addendum.
